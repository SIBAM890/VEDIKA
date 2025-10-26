import os
import re
import time
import json
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from typing import Set, List, Deque
from pathlib import Path
from langchain_core.documents import Document
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PyPDF2 import PdfReader

# Import config and logger from our project structure
from backend.config import BASE_URL, CRAWL_DELAY, USER_AGENT, DATA_DIR
from backend.utils.logger import get_logger

logger = get_logger("scraper")

# --- Setup directories defined relative to DATA_DIR ---
PDF_DIR = DATA_DIR / "pdfs"
IMG_DIR = DATA_DIR / "images"
JSON_DIR = DATA_DIR / "json"
for d in [PDF_DIR, IMG_DIR, JSON_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class SSUWebCrawler:
    """Powerful recursive scraper for Sri Sri University website, integrated."""

    def __init__(self):
        self.root_url = BASE_URL.rstrip("/")
        self.base_domain = urlparse(self.root_url).netloc
        self.delay = CRAWL_DELAY
        self.visited: Set[str] = set()
        self.to_visit: Deque[str] = deque([self.root_url])
        self.start_time = time.time()

        # Robust session with retry logic
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    # --- URL Filtering ---
    def _is_valid_url(self, url: str) -> bool:
        """Checks if URL is internal and not a special type."""
        parsed = urlparse(url)
        if parsed.netloc != self.base_domain:
            return False
        if url.startswith(("mailto:", "tel:", "javascript:")):
            return False
        # Let the extension check handle file types later
        return True

    def _get_extension_type(self, url: str) -> str | None:
        """Determines the content type based on URL extension."""
        path = urlparse(url).path.lower()
        if path.endswith(".pdf"):
            return "pdf"
        if path.endswith(".json"):
            return "json"
        if re.search(r"\.(jpg|jpeg|png|gif|svg|webp)$", path):
            return "image"
        # Assume HTML otherwise, filter non-HTML later in fetch
        return "html"

    # --- Data Fetchers ---
    def _get_html(self, url: str) -> str | None:
        """Fetches and validates HTML content."""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status() # Raise error for bad status codes
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type:
                logger.debug(f"Skipping non-HTML URL: {url} (Content-Type: {content_type})")
                return None
            return response.text
        except requests.RequestException as e:
            logger.error(f"⚠️ Failed to fetch HTML from {url}: {e}")
            return None

    def _get_pdf_text(self, url: str) -> tuple[str | None, Path | None]:
        """Downloads PDF, saves it, and extracts text."""
        pdf_path = None
        try:
            response = self.session.get(url, timeout=30, stream=True) # Stream for large files
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "application/pdf" not in content_type:
                logger.warning(f"Skipping non-PDF URL marked as PDF: {url} (Content-Type: {content_type})")
                return None, None

            # Generate filename based on URL path
            url_path = urlparse(url).path.strip('/')
            filename = url_path.split('/')[-1] if url_path else f"doc_{int(time.time()*1000)}.pdf"
            if not filename.lower().endswith('.pdf'):
                filename += ".pdf"
            pdf_path = PDF_DIR / filename

            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract text
            reader = PdfReader(pdf_path)
            text = " ".join([page.extract_text() or "" for page in reader.pages])
            cleaned_text = re.sub(r'\s+', ' ', text).strip()
            logger.info(f"📄 Successfully extracted text from PDF: {pdf_path.name}")
            return cleaned_text, pdf_path
        except Exception as e:
            logger.warning(f"Failed to process PDF from {url}: {e}")
            if pdf_path and pdf_path.exists():
                os.remove(pdf_path) # Clean up failed download
            return None, None

    def _get_json_content(self, url: str) -> tuple[str | None, Path | None]:
        """Downloads JSON, saves it, and returns its string representation."""
        json_path = None
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "application/json" not in content_type:
                logger.warning(f"Skipping non-JSON URL marked as JSON: {url} (Content-Type: {content_type})")
                return None, None

            data = response.json()
            # Generate filename
            url_path = urlparse(url).path.strip('/')
            filename = url_path.split('/')[-1] if url_path else f"data_{int(time.time()*1000)}.json"
            if not filename.lower().endswith('.json'):
                filename += ".json"
            json_path = JSON_DIR / filename

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            json_string = json.dumps(data) # Convert JSON data to string for Document
            logger.info(f"💾 Successfully saved JSON: {json_path.name}")
            return json_string, json_path
        except Exception as e:
            logger.warning(f"Failed to fetch/parse/save JSON from {url}: {e}")
            if json_path and json_path.exists():
                os.remove(json_path) # Clean up
            return None, None

    def _save_image(self, url: str) -> Path | None:
        """Downloads and saves an image file."""
        img_path = None
        try:
            response = self.session.get(url, timeout=20, stream=True)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if not content_type.startswith("image/"):
                logger.warning(f"Skipping non-image URL marked as image: {url} (Content-Type: {content_type})")
                return None

            # Generate filename
            url_path = urlparse(url).path.strip('/')
            filename = url_path.split('/')[-1] if url_path else f"img_{int(time.time()*1000)}"
            ext = os.path.splitext(filename)[1] or f".{content_type.split('/')[-1]}" # Get ext from filename or content type
            if not ext or len(ext) > 5: # Basic check for valid extension
                 ext = ".jpg" # Default fallback
            filename = os.path.splitext(filename)[0] + ext # Ensure correct extension
            img_path = IMG_DIR / filename

            with open(img_path, "wb") as f:
                 for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"🖼️ Successfully saved image: {img_path.name}")
            return img_path
        except Exception as e:
            logger.warning(f"Failed to save image {url}: {e}")
            if img_path and img_path.exists():
                os.remove(img_path) # Clean up
            return None

    # --- HTML Content Extraction ---
    def _extract_content_and_links(self, soup: BeautifulSoup, url: str) -> tuple[str, str, List[str]]:
        """Extracts title, clean main text, and new internal links."""
        title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled Page"

        main_content = None
        # Use your effective selectors
        selectors = [
            "main", "article", "section.content", "div.entry-content",
            "div.main-content", "div#content", "div.td-post-content"
        ]
        for sel in selectors:
            candidate = soup.select_one(sel)
            if candidate and len(candidate.get_text(strip=True)) > 100:
                main_content = candidate
                break
        
        if not main_content:
            main_content = soup.body
            if main_content:
                # Be more aggressive removing noise if using body
                for tag in main_content(["header", "footer", "nav", "aside", "form", "script", "style", "noscript", "svg", ".sidebar", "#sidebar", ".related-posts"]):
                    if tag: tag.decompose()

        if not main_content:
            logger.warning(f"Could not find suitable main content for: {url}")
            return title, "", []

        # Remove remaining script/style before text extraction
        for tag in main_content(["script", "style", "noscript"]):
             if tag: tag.decompose()

        # Get text, clean whitespace, remove common irrelevant phrases
        text = main_content.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        ignore_phrases = ["404 Not Found", "Page not found", "Coming Soon", "This site uses cookies", "Back to top", "Search \u00bb", "Skip to content"]
        for phrase in ignore_phrases:
            text = text.replace(phrase, "")
        text = text.strip()

        # Extract links
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            # Resolve relative URLs, remove fragments, ensure trailing slash for consistency
            full_url = urljoin(url, href).split("#")[0].rstrip('/')
            if full_url and self._is_valid_url(full_url): # Check if URL is valid and not empty
                links.append(full_url)

        return title, text, list(set(links)) # Unique links

    # --- Main Crawl Loop ---
    def crawl(self) -> List[Document]:
        """Executes the full website crawl."""
        logger.info(f"🚀 Starting powerful crawl from: {self.root_url}")
        documents: List[Document] = []
        page_count = 0

        while self.to_visit:
            current_url = self.to_visit.popleft()
            normalized_url = current_url.rstrip('/') # Normalize for visited check

            if normalized_url in self.visited:
                continue

            self.visited.add(normalized_url)
            page_count += 1
            logger.info(f"🔗 Crawling #{page_count}: {current_url}")

            ext_type = self._get_extension_type(current_url)
            content = None
            metadata = {"source": current_url}
            new_links = []

            # --- Handle different content types ---
            if ext_type == "pdf":
                content, file_path = self._get_pdf_text(current_url)
                if content:
                     metadata["type"] = "pdf"
                     if file_path: metadata["local_path"] = str(file_path)

            elif ext_type == "json":
                content, file_path = self._get_json_content(current_url)
                if content:
                    metadata["type"] = "json"
                    if file_path: metadata["local_path"] = str(file_path)

            elif ext_type == "image":
                file_path = self._save_image(current_url)
                if file_path:
                    # For images, content is just a placeholder
                    content = f"[Image saved locally: {file_path.name}]"
                    metadata["type"] = "image"
                    metadata["local_path"] = str(file_path)
            
            else: # Assume HTML
                html = self._get_html(current_url)
                if html:
                    try:
                        soup = BeautifulSoup(html, "html.parser")
                        title, content, new_links = self._extract_content_and_links(soup, current_url)
                        metadata["title"] = title
                        metadata["type"] = "html"
                    except Exception as e:
                        logger.error(f"Error parsing HTML for {current_url}: {e}")
                        content = None # Ensure content is None on error
                
            # --- Create Document if content is valid ---
            if content and len(content) > 100: # Check content length
                documents.append(Document(page_content=content, metadata=metadata))
            elif ext_type not in ["pdf", "json", "image"]: # Log skipped HTML only if not handled file type
                 logger.debug(f"Skipped page (short or no content): {current_url}")

            # --- Add new links to queue ---
            for link in new_links:
                normalized_link = link.rstrip('/')
                if normalized_link not in self.visited and link not in self.to_visit:
                    self.to_visit.append(link)

            time.sleep(self.delay) # Politeness delay

        duration = time.time() - self.start_time
        logger.info(f"✅ Crawl complete in {duration:.2f}s — Processed {len(self.visited)} URLs, created {len(documents)} documents.")
        return documents


# --- Entry function for ingest.py ---
def scrape_all() -> List[Document]:
    """Initializes and runs the full website crawler."""
    crawler = SSUWebCrawler()
    return crawler.crawl()