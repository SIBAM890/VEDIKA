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
from PyPDF2 import PdfReader, errors as pdf_errors # Import specific error

# Import config and logger
from backend.config import BASE_URL, CRAWL_DELAY, USER_AGENT, DATA_DIR
from backend.utils.logger import get_logger

logger = get_logger("scraper")

# --- Setup asset directories ---
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

        # Robust session with retry
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _is_valid_url(self, url: str) -> bool:
        """Checks if URL is internal and not a special type."""
        parsed = urlparse(url)
        # Ensure scheme is http or https
        if parsed.scheme not in ['http', 'https']:
            return False
        if parsed.netloc != self.base_domain:
            return False
        if url.startswith(("mailto:", "tel:", "javascript:")):
            return False
        return True

    def _get_extension_type(self, url: str) -> str:
        """Determines the content type based on URL extension, defaulting to html."""
        path = urlparse(url).path.lower()
        if path.endswith(".pdf"):
            return "pdf"
        if path.endswith(".json"):
            return "json"
        if re.search(r"\.(jpg|jpeg|png|gif|svg|webp)$", path):
            return "image"
        # Default to HTML, handle content-type check during fetch
        return "html"

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

            url_path = urlparse(url).path.strip('/')
            # Sanitize filename to prevent directory traversal or invalid characters
            filename_base = url_path.split('/')[-1] if url_path else f"doc_{int(time.time()*1000)}"
            filename_safe = re.sub(r'[^\w\-_\.]', '_', filename_base)
            filename = filename_safe if filename_safe.lower().endswith('.pdf') else filename_safe + ".pdf"
            pdf_path = PDF_DIR / filename

            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)

            reader = PdfReader(pdf_path)
            if reader.is_encrypted:
                logger.warning(f"Skipping encrypted PDF: {url}")
                # Return path even if encrypted, but no text
                return "[Encrypted PDF content - cannot extract text]", pdf_path

            text = " ".join([page.extract_text() or "" for page in reader.pages if hasattr(page, 'extract_text')])
            cleaned_text = re.sub(r'\s+', ' ', text).strip()

            if cleaned_text:
                logger.info(f"📄 Successfully extracted text from PDF: {pdf_path.name}")
            else:
                 logger.warning(f"📄 No text extracted from PDF (possibly image-based or empty): {pdf_path.name}")
                 # Provide placeholder text if extraction yields nothing
                 cleaned_text = "[PDF content - No text extracted or image-based]"

            return cleaned_text, pdf_path
        except pdf_errors.PdfReadError as pe:
             logger.warning(f"Failed to read PDF (corrupted or invalid format?) from {url}: {pe}")
             if pdf_path and pdf_path.exists(): os.remove(pdf_path)
             return None, None
        except Exception as e:
            logger.warning(f"Failed to process PDF from {url}: {e}")
            if pdf_path and pdf_path.exists(): os.remove(pdf_path)
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
            url_path = urlparse(url).path.strip('/')
            filename_base = url_path.split('/')[-1] if url_path else f"data_{int(time.time()*1000)}"
            filename_safe = re.sub(r'[^\w\-_\.]', '_', filename_base) # Sanitize
            filename = filename_safe if filename_safe.lower().endswith('.json') else filename_safe + ".json"
            json_path = JSON_DIR / filename

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Convert JSON data to a readable string format for the Document
            json_string = json.dumps(data, indent=2)
            logger.info(f"💾 Successfully saved and stringified JSON: {json_path.name}")
            return json_string, json_path
        except json.JSONDecodeError as je:
            logger.warning(f"Failed to parse JSON from {url}: {je}")
            if json_path and json_path.exists(): os.remove(json_path)
            return None, None
        except Exception as e:
            logger.warning(f"Failed to fetch/save JSON from {url}: {e}")
            if json_path and json_path.exists(): os.remove(json_path)
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

            url_path = urlparse(url).path.strip('/')
            filename_base = url_path.split('/')[-1] if url_path else f"img_{int(time.time()*1000)}"
            filename_safe = re.sub(r'[^\w\-_\.]', '_', filename_base) # Sanitize
            # More robust extension handling
            ext_from_path = os.path.splitext(filename_safe)[1].lower()
            ext_from_content = f".{content_type.split('/')[-1].split(';')[0]}" if '/' in content_type else ''
            ext = ext_from_path if ext_from_path and len(ext_from_path) <=5 else ext_from_content # Prefer path ext if valid
            if not ext or len(ext) > 5 or ext == '.': ext = ".jpg" # Default fallback
            filename = os.path.splitext(filename_safe)[0] + ext
            img_path = IMG_DIR / filename

            with open(img_path, "wb") as f:
                 for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
            logger.info(f"🖼️ Successfully saved image: {img_path.name}")
            return img_path
        except Exception as e:
            logger.warning(f"Failed to save image {url}: {e}")
            if img_path and img_path.exists(): os.remove(img_path)
            return None

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
                 for tag_name in ["header", "footer", "nav", "aside", "form", "script", "style", "noscript", "svg"]:
                     for tag in main_content.find_all(tag_name):
                         if tag: tag.decompose()
                 for noisy_selector in [".sidebar", "#sidebar", ".related-posts", ".comments-area", ".breadcrumb", ".social-links", ".cookie-notice"]:
                     for tag in main_content.select(noisy_selector):
                         if tag: tag.decompose()


        if not main_content:
            logger.warning(f"Could not find suitable main content for: {url}")
            return title, "", []

        for tag_name in ["script", "style", "noscript"]:
             for tag in main_content.find_all(tag_name):
                if tag: tag.decompose()

        text = main_content.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        ignore_phrases = ["404 Not Found", "Page not found", "Coming Soon", "This site uses cookies", "Back to top", "Search \u00bb", "Skip to content"]
        for phrase in ignore_phrases:
            text = text.replace(phrase, "")
        text = text.strip()

        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            full_url = urljoin(url, href).split("#")[0].rstrip('/')
            if full_url and self._is_valid_url(full_url):
                links.append(full_url)

        return title, text, list(set(links)) # Unique links

    def crawl(self) -> List[Document]:
        """Executes the full website crawl."""
        logger.info(f"🚀 Starting powerful crawl from: {self.root_url}")
        documents: List[Document] = []
        page_count = 0
        skipped_count = 0

        while self.to_visit:
            current_url = self.to_visit.popleft()
            normalized_url = current_url.rstrip('/')

            if normalized_url in self.visited:
                continue

            self.visited.add(normalized_url)
            page_count += 1
            logger.info(f"🔗 Crawling #{page_count}: {current_url}")

            ext_type = self._get_extension_type(current_url)
            content = None
            metadata = {"source": current_url}
            new_links = [] # Initialize for all types

            try:
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
                        # Add placeholder content for images
                        content = f"[Image: {file_path.name} saved locally. Source: {current_url}]"
                        metadata["type"] = "image"
                        metadata["local_path"] = str(file_path)
                
                else: # Assume HTML
                    html = self._get_html(current_url)
                    if html:
                        soup = BeautifulSoup(html, "html.parser")
                        title, content, new_links = self._extract_content_and_links(soup, current_url)
                        metadata["title"] = title
                        metadata["type"] = "html"
                    else:
                        skipped_count += 1

                min_content_length = 20 if ext_type == "pdf" else 50 # Adjusted thresholds
                if content and len(content) >= min_content_length:
                    # Ensure metadata is serializable for FAISS later
                    serializable_metadata = {k: str(v) for k, v in metadata.items()}
                    documents.append(Document(page_content=content, metadata=serializable_metadata))
                elif content is not None:
                    logger.debug(f"Skipped {ext_type.upper()} (short content: {len(content)} chars): {current_url}")
                    skipped_count += 1
                elif ext_type == 'html': # Only count HTML skips if no content was found at all
                    skipped_count +=1

                for link in new_links:
                    normalized_link = link.rstrip('/')
                    # Add to queue only if not visited and not already in queue
                    if normalized_link not in self.visited and link not in self.to_visit:
                        self.to_visit.append(link)

            except Exception as e:
                 logger.error(f"Unexpected error processing {current_url}: {e}", exc_info=True)
                 skipped_count += 1

            time.sleep(self.delay) # Politeness delay

        duration = time.time() - self.start_time
        logger.info(f"✅ Crawl complete in {duration:.2f}s — Processed {len(self.visited)} URLs, created {len(documents)} documents, skipped {skipped_count}.")
        return documents


# --- Entry function for ingest.py ---
def scrape_all() -> List[Document]:
    """Initializes and runs the full website crawler."""
    crawler = SSUWebCrawler()
    return crawler.crawl()