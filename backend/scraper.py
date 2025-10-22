import re
import time
import requests
import logging  # <-- FIX 1: Added missing import
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from collections import deque
from typing import Set, List, Deque
from langchain_core.documents import Document

# Import config and logger from our project structure
from backend.config import BASE_URL, CRAWL_LIMIT, CRAWL_DELAY, USER_AGENT
from backend.utils.logger import get_logger

logger = get_logger("scraper")

class SSUWebCrawler:
    """
    Advanced recursive scraper for the Sri Sri University official website,
    designed to integrate with a LangChain pipeline.
    """

    def __init__(self):
        self.root_url: str = BASE_URL.rstrip('/')
        self.base_domain: str = urlparse(self.root_url).netloc
        self.max_pages: int = CRAWL_LIMIT
        self.delay: float = CRAWL_DELAY
        
        self.visited_urls: Set[str] = set()
        self.to_visit: Deque[str] = deque([self.root_url])
        
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.start_time = time.time()  # <-- FIX 2: Added missing attribute

    def _is_valid_url(self, url: str) -> bool:
        """Checks if a URL should be crawled."""
        parsed = urlparse(url)
        if parsed.netloc != self.base_domain:
            return False
        if re.search(r'\.(pdf|docx?|xlsx?|pptx?|jpg|jpeg|png|gif|zip|rar|mp3|mp4|xml|svg)$', parsed.path, re.IGNORECASE):
            return False
        if url.startswith(('mailto:', 'tel:', 'javascript:')):
            return False
        return True

    def _get_html(self, url: str) -> str | None:
        """Safely fetches HTML content from a URL."""
        try:
            response = self.session.get(url, timeout=12)
            if response.status_code != 200:
                logger.warning(f"Skipping {url} — Status {response.status_code}")
                return None
            if "text/html" not in response.headers.get("Content-Type", ""):
                logger.debug(f"Skipping {url} — Non-HTML content.")
                return None
            return response.text
        except requests.RequestException as e:
            logger.error(f"⚠️ Failed to fetch {url}: {e}")
            return None

    def _extract_content_and_links(self, soup: BeautifulSoup, url: str) -> tuple[str, str, List[str]]:
        """Extracts title, clean text, and new internal links from a BeautifulSoup object."""
        
        title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled Page"
        
        main_content = None
        selectors = ['main', 'article', 'div.entry-content', 'div.main-content', 'div#content']
        for selector in selectors:
            content_area = soup.select_one(selector)
            if content_area:
                main_content = content_area
                break
        
        if not main_content:
            main_content = soup.body
        
        if not main_content:
            return title, "", []

        for tag in main_content(["header", "footer", "nav", "aside", "form", "script", "style"]):
            tag.decompose()
            
        text = main_content.get_text(" ", strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        
        links = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            full_url = urljoin(url, href).split("#")[0].rstrip('/')
            
            if self._is_valid_url(full_url) and full_url not in self.visited_urls:
                links.append(full_url)
                
        return title, text, list(set(links))

    def crawl(self) -> List[Document]:
        """
        Executes the crawl and returns a list of LangChain Documents.
        """
        logger.info(f"🚀 Starting crawl from: {self.root_url} (Limit: {self.max_pages} pages)")
        scraped_docs: List[Document] = []

        while self.to_visit and len(self.visited_urls) < self.max_pages:
            current_url = self.to_visit.popleft()
            
            if current_url in self.visited_urls:
                continue

            self.visited_urls.add(current_url)
            # <-- FIX 3: Standardized logger -->
            logger.info(f"🔗 Crawling ({len(self.visited_urls)}/{self.max_pages}): {current_url}")

            html = self._get_html(current_url)
            if not html:
                time.sleep(self.delay)
                continue
            
            soup = BeautifulSoup(html, 'html.parser')
            title, content, new_links = self._extract_content_and_links(soup, current_url)
            
            if len(content) > 150:
                doc = Document(
                    page_content=content,
                    metadata={"source": current_url, "title": title}
                )
                scraped_docs.append(doc)
            else:
                logger.debug(f"Skipping {current_url} (content too short).")

            for link in new_links:
                if link not in self.to_visit and link not in self.visited_urls:
                    self.to_visit.append(link)

            time.sleep(self.delay)

        duration = time.time() - self.start_time
        logger.info(f"✅ Crawl complete in {duration:.2f} seconds. Scraped {len(scraped_docs)} valuable pages.")
        return scraped_docs

def scrape_all() -> List[Document]:
    """
    The main function called by the ingestion pipeline.
    Initializes and runs the web crawler.
    """
    crawler = SSUWebCrawler()
    documents = crawler.crawl()
    return documents

    

