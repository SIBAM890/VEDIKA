from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from backend.config import UNIVERSITY_URLS
from backend.utils.logger import get_logger
from typing import List

logger = get_logger("scraper")

def scrape_all() -> List[Document]:
    """
    Scrapes all URLs defined in the config file using WebBaseLoader.
    
    Returns:
        A list of LangChain Document objects.
    """
    try:
        logger.info(f"Loading {len(UNIVERSITY_URLS)} URLs...")
        
        loader = WebBaseLoader(UNIVERSITY_URLS)
        loader.requests_per_second = 1 # Be polite to the server
        
        documents = loader.load()
        
        logger.info(f"Successfully loaded {len(documents)} documents.")
        return documents
        
    except Exception as e:
        logger.error(f"Error during web scraping: {e}")
        return []