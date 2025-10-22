import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# --- General Config ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Project Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
VECTOR_STORE_PATH = str(VECTOR_STORE_DIR / "faiss_index")

# --- Scraping Config (For Advanced Crawler) ---
BASE_URL = "https://srisriuniversity.edu.in/"

CRAWL_LIMIT = 150  # Maximum number of pages to scrape (matches scraper.py)
CRAWL_DELAY = 1.0  # Seconds to wait between requests
USER_AGENT = "VedikaSSUCrawler/2.0 (AI Chatbot for Sri Sri University)"

# --- RAG Config ---
EMBEDDING_MODEL_NAME = "models/embedding-001"
GENERATION_MODEL_NAME = "models/gemini-pro-latest"
