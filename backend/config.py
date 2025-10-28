import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Assumes .env is in the project root, one level up from backend/
env_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(dotenv_path=env_path)

# --- General Config ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Keep just in case
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Essential for current setup

# --- Project Paths ---
BASE_DIR = Path(__file__).resolve().parents[1]
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
VECTOR_STORE_PATH = str(VECTOR_STORE_DIR / "faiss_index")
# Path for scraper to save raw assets (PDFs, images, etc.)
DATA_DIR = BASE_DIR / "data" / "raw"

# --- Scraping Config (For Advanced Crawler) ---
# Starting point for the crawl
BASE_URL = os.getenv("SSU_ROOT_URL", "https://srisriuniversity.edu.in/")
# Safety limit (can be adjusted or removed if you truly want *everything*)
# CRAWL_LIMIT = int(os.getenv("MAX_PAGES_TO_SCRAPE", 300)) # Commented out to allow full crawl
# Polite delay between requests (seconds)
CRAWL_DELAY = float(os.getenv("SCRAPE_DELAY", 1.0))
# User agent to identify the crawler
USER_AGENT = os.getenv("SCRAPER_USER_AGENT", "VedikaSSUCrawler/3.0 (Educational AI Assistant for Sri Sri University)")

# --- RAG Config ---
EMBEDDING_MODEL_NAME = "text-embedding-3-small" # OpenAI
GENERATION_MODEL_NAME = "gpt-4o-mini"          # OpenAI

