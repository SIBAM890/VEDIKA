import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).resolve().parents[1] / '.env' # Adjusted path finding
load_dotenv(dotenv_path=env_path)

# --- General Config ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Project Paths ---
BASE_DIR = Path(__file__).resolve().parents[1] # Adjusted path finding
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
VECTOR_STORE_PATH = str(VECTOR_STORE_DIR / "faiss_index")
# --- ADDED: Data directory for scraper assets ---
DATA_DIR = BASE_DIR / "data" / "raw" # Base directory for raw scraped data

# --- Scraping Config (Using values from your advanced scraper) ---
BASE_URL = os.getenv("SSU_ROOT_URL", "https://srisriuniversity.edu.in")
# MAX_PAGES_TO_SCRAPE = int(os.getenv("MAX_PAGES_TO_SCRAPE", 300)) # No limit needed anymore for full crawl
CRAWL_DELAY = float(os.getenv("SCRAPE_DELAY", 1.0)) # Adjusted default delay slightly
USER_AGENT = os.getenv("SCRAPER_USER_AGENT", "VedikaSSUCrawler/3.0 (Educational AI Assistant for Sri Sri University)")


# --- RAG Config ---
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATION_MODEL_NAME = "gpt-4o-mini"