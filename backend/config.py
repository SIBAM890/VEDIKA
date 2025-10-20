import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Go up one level (from backend/) to the root to find .env
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# --- General Config ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Project Paths ---
# Base directory of the project (vedika-chatbot/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Vector store path
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
VECTOR_STORE_PATH = str(VECTOR_STORE_DIR / "faiss_index") # FAISS needs a string path

# --- Scraping Config ---
UNIVERSITY_URLS = [
    "https://srisriuniversity.edu.in/admission-process/",
    "https://srisriuniversity.edu.in/explore-programs/",
    "https://srisriuniversity.edu.in/fee-structure-2025-26/",
    "https://srisriuniversity.edu.in/hostel-transport-facility/",
    "https://srisriuniversity.edu.in/contact-us/",
    "https://srisriuniversity.edu.in/important-links/",
]

# --- RAG Config ---
EMBEDDING_MODEL_NAME = "models/embedding-001"
GENERATION_MODEL_NAME = "models/gemini-pro-latest"
