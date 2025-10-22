# ingest.py
import logging
from backend.scraper import scrape_all
from backend.vectorstore import VectorStore, Embedder
from backend.utils.text_cleaner import chunk_text, clean_text
from backend.utils.logger import get_logger

logger = get_logger("ingest")

def chunk_documents(documents, chunk_size=300, overlap=40):
    all_chunks = []
    for doc in documents:
        cleaned = clean_text(doc)
        chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
    return all_chunks

def create_and_save_store(chunks):
    if not chunks:
        logger.error("No chunks to create vector store.")
        return

    vector_store = VectorStore()
    embedder = Embedder()

    logger.info("Generating embeddings for chunks...")
    embeddings = embedder.embed_texts(chunks)
    metadata = [(f"chunk_{i}", chunks[i]) for i in range(len(chunks))]

    vector_store.add_vectors(embeddings, metadata)
    vector_store.save()

def main():
    logger.info("Starting ingestion pipeline...")

    # 1. Scrape all sites
    logger.info("Scraping documents from university websites...")
    documents = scrape_all()
    if not documents:
        logger.error("No documents scraped. Aborting.")
        return
    logger.info(f"Scraped {len(documents)} documents.")

    # 2. Chunk documents
    logger.info("Chunking documents...")
    chunks = chunk_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks.")

    # 3. Create and save vector store
    logger.info("Creating and saving vector store...")
    create_and_save_store(chunks)

    logger.info("Ingestion pipeline completed successfully.")

if __name__ == "__main__":
    main()