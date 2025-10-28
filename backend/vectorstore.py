import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from backend.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_PATH,
)
from backend.utils.logger import get_logger
from typing import List, Tuple

logger = get_logger("vectorstore")


if not OPENAI_API_KEY:
    raise EnvironmentError("❌ OPENAI_API_KEY not found in environment variables.")


class Embedder:
    """Handles text embedding with automatic batching to avoid token limit issues."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        try:
            self.model = OpenAIEmbeddings(model=model_name, api_key=OPENAI_API_KEY)
            logger.info(f"✅ Embedder initialized with OpenAI model: {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI Embedder: {e}")
            self.model = None

    def embed_texts(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """Embeds text in small batches to avoid token overflow errors."""
        if not self.model:
            logger.error("❌ Embedding model is not initialized.")
            return []

        embeddings = []
        total = len(texts)
        logger.info(f"Embedding {total} text chunks using OpenAI (batch size = {batch_size})...")

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = self.model.embed_documents(batch)
                embeddings.extend(batch_embeddings)
                logger.info(f"✅ Embedded batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")
            except Exception as e:
                logger.error(f"❌ Error embedding batch {i}: {e}")
        return embeddings


class VectorStore:
    """Manages FAISS-based vector storage for OpenAI embeddings."""

    def __init__(self, path: str = VECTOR_STORE_PATH):
        self.path = path
        self.embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=OPENAI_API_KEY)
        self.store = None
        logger.info(f"VectorStore initialized for OpenAI. Path: {self.path}")

    def add_vectors(self, embeddings: List[List[float]], metadata: List[Tuple[str, str]]):
        """Creates FAISS store from precomputed embeddings."""
        if not embeddings or not metadata:
            logger.error("No embeddings or metadata to add.")
            return

        try:
            texts = [m[1] for m in metadata]
            metadatas = [{"id": m[0]} for m in metadata]

            logger.info("Creating FAISS store from OpenAI embeddings...")
            self.store = FAISS.from_embeddings(
                text_embeddings=list(zip(texts, embeddings)),
                embedding=self.embeddings_model,
                metadatas=metadatas,
            )
            logger.info("✅ FAISS store created successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to create FAISS store: {e}")

    def save(self):
        """Saves FAISS store locally."""
        if not self.store:
            logger.error("No vector store instance to save.")
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.store.save_local(self.path)
            logger.info(f"✅ Vector store saved successfully to: {self.path}")
        except Exception as e:
            logger.error(f"❌ Failed to save vector store: {e}")

    def load_store(self):
        """
        Loads and returns the raw FAISS vector store instance.
        Use this for dynamic k retrieval in queries.
        
        Returns:
            FAISS vectorstore instance or None if loading fails
        """
        try:
            if not os.path.exists(self.path):
                logger.error(f"Vector store not found at {self.path}. Please run ingest.py first.")
                return None

            logger.info(f"Loading vector store from: {self.path}")
            self.store = FAISS.load_local(
                folder_path=self.path,
                embeddings=self.embeddings_model,
                allow_dangerous_deserialization=True,
            )
            logger.info("✅ Vector store loaded successfully (raw instance for dynamic retrieval).")
            return self.store

        except Exception as e:
            logger.error(f"❌ Error loading vector store: {e}")
            return None

    def load(self, k: int = 5):
        """
        Loads FAISS store and returns as retriever with fixed k.
        
        Args:
            k: Number of documents to retrieve (default: 5)
            
        Returns:
            Retriever instance or None if loading fails
            
        Note: For dynamic k based on query type, use load_store() instead
        """
        try:
            if not os.path.exists(self.path):
                logger.error(f"Vector store not found at {self.path}. Please run ingest.py first.")
                return None

            logger.info(f"Loading vector store from: {self.path}")
            self.store = FAISS.load_local(
                folder_path=self.path,
                embeddings=self.embeddings_model,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"✅ Vector store loaded successfully. Retriever configured with k={k}")
            return self.store.as_retriever(search_kwargs={"k": k})

        except Exception as e:
            logger.error(f"❌ Error loading vector store: {e}")
            return None
    
    def similarity_search(self, query: str, k: int = 5):
        """
        Convenience method for direct similarity search.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        if not self.store:
            logger.warning("Vector store not loaded. Loading now...")
            self.store = self.load_store()
            
        if not self.store:
            logger.error("Failed to load vector store for search.")
            return []
            
        try:
            results = self.store.similarity_search(query, k=k)
            logger.info(f"✅ Retrieved {len(results)} documents for query (k={k})")
            return results
        except Exception as e:
            logger.error(f"❌ Similarity search failed: {e}")
            return []
    
    def mmr_search(self, query: str, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.5):
        """
        Maximum Marginal Relevance search for diverse results.
        Useful for getting varied perspectives on a topic.
        
        Args:
            query: Search query string
            k: Number of results to return
            fetch_k: Number of documents to fetch before MMR reranking
            lambda_mult: Diversity parameter (0=max diversity, 1=max relevance)
            
        Returns:
            List of Document objects
        """
        if not self.store:
            logger.warning("Vector store not loaded. Loading now...")
            self.store = self.load_store()
            
        if not self.store:
            logger.error("Failed to load vector store for MMR search.")
            return []
            
        try:
            results = self.store.max_marginal_relevance_search(
                query, 
                k=k, 
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )
            logger.info(f"✅ MMR search retrieved {len(results)} diverse documents (k={k}, fetch_k={fetch_k})")
            return results
        except Exception as e:
            logger.error(f"❌ MMR search failed: {e}")
            return []