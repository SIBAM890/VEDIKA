import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from backend.config import (
    GOOGLE_API_KEY, 
    EMBEDDING_MODEL_NAME, 
    VECTOR_STORE_PATH
)
from backend.utils.logger import get_logger
from typing import List, Tuple, List

logger = get_logger("vectorstore")

# Configure the genai library
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not found in environment variables.")
genai.configure(api_key=GOOGLE_API_KEY)


class Embedder:
    """A wrapper for the Google Generative AI Embeddings model."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """Initializes the embedding model."""
        try:
            self.model = GoogleGenerativeAIEmbeddings(model=model_name)
            logger.info(f"Embedder initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Embedder: {e}")
            self.model = None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of text chunks.
        
        Args:
            texts (List[str]): The list of text chunks to embed.
            
        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        if not self.model:
            logger.error("Embedding model is not initialized.")
            return []
        
        logger.info(f"Embedding {len(texts)} text chunks...")
        try:
            embeddings = self.model.embed_documents(texts)
            logger.info("Embedding complete.")
            return embeddings
        except Exception as e:
            logger.error(f"Error during text embedding: {e}")
            return []


class VectorStore:
    """A wrapper for the FAISS Vector Store."""
    
    def __init__(self, path: str = VECTOR_STORE_PATH):
        """
        Initializes the VectorStore.
        
        Args:
            path (str): Path to save/load the FAISS index.
        """
        self.path = path
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        self.store = None
        logger.info(f"VectorStore initialized. Path: {self.path}")

    def add_vectors(self, embeddings: List[List[float]], metadata: List[Tuple[str, str]]):
        """
        Creates a FAISS store from pre-computed embeddings and metadata.
        
        Args:
            embeddings (List[List[float]]): The list of embedding vectors.
            metadata (List[Tuple[str, str]]): List of (id, text_chunk) tuples.
        """
        if not embeddings or not metadata:
            logger.error("No embeddings or metadata to add.")
            return
            
        try:
            # 1. Reformat data for FAISS.from_embeddings
            # It needs:
            # - text_embeddings: List[Tuple[str, List[float]]]
            # - metadatas_list: List[dict]
            
            text_embeddings = []
            metadatas_list = []
            
            for i in range(len(embeddings)):
                text_chunk = metadata[i][1]
                embedding = embeddings[i]
                chunk_id = metadata[i][0]
                
                text_embeddings.append((text_chunk, embedding))
                metadatas_list.append({"source_chunk_id": chunk_id})

            logger.info("Creating FAISS store from embeddings...")
            
            # 2. Create the FAISS store
            self.store = FAISS.from_embeddings(
                text_embeddings=text_embeddings,
                embedding=self.embeddings_model,
                metadatas=metadatas_list
            )
            logger.info("FAISS store created successfully.")
            
        except Exception as e:
            logger.error(f"Failed to create FAISS store from embeddings: {e}")

    def save(self):
        """Saves the FAISS store to the local disk."""
        if not self.store:
            logger.error("No vector store to save.")
            return
            
        try:
            self.store.save_local(self.path)
            logger.info(f"Vector store saved successfully to: {self.path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")

    def load(self):
        """
Other AI
Loads the FAISS store from disk and returns it as a retriever.
        
        Returns:
            A FAISS retriever object, or None if loading fails.
        """
        try:
            if not os.path.exists(self.path):
                logger.error(f"Vector store not found at {self.path}. Run ingest.py.")
                raise FileNotFoundError(f"Vector store not found at {self.path}")
                
            logger.info(f"Loading vector store from: {self.path}")
            
            self.store = FAISS.load_local(
                folder_path=self.path, 
                embeddings=self.embeddings_model,
                allow_dangerous_deserialization=True
            )
            
            logger.info("Vector store loaded successfully.")
            
            # Return the store as a retriever for the RAG engine
            return self.store.as_retriever(search_kwargs={"k": 3})

        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return None