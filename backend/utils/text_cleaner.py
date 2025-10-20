import re
from typing import List
from langchain_core.documents import Document  # Import the Document object

def clean_text(doc: Document) -> str:
    """
    Extracts text from a Document object and performs basic cleaning:
    - Removes extra spaces, newlines, tabs
    """
    if not doc or not doc.page_content:
        return ""
    
    # --- THIS IS THE FIX ---
    # Extract the string content from the Document object
    text = doc.page_content
    # ---------------------
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 40) -> List[str]:
    """
    Splits a large text into smaller chunks based on word count.

    Args:
        text (str): The text to split.
        chunk_size (int): Max words per chunk.
        overlap (int): Words to overlap between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    if not text:
        return []

    words = text.split()
    chunks = []
    
    if not words:
        return []

    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        
        # Move the index forward
        step = chunk_size - overlap
        
        # Add a safeguard against infinite loops if overlap >= chunk_size
        if step <= 0:
            step = chunk_size 
            
        i += step

    return chunks