from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.rag_engine import get_chat_chain
from backend.utils.logger import get_logger

# Define the API router
router = APIRouter()
logger = get_logger("chat_api")

# Load the main chat chain
try:
    chain = get_chat_chain()
    if chain is None:
        raise ImportError("Failed to load the RAG chain.")
except Exception as e:
    logger.error(f"Fatal error initializing chat chain: {e}")
    chain = None

# --- Pydantic Models for Request/Response ---

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

# --- API Endpoint ---

@router.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    Handles a user chat message.
    It invokes the main RAG chain asynchronously.
    """
    if chain is None:
        logger.error("Chat endpoint called, but chain is not available.")
        raise HTTPException(
            status_code=500, 
            detail="Chatbot is not initialized. Please check server logs."
        )
        
    logger.info(f"Received message: {request.message}")
    
    try:
        # Asynchronously invoke the chain
        response = await chain.ainvoke({"input": request.message})
        
        logger.info(f"Sending reply: {response}")
        return ChatResponse(reply=response)
        
    except Exception as e:
        logger.error(f"Error during chain invocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))