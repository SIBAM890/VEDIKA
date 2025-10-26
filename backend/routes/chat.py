# backend/routes/chat.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.rag_engine import get_chat_chain
from backend.utils.logger import get_logger
import traceback

router = APIRouter()
logger = get_logger("chat_api")

# ✅ Initialize the RAG chat chain (OpenAI backend)
try:
    chain = get_chat_chain()
    if chain is None:
        raise ImportError("Failed to load the RAG chain.")
    logger.info("✅ Chat chain initialized successfully with OpenAI backend.")
except Exception as e:
    logger.error(f"❌ Fatal error initializing chat chain: {e}")
    chain = None

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

# --- Chat Endpoint ---
@router.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    if chain is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized.")

    user_message = request.message
    logger.info(f"Received message: {user_message}")

    try:
        # 🔁 Invoke the RAG chain asynchronously
        response = await chain.ainvoke({"input": user_message})

        # Normalize the response format
        if isinstance(response, dict):
            # Try common keys where the main response might be stored
            response_text = (
                response.get("output")
                or response.get("answer")
                or response.get("result")
                or str(response) # Fallback to string representation of the dict
            )
        else:
            # If it's not a dict, assume it's the string response
            response_text = str(response)

        # Handle cases where the chain might return an empty or whitespace-only string
        if not response_text.strip():
            logger.warning("Empty response received from the RAG chain.")
            response_text = "I'm sorry, I couldn't retrieve relevant information at the moment. Please try rephrasing your question."

        logger.info(f"Sending reply: {response_text[:100]}...") # Log truncated reply
        return ChatResponse(reply=response_text)

    except Exception as e:
        # Log the full error traceback for easier debugging
        error_trace = traceback.format_exc()
        logger.error(f"Error during chain invocation: {e}\n{error_trace}")
        # Return a generic error message to the user
        raise HTTPException(status_code=500, detail="Internal processing error. Please try again later.")