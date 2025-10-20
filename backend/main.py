from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import chat
from backend.utils.logger import get_logger
import uvicorn

logger = get_logger("main_server")

# Initialize the FastAPI app
app = FastAPI(
    title="Vedika Chatbot API",
    description="API for the Sri Sri University RAG chatbot.",
    version="1.0.0"
)

# --- CORS Middleware ---
# This allows your frontend (running on a different port)
# to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["GET", "POST"], # Allows GET and POST requests
    allow_headers=["*"], # Allows all headers
)

# --- Include Routers ---
app.include_router(chat.router, prefix="/api")

# --- Root Endpoint ---
@app.get("/", tags=["Health Check"])
def read_root():
    """
    Root endpoint for health check.
    """
    return {"status": "ok", "message": "Welcome to the Vedika Chatbot API!"}

# --- Main entry point for running the server ---
if __name__ == "__main__":
    logger.info("Starting Vedika Chatbot server...")
    # Run the server with Uvicorn
    # Host="127.0.0.1" is standard for local development
    uvicorn.run(app, host="127.0.0.1", port=8000)