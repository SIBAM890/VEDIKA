"""
RAG Engine for Sri Sri University Chatbot
Handles intelligent routing between university-specific queries and casual chat.
"""

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser

from backend.vectorstore import VectorStore
from backend.config import GOOGLE_API_KEY, GENERATION_MODEL_NAME
from backend.utils.logger import get_logger

logger = get_logger("rag_engine")


# =============================================================================
# CONFIGURATION
# =============================================================================

def configure_gemini() -> None:
    """Configure Google Generative AI with API key."""
    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "❌ GOOGLE_API_KEY not found in environment variables."
        )
    
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("✅ Gemini API configured successfully.")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to configure Google Generative AI: {e}")


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

ROUTER_PROMPT = PromptTemplate.from_template("""
You are an expert intent classifier.
Classify the following user query into one of these categories:
1. university_related
2. casual_chat
3. greeting
4. thanks

Respond with only the category name.

User Query:
{input}
""")

RAG_PROMPT = PromptTemplate.from_template("""
You are Vedika — a helpful AI assistant for Sri Sri University.
You are currently in a conversation. Be direct and natural.
**Do not re-introduce yourself.**
Answer the user's question using ONLY the information provided in the context below.
If the context does not contain the answer, reply:
"I'm sorry, I don't have that information. Please contact the university for details."
When listing courses or programs, format them in **clear, bullet-pointed sections** with headings for undergraduate, postgraduate, and other programs, so it is easy to read.
Context:
{context}

Question:
{question}

Answer:
""")

CASUAL_PROMPT = PromptTemplate.from_template("""
You are Vedika — a friendly conversational assistant.
You are currently in a conversation.
**Do not re-introduce yourself.** Respond in a warm, natural, and helpful tone.

User Message:
{input}

Response:
""")


# =============================================================================
# COMPONENT INITIALIZATION
# =============================================================================

class RAGComponents:
    """Container for RAG system components."""
    
    def __init__(self):
        self.llm = None
        self.retriever = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize LLM and retriever components."""
        try:
            self._init_llm()
            self._init_retriever()
            logger.info("✅ LLM and retriever successfully initialized.")
        except Exception as e:
            logger.error(
                f"❌ Failed to initialize RAG components: {e}",
                exc_info=True
            )
    
    def _init_llm(self) -> None:
        """Initialize the language model."""
        self.llm = ChatGoogleGenerativeAI(
            model=GENERATION_MODEL_NAME,
            temperature=0.7,
            convert_system_message_to_human=True,
        )
    
    def _init_retriever(self) -> None:
        """Initialize the vector store retriever."""
        vector_store_manager = VectorStore()
        self.retriever = vector_store_manager.load()
        
        if self.retriever is None:
            raise ValueError(
                "Retriever initialization failed (Vector store not loaded)."
            )
    
    def is_ready(self) -> bool:
        """Check if all components are initialized."""
        return self.llm is not None and self.retriever is not None


# =============================================================================
# CHAIN CONSTRUCTION
# =============================================================================

def build_chat_chain(components: RAGComponents):
    """
    Build the main chat chain with routing logic.
    
    Args:
        components: Initialized RAG components
        
    Returns:
        The complete chat chain with routing
    """
    # Individual chains
    router_chain = ROUTER_PROMPT | components.llm | StrOutputParser()
    
    rag_chain = (
        {
            "context": components.retriever,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | components.llm
        | StrOutputParser()
    )
    
    casual_chain = CASUAL_PROMPT | components.llm | StrOutputParser()
    
    # Simple response chains
    greeting_chain = RunnableLambda(lambda x: "JayGurudev!")
    thanks_chain = RunnableLambda(lambda x: "JayGurudev!")
    
    # Routing logic
    chain_with_routing = {
        "topic": router_chain,
        "input": lambda x: x["input"],
    }
    
    # Branch based on classification
    main_chain = chain_with_routing | RunnableBranch(
        (
            lambda x: "greeting" in x.get("topic", "").lower(),
            greeting_chain,
        ),
        (
            lambda x: "thanks" in x.get("topic", "").lower(),
            thanks_chain,
        ),
        (
            lambda x: "university_related" in x.get("topic", "").lower(),
            RunnableLambda(lambda x: x["input"]) | rag_chain,
        ),
        RunnableLambda(lambda x: x["input"]) | casual_chain,
    )
    
    return main_chain


# =============================================================================
# INITIALIZATION & API
# =============================================================================

# Configure and initialize
configure_gemini()
rag_components = RAGComponents()
main_chat_chain = build_chat_chain(rag_components) if rag_components.is_ready() else None


def get_chat_chain():
    """
    Returns the main RAG chain for chat API.
    
    Returns:
        The initialized chat chain or None if initialization failed
    """
    if not rag_components.is_ready():
        logger.error(
            "❌ RAG engine not fully initialized — cannot serve chat requests."
        )
        return None
    
    logger.info("✅ Chat chain ready for requests.")
    return main_chat_chain