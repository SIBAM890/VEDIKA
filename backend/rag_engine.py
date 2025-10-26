import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from backend.vectorstore import VectorStore
from backend.config import OPENAI_API_KEY, GENERATION_MODEL_NAME
from backend.utils.logger import get_logger

logger = get_logger("rag_engine")

# -------------------------------------------------------------------------
# 1. OpenAI Configuration
# -------------------------------------------------------------------------
if not OPENAI_API_KEY:
    raise EnvironmentError("❌ OPENAI_API_KEY not found in environment variables.")

# -------------------------------------------------------------------------
# 2. Initialize LLM and Retriever
# -------------------------------------------------------------------------
try:
    llm = ChatOpenAI(
        model=GENERATION_MODEL_NAME,
        temperature=0.7,
        api_key=OPENAI_API_KEY
    )
    logger.info(f"✅ LLM initialized with OpenAI model: {GENERATION_MODEL_NAME}")

    vector_store_manager = VectorStore()
    retriever = vector_store_manager.load()

    if retriever is None:
        raise ValueError("Retriever initialization failed (Vector store not loaded).")
    logger.info("✅ Retriever successfully initialized (using OpenAI embeddings).")

except Exception as e:
    logger.error(f"❌ Failed to initialize LLM or retriever: {e}", exc_info=True)
    llm = None
    retriever = None

# --- Hardcoded Course List (Keep as is) ---
HARDCODED_COURSE_LIST = """
Okay, here is a comprehensive list of the programs offered at Sri Sri University:

**Undergraduate Programs:**
* Bachelor of Architecture
* Bachelor of Business Administration (BBA)
* BBA Logistics Management
* Bachelor of Ayurveda, Medicine and Surgery (BAMS)
* Bachelor in Computer Application (BCA)
* BCA in Data Science and Analytics
* BCA Cyber Security
* B.Com (Hons.)
* B.Sc. Osteopathy
* B.Sc. Yoga
* B.Sc. (Hons.) in Agriculture
* B.Sc. (Hons.) Horticulture
* B.Sc. (Hons.) Agribusiness
* B.Sc. (Hons.) Food Nutrition and Dietetics
* B.Sc. (Hons.) Physics
* B.Sc. (Hons.) Computer Science
* B.Sc. (Hons.) Environmental Science
* B.Sc. (Hons.) in Psychology and Contemplative Studies
* B.Sc. Forensic Science
* B.Sc. in Forensic Science (Specialization in Digital Forensics & Cyber Forensics)
* B.Sc. Nursing
* B.Sc. in Optometry
* Bachelor of Fine Arts
* Bachelor in Interior Design
* Bachelor of Performing Arts (Odissi Dance)
* Bachelor of Performing Arts (Hindustani Vocal)
* B.A. (Hons.) English
* B.Tech. Computer Science Engineering
* B.Tech CSE (AI and Machine Learning)
* B.Tech CSE (Data Science)
* B.Tech CSE (Cyber Security and Cyber Defence)
* B.Tech. in CSE – Cyber Security & Cyber Defense (Apprentices Embedded)
* B.Tech. Electrical Engineering (EV, AI and Renewable Energy)
* BTech CSE in Animation and VFX
* B.Tech. Electronics & Communication Engineering (VLSI Chip Design, EV and AI)
* B.Tech. in Mechanical Engg. (Robotics & Automation, EV and AI)
* Bachelor in Physical Education & Sports (BPES)
* General Nursing and Midwifery
* Bachelor of Naturopathy and Yogic Science (BNYS)
* Bachelor of Physiotherapy (BPT)

**Postgraduate Programs:**
* Master of Business Administration (MBA)
* MBA Agribusiness
* MBA Fintech
* Global MBA
* M.Sc. Osteopathy
* M.Sc. Yoga
* MA Yoga
* M.A. (Sanskrit)
* MPA (Odissi Dance)
* M.A. (English)
* M.A. in Hindu Studies
* M.Sc in Psychology and Contemplative Studies
* Master of Computer Application (MCA)
* MCA – Cyber Security (Apprentices Embedded)
* MCA with specialization in Cyber Security & Incident Management
* MCA in Data Science and Analytics
* M.Sc. Agronomy
* M.Sc. Soil Science and Agricultural Chemistry
* M.Sc. Genetics and Plant Breeding
* M.Sc. Entomology
* M.Sc. Horticulture
* M.Sc. Agricultural Statistics
* M.Sc. Agricultural Economics
* M.Sc. Agricultural Extension
* M.Sc. Plant Pathology
* MSc in Exercise & Sports Physiology
* M.Sc. in Cyber Security & Digital Forensic
* M.Sc. in Artificial Intelligence & Machine Learning
* M.Tech. – Cyber Security (Apprentices Embedded)
* M.Tech. in Cyber Security & Digital Forensic
* M.Tech. in Artificial Intelligence & Machine Learning
* MTech in Cybersecurity & Incident Management

**Other Programs:**
* Post Graduate Programme in Startup Development
* Post Graduate Diploma in Disaster Management
* PG Diploma in AI Application for Disaster Risk Reduction
* Integrated M.Sc. in Osteopathy
* PG Diploma in Geospatial Technologies
* Advanced Certification in Geospatial Technologies
* Ph.D. (Research & Development Cell) programs are also available.
* Open and Distance Learning (ODL) options exist.

For the most current details, admission criteria, and fee structures, please refer to the official Sri Sri University website or contact their admissions office.
"""
# --- END: Hardcoded Course List ---


# -------------------------------------------------------------------------
# 3. Prompt Templates
# -------------------------------------------------------------------------

# --- **THIS IS THE FIX** ---
# Router prompt: Made 'list_courses' more specific and 'university_related' broader.
router_prompt = PromptTemplate.from_template("""
You are an expert intent classifier.
Classify the following user query into one of these categories:

1.  **greeting**: Simple hellos, hi, good morning, etc.
2.  **thanks**: Simple thank you, thanks, thnx, etc.
3.  **list_courses**: Questions asking *only* for a list of *all* programs, courses, or degrees offered (e.g., "What courses do you offer?", "List all programs").
4.  **university_related**: All other specific questions about Sri Sri University, including details about a *specific course* (like "Tell me about BPT", "B.Tech CSE details"), fees, admissions, faculty, facilities, contact info, etc.
5.  **casual_chat**: General conversation, jokes, non-university topics.

Respond with only the category name (e.g., greeting, thanks, list_courses, university_related, casual_chat).

User Query:
{input}
""")
# --- **END OF FIX** ---

# RAG prompt (No changes needed)
rag_prompt = PromptTemplate.from_template("""
You are Vedika — a helpful AI assistant for Sri Sri University.
You are currently in a conversation. Be direct and natural.
**Do not re-introduce yourself.**

Answer the user's question using ONLY the information provided in the context below.
* If the user asks for a list or count of items (like courses, departments, etc.), list the specific items found in the context first. Then, *if possible*, provide a count based *only* on the listed items.
* If the context does not contain the answer or enough information to count, reply:
    "I found information about [mention topic, e.g., 'courses'], but I cannot provide an exact count based on the available details. Please check the university website or contact them directly for a complete number."

Context:
{context}

Question:
{question}

Answer:
""")

# Casual prompt (No changes needed)
casual_prompt = PromptTemplate.from_template("""
You are Vedika — a friendly conversational assistant.
You are currently in a conversation.
**Do not re-introduce yourself.** Respond in a warm, natural, and helpful tone.

User Message:
{input}

Response:
""")

# -------------------------------------------------------------------------
# 4. Define Processing Chains
# -------------------------------------------------------------------------
# (No changes needed here)
router_chain = router_prompt | llm | StrOutputParser()
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)
casual_chain = casual_prompt | llm | StrOutputParser()

# -------------------------------------------------------------------------
# 5. Routing Logic (Router → Conditional Chain Execution)
# -------------------------------------------------------------------------
# (No changes needed here - the logic relies on the router's output)
chain_with_routing = {
    "topic": router_chain,
    "input": lambda x: x["input"],
}

main_chat_chain = chain_with_routing | RunnableBranch(
    # Branch 1: Handle greetings
    (
        lambda x: "greeting" in x.get("topic", "").lower(),
        RunnableLambda(lambda x: "Jay Gurudev!")
    ),
    # Branch 2: Handle thanks
    (
        lambda x: "thanks" in x.get("topic", "").lower(),
        RunnableLambda(lambda x: "Jay Gurudev!")
    ),
    # Branch 3: Handle list_courses intent
    (
        lambda x: "list_courses" in x.get("topic", "").lower(),
        RunnableLambda(lambda x: HARDCODED_COURSE_LIST)
    ),
    # Branch 4: Handle other university questions via RAG
    (
        lambda x: "university_related" in x.get("topic", "").lower(),
        RunnableLambda(lambda x: x["input"]) | rag_chain,
    ),
    # Branch 5: Default fallback to casual chat
    RunnableLambda(lambda x: x["input"]) | casual_chain
)

# -------------------------------------------------------------------------
# 6. API Entry Point
# -------------------------------------------------------------------------
# (No changes needed here)
def get_chat_chain():
    """
    Returns the main runnable chain for the chat API endpoint.
    """
    if not llm or not retriever:
        logger.error("❌ RAG engine not fully initialized — cannot serve chat requests.")
        return None

    logger.info("✅ Chat chain initialized and ready for requests.")
    return main_chat_chain

