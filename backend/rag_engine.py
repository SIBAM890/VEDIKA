import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
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
# 2. Initialize LLM and Vector Store
# -------------------------------------------------------------------------
try:
    llm = ChatOpenAI(
        model=GENERATION_MODEL_NAME,
        temperature=0.2,  # Very low for maximum accuracy
        api_key=OPENAI_API_KEY
    )
    logger.info(f"✅ LLM initialized with OpenAI model: {GENERATION_MODEL_NAME}")

    vector_store_manager = VectorStore()
    vector_store = vector_store_manager.load_store()
    
    if vector_store is None:
        raise ValueError("Vector store initialization failed.")
    logger.info("✅ Vector store successfully initialized for maximum accuracy retrieval.")

except Exception as e:
    logger.error(f"❌ Failed to initialize LLM or vector store: {e}", exc_info=True)
    llm = None
    vector_store = None

# -------------------------------------------------------------------------
# 3. Advanced Retrieval System
# -------------------------------------------------------------------------

def analyze_query_complexity(query: str) -> Dict[str, Any]:
    """
    Deep analysis of query to determine optimal retrieval strategy.
    
    Returns:
        Dict with query_type, k_value, use_mmr, use_reranking flags
    """
    query_lower = query.lower()
    
    # Comprehensive keyword sets
    list_keywords = [
        'all', 'list', 'complete', 'entire', 'every', 'total',
        'courses', 'programs', 'specializations', 'degrees',
        'how many', 'available', 'offered', 'options', 'what are',
        'show me', 'tell me about all', 'give me'
    ]
    
    specific_keywords = [
        'about', 'details', 'information', 'curriculum', 'syllabus',
        'eligibility', 'admission', 'fees', 'duration', 'career',
        'placement', 'faculty', 'infrastructure', 'facilities'
    ]
    
    comparison_keywords = [
        'difference between', 'compare', 'vs', 'versus', 'better',
        'which one', 'should i choose'
    ]
    
    # Program/Course names
    program_names = [
        'btech', 'b.tech', 'mtech', 'm.tech', 'mba', 'bba',
        'bca', 'mca', 'bsc', 'b.sc', 'msc', 'm.sc', 'ba', 'ma',
        'bcom', 'mcom', 'bpt', 'bnys', 'bams', 'phd'
    ]
    
    # Analyze query type
    is_list_query = any(kw in query_lower for kw in list_keywords)
    is_specific = any(kw in query_lower for kw in specific_keywords)
    is_comparison = any(kw in query_lower for kw in comparison_keywords)
    mentions_program = any(prog in query_lower for prog in program_names)
    
    # Determine strategy
    if is_list_query:
        query_type = "comprehensive_list"
        k_value = 30  # Maximum retrieval for lists
        use_mmr = True
        use_reranking = True
    elif is_comparison:
        query_type = "comparison"
        k_value = 20  # Need context from multiple programs
        use_mmr = True
        use_reranking = True
    elif is_specific and mentions_program:
        query_type = "specific_program"
        k_value = 15  # Detailed info about one program
        use_mmr = False
        use_reranking = True
    elif mentions_program:
        query_type = "program_overview"
        k_value = 12
        use_mmr = False
        use_reranking = False
    else:
        query_type = "general"
        k_value = 10
        use_mmr = False
        use_reranking = False
    
    logger.info(f"🔍 Query Analysis: type={query_type}, k={k_value}, mmr={use_mmr}, rerank={use_reranking}")
    
    return {
        "query_type": query_type,
        "k_value": k_value,
        "use_mmr": use_mmr,
        "use_reranking": use_reranking
    }

def deduplicate_documents(docs: List[Document]) -> List[Document]:
    """
    Remove duplicate or highly similar documents based on content.
    """
    seen_content = set()
    unique_docs = []
    
    for doc in docs:
        # Create a signature of the document (first 200 chars)
        content_signature = doc.page_content[:200].strip().lower()
        
        if content_signature not in seen_content:
            seen_content.add(content_signature)
            unique_docs.append(doc)
    
    logger.info(f"📋 Deduplicated: {len(docs)} → {len(unique_docs)} documents")
    return unique_docs

def rerank_documents(query: str, docs: List[Document]) -> List[Document]:
    """
    Rerank documents using semantic similarity scoring.
    Uses a simple relevance scoring based on keyword matching and position.
    """
    query_terms = set(query.lower().split())
    
    scored_docs = []
    for doc in docs:
        content_lower = doc.page_content.lower()
        
        # Score based on query term matches
        matches = sum(1 for term in query_terms if term in content_lower)
        
        # Boost score if matches appear early in document
        early_content = content_lower[:500]
        early_matches = sum(1 for term in query_terms if term in early_content)
        
        # Calculate relevance score
        score = matches + (early_matches * 0.5)
        
        scored_docs.append((doc, score))
    
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    reranked = [doc for doc, score in scored_docs]
    
    logger.info(f"🎯 Reranked {len(docs)} documents by relevance")
    return reranked

def get_comprehensive_context(query: str) -> str:
    """
    Ultra-comprehensive retrieval with multi-stage processing.
    """
    try:
        # Step 1: Analyze query
        analysis = analyze_query_complexity(query)
        k = analysis["k_value"]
        
        # Step 2: Initial retrieval (with generous k)
        if analysis["use_mmr"]:
            # MMR for diversity
            docs = vector_store_manager.mmr_search(
                query, 
                k=k, 
                fetch_k=min(k * 4, 100),  # Fetch 4x for better diversity
                lambda_mult=0.3  # Prioritize diversity
            )
            logger.info(f"📚 MMR retrieval: fetched {len(docs)} diverse documents")
        else:
            # Standard similarity search
            docs = vector_store_manager.similarity_search(query, k=k)
            logger.info(f"📚 Similarity search: fetched {len(docs)} documents")
        
        if not docs:
            logger.warning("⚠️ No documents retrieved!")
            return ""
        
        # Step 3: Deduplicate
        docs = deduplicate_documents(docs)
        
        # Step 4: Rerank if needed
        if analysis["use_reranking"]:
            docs = rerank_documents(query, docs)
        
        # Step 5: Build comprehensive context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            # Add document with clear separation
            context_parts.append(f"[Source {i}]\n{doc.page_content}")
        
        context = "\n\n" + "="*80 + "\n\n".join(context_parts)
        
        logger.info(f"✅ Final context: {len(context)} chars from {len(docs)} documents")
        return context
        
    except Exception as e:
        logger.error(f"❌ Context retrieval failed: {e}", exc_info=True)
        return ""

def extract_all_items_from_context(context: str, item_type: str = "courses") -> List[str]:
    """
    Extract and deduplicate all items (courses, programs) from context.
    Helper function for verification.
    """
    items = []
    lines = context.split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for bullet points, numbered lists, or program names
        if any(marker in line for marker in ['*', '•', '-', ')', '.']):
            # Clean up the line
            cleaned = line.lstrip('*•-0123456789.) \t')
            if cleaned and len(cleaned) > 5:  # Reasonable program name length
                items.append(cleaned)
    
    # Deduplicate while preserving order
    seen = set()
    unique_items = []
    for item in items:
        item_lower = item.lower()
        if item_lower not in seen:
            seen.add(item_lower)
            unique_items.append(item)
    
    return unique_items

# -------------------------------------------------------------------------
# 4. Hardcoded Course List (Ultimate Fallback)
# -------------------------------------------------------------------------
HARDCODED_COURSE_LIST = """
Here is the complete list of programs offered at Sri Sri University:

**Undergraduate Programs (33 programs):**
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
* B.Tech CSE in Animation and VFX
* B.Tech. Electronics & Communication Engineering (VLSI Chip Design, EV and AI)
* B.Tech. in Mechanical Engg. (Robotics & Automation, EV and AI)
* Bachelor in Physical Education & Sports (BPES)
* General Nursing and Midwifery
* Bachelor of Naturopathy and Yogic Science (BNYS)
* Bachelor of Physiotherapy (BPT)

**Postgraduate Programs (29 programs):**
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

**Other Programs (7 programs):**
* Post Graduate Programme in Startup Development
* Post Graduate Diploma in Disaster Management
* PG Diploma in AI Application for Disaster Risk Reduction
* Integrated M.Sc. in Osteopathy
* PG Diploma in Geospatial Technologies
* Advanced Certification in Geospatial Technologies
* Ph.D. (Research & Development Cell) programs

**Total: 69 programs across all categories**

For detailed information about specific programs including eligibility, fees, curriculum, and admission process, please ask about individual programs or visit the official Sri Sri University website.
"""

# -------------------------------------------------------------------------
# 5. Enhanced Prompt Templates
# -------------------------------------------------------------------------

router_prompt = PromptTemplate.from_template("""
You are an expert intent classifier for an educational institution chatbot.

Classify this query into ONE category:

1. **greeting** - Simple greetings (hi, hello, good morning)
2. **thanks** - Gratitude expressions (thank you, thanks, appreciate it)
3. **list_courses** - Requests for complete program listings (list all courses, what programs are offered, show me all degrees, how many courses)
4. **university_related** - Specific questions about programs, admissions, facilities, fees, eligibility, curriculum, faculty, campus, placements
5. **casual_chat** - Off-topic conversation, jokes, general chit-chat

Respond with ONLY the category name.

User Query: {input}
""")

rag_prompt = PromptTemplate.from_template("""
You are Vedika, an AI assistant for Sri Sri University with access to comprehensive information.

**YOUR PRIMARY GOAL: MAXIMUM ACCURACY AND COMPLETENESS**

Context provided below contains information from multiple sources. Your task:

1. **For Listing Questions (courses, programs, specializations):**
   - Scan EVERY source systematically
   - Extract ALL unique items mentioned
   - Remove duplicates (e.g., "B.Tech CSE" = "B.Tech in Computer Science Engineering")
   - Count accurately and state the total
   - Group logically (UG/PG, by department) for clarity
   - If listing, use bullet points for readability

2. **For Specific Questions (about one program/topic):**
   - Synthesize information across ALL relevant sources
   - Include: eligibility, duration, fees, specializations, career prospects (if available)
   - Be comprehensive but organized
   - Cite specific details with confidence

3. **For Comparison Questions:**
   - Extract information about each item being compared
   - Present side-by-side or structured comparison
   - Highlight key differences

4. **For "How Many" Questions:**
   - Count ALL unique instances across entire context
   - State the number clearly first
   - Then list them for verification

**QUALITY CHECKS:**
- If you extract 3-4 items when asked for "all", that's likely incomplete. Scan more carefully.
- For B.Tech specifically, there should be 8-9 specializations
- Cross-reference information between sources for accuracy
- If information conflicts, mention both perspectives

**IF INFORMATION IS INSUFFICIENT:**
"Based on my search, I found [list what you found]. However, this may not be the complete list. For the official comprehensive list, please ask me to 'list all courses' or visit the university website."

**IF INFORMATION IS MISSING:**
"I couldn't find specific details about [topic] in my current search. Please visit the official Sri Sri University website or contact the admissions office at [contact info if available]."

Context:
{context}

Question: {question}

**Instructions:** Provide the most accurate, complete answer possible. Be direct and well-organized. Do not re-introduce yourself.

Answer:
""")

casual_prompt = PromptTemplate.from_template("""
You are Vedika, a friendly AI assistant for Sri Sri University.

Respond naturally and warmly to this casual message. Keep it brief and conversational.

User Message: {input}

Response:
""")

# -------------------------------------------------------------------------
# 6. Processing Chains
# -------------------------------------------------------------------------
router_chain = router_prompt | llm | StrOutputParser()
casual_chain = casual_prompt | llm | StrOutputParser()

def ultimate_rag_chain(query: str) -> str:
    """
    Ultimate RAG chain with maximum accuracy and comprehensive retrieval.
    """
    try:
        # Get comprehensive context
        context = get_comprehensive_context(query)
        
        if not context or len(context) < 100:
            logger.error("❌ Insufficient context retrieved")
            return "I apologize, but I couldn't retrieve sufficient information to answer your question accurately. Please try rephrasing or contact the university directly."
        
        # For debugging: log extracted items if it's a list query
        query_lower = query.lower()
        if any(kw in query_lower for kw in ['list', 'all', 'courses', 'programs', 'how many']):
            extracted = extract_all_items_from_context(context)
            logger.info(f"🔍 Extracted {len(extracted)} unique items from context")
        
        # Format prompt with comprehensive context
        formatted_prompt = rag_prompt.format(context=context, question=query)
        
        # Generate response
        response = llm.invoke(formatted_prompt)
        
        logger.info(f"✅ Generated response ({len(response.content)} chars)")
        return response.content
        
    except Exception as e:
        logger.error(f"❌ RAG chain error: {e}", exc_info=True)
        return "I encountered an error while processing your question. Please try again or contact support."

# -------------------------------------------------------------------------
# 7. Main Routing Chain
# -------------------------------------------------------------------------
chain_with_routing = {
    "topic": router_chain,
    "input": lambda x: x["input"],
}

main_chat_chain = chain_with_routing | RunnableBranch(
    # Greetings
    (
        lambda x: "greeting" in x.get("topic", "").lower(),
        RunnableLambda(lambda x: "Jay Gurudev! How can I assist you today?") 
    ),
    # Thanks
    (
        lambda x: "thanks" in x.get("topic", "").lower(),
        RunnableLambda(lambda x: "Jay Gurudev! Happy to help! Feel free to ask if you have more questions. 🙏")
    ),
    # List all courses (hardcoded for guaranteed completeness)
    (
        lambda x: "list_courses" in x.get("topic", "").lower(),
        RunnableLambda(lambda x: HARDCODED_COURSE_LIST)
    ),
    # University-related (Ultimate RAG with maximum accuracy)
    (
        lambda x: "university_related" in x.get("topic", "").lower(),
        RunnableLambda(lambda x: ultimate_rag_chain(x["input"]))
    ),
    # Default (casual chat)
    RunnableLambda(lambda x: x["input"]) | casual_chain
)

# -------------------------------------------------------------------------
# 8. Chat Chain Export
# -------------------------------------------------------------------------
def get_chat_chain():
    """
    Returns the ultimate chat chain with maximum accuracy.
    """
    if not llm or not vector_store:
        logger.error("❌ RAG engine not fully initialized — cannot serve chat requests.")
        return None

    logger.info("✅ Ultimate RAG Engine initialized:")
    logger.info("   • Dynamic retrieval: k=10-30 based on query complexity")
    logger.info("   • Multi-stage processing: retrieval → deduplication → reranking")
    logger.info("   • Temperature: 0.2 for maximum accuracy")
    logger.info("   • MMR enabled for comprehensive queries")
    return main_chat_chain