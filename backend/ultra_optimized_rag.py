"""
BALANCED RAG SYSTEM - Speed + Comprehensiveness
Target: 3-5 second responses with full document coverage

Key improvements:
1. Larger context windows with smart chunking
2. Multi-stage retrieval (broad → specific)
3. Parallel processing with better deduplication
4. Adaptive model selection (fast for simple, quality for complex)
5. Better chunk overlap and semantic coherence
"""

import os
import json
import hashlib
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from backend.vectorstore import VectorStore
from backend.config import OPENAI_API_KEY, GENERATION_MODEL_NAME
from backend.utils.logger import get_logger
from datetime import datetime, timedelta
import time
from functools import lru_cache

logger = get_logger("balanced_rag")

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
class BalancedConfig:
    # LLM Settings - Adaptive model selection
    FAST_MODEL = "gpt-3.5-turbo-1106"
    QUALITY_MODEL = "gpt-4-turbo-preview"
    USE_ADAPTIVE_MODEL = True  # Choose model based on query complexity
    
    # Retrieval Settings - MORE COMPREHENSIVE
    MAX_DOCS_COMPREHENSIVE = 40   # Increased from 20
    MAX_DOCS_SPECIFIC = 15        # Increased from 6
    MAX_DOCS_GENERAL = 20         # Increased from 8
    
    # Multi-stage retrieval
    ENABLE_MULTI_STAGE = True
    INITIAL_BROAD_K = 50          # First pass: broad retrieval
    RERANK_TOP_K = 30             # Second pass: rerank and select top
    
    # Context Settings - LARGER WINDOWS
    MAX_CHARS_PER_DOC = 1200      # Increased from 600
    MAX_TOTAL_CONTEXT = 16000     # Increased from 8000 (GPT-4 can handle more)
    MIN_CHARS_PER_DOC = 200       # Don't include tiny snippets
    
    # Chunk overlap for better context
    ENABLE_CONTEXT_EXPANSION = True
    EXPANSION_CHARS = 200         # Add 200 chars before/after match
    
    # Cache Settings
    ENABLE_MEMORY_CACHE = True
    ENABLE_DISK_CACHE = True
    CACHE_TTL_HOURS = 24
    
    # Processing Settings
    PARALLEL_PROCESSING = True
    MAX_WORKERS = 8               # Increased from 5
    
    # Smart filtering
    ENABLE_RELEVANCE_THRESHOLD = True
    MIN_RELEVANCE_SCORE = 0.3     # Filter low-quality matches

config = BalancedConfig()

# -------------------------------------------------------------------------
# Ultra-Fast Response Cache (moved from ultra_optimized_rag)
# -------------------------------------------------------------------------
class UltraFastCache:
    """Optimized caching with LRU memory cache"""
    
    def __init__(self, cache_dir: str = "./fast_cache", max_memory_items: int = 100):
        self.cache_dir = cache_dir
        self.memory_cache = {}  # Hot cache
        self.max_memory_items = max_memory_items
        self.access_count = {}  # Track popular queries
        self.ttl = timedelta(hours=config.CACHE_TTL_HOURS)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load popular queries into memory on startup
        self._warm_memory_cache()
    
    def _warm_memory_cache(self):
        """Load most accessed queries into memory"""
        if not config.ENABLE_MEMORY_CACHE:
            return
            
        try:
            # Load access stats if exists
            stats_file = os.path.join(self.cache_dir, "_stats.json")
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    self.access_count = json.load(f)
                
                # Load top 20 into memory
                top_queries = sorted(self.access_count.items(), 
                                   key=lambda x: x[1], reverse=True)[:20]
                
                for cache_key, _ in top_queries:
                    cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
                    if os.path.exists(cache_file):
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            self.memory_cache[cache_key] = {
                                'response': data['response'],
                                'timestamp': datetime.fromisoformat(data['timestamp'])
                            }
                
                logger.info(f"🔥 Warmed cache with {len(self.memory_cache)} popular queries")
        except Exception as e:
            logger.error(f"Cache warming error: {e}")
    
    def _get_cache_key(self, query: str) -> str:
        # Normalize query for better cache hits
        normalized = query.lower().strip()
        # Remove extra spaces
        normalized = ' '.join(normalized.split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, query: str) -> str | None:
        """Ultra-fast cache retrieval"""
        cache_key = self._get_cache_key(query)
        
        # Memory cache (instant)
        if config.ENABLE_MEMORY_CACHE and cache_key in self.memory_cache:
            data = self.memory_cache[cache_key]
            if datetime.now() - data['timestamp'] < self.ttl:
                self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
                logger.info(f"⚡ CACHE HIT (memory): {query[:50]}...")
                return data['response']
        
        # Disk cache
        if config.ENABLE_DISK_CACHE:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    if datetime.now() - timestamp < self.ttl:
                        # Promote to memory cache
                        if config.ENABLE_MEMORY_CACHE:
                            self._promote_to_memory(cache_key, data['response'], timestamp)
                        
                        self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
                        logger.info(f"⚡ CACHE HIT (disk): {query[:50]}...")
                        return data['response']
                except Exception as e:
                    logger.error(f"Cache read error: {e}")
        
        return None
    
    def _promote_to_memory(self, cache_key: str, response: str, timestamp: datetime):
        """Move hot cache item to memory"""
        # Evict least accessed if full
        if len(self.memory_cache) >= self.max_memory_items:
            least_accessed = min(self.access_count.items(), key=lambda x: x[1])[0]
            self.memory_cache.pop(least_accessed, None)
        
        self.memory_cache[cache_key] = {
            'response': response,
            'timestamp': timestamp
        }
    
    def set(self, query: str, response: str):
        """Cache a response"""
        cache_key = self._get_cache_key(query)
        timestamp = datetime.now()
        
        # Always save to memory if enabled
        if config.ENABLE_MEMORY_CACHE:
            self._promote_to_memory(cache_key, response, timestamp)
        
        # Save to disk if enabled
        if config.ENABLE_DISK_CACHE:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'query': query,
                        'response': response,
                        'timestamp': timestamp.isoformat()
                    }, f, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Cache write error: {e}")
        
        # Update access stats
        self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
        
        # Periodically save stats
        if len(self.access_count) % 10 == 0:
            self._save_stats()
    
    def _save_stats(self):
        """Save access statistics"""
        try:
            stats_file = os.path.join(self.cache_dir, "_stats.json")
            with open(stats_file, 'w') as f:
                json.dump(self.access_count, f)
        except Exception as e:
            logger.error(f"Stats save error: {e}")

# -------------------------------------------------------------------------
# Enhanced Query Classification
# -------------------------------------------------------------------------
@lru_cache(maxsize=1000)
def classify_query_enhanced(query: str) -> Tuple[str, int, str]:
    """
    Enhanced query classification
    Returns: (query_type, optimal_k, model_preference)
    """
    query_lower = query.lower()
    
    # NEW: Detect "current" or "who is" queries - need precise, single answer
    current_keywords = ['current', 'who is', 'present', 'now', 'today', 'latest']
    is_current_query = any(kw in query_lower for kw in current_keywords)
    
    # Dean/HOD/faculty queries - need fewer, more precise docs
    if any(kw in query_lower for kw in ['dean', 'head of department', 'hod', 'director', 'principal']):
        if is_current_query:
            return 'current_person', 3, 'fast'  # Only 3 docs for precision
        return 'person', 8, 'fast'
    
    # Comprehensive list queries
    list_keywords = [
        'all courses', 'all programs', 'list all', 'complete list',
        'what courses', 'what programs', 'which courses', 'which programs',
        'courses offered', 'programs offered', 'courses available', 'programs available',
        'how many courses', 'how many programs', 'tell me about courses', 
        'tell me about programs', 'show me courses', 'show me programs',
        'what are the courses', 'what are the programs', 'available courses',
        'available programs', 'course list', 'program list'
    ]
    
    if any(kw in query_lower for kw in list_keywords):
        return 'list', config.MAX_DOCS_COMPREHENSIVE, 'quality'
    
    # Greeting/casual - fast model
    if any(kw in query_lower for kw in ['hi', 'hello', 'hey', 'thanks', 'thank you']):
        return 'casual', 0, 'fast'
    
    # Complex queries - quality model + more docs
    complexity_indicators = [
        'compare', 'difference', 'vs', 'versus', 'better',
        'detailed', 'explain', 'comprehensive', 'about',
        'eligibility and', 'fees and', 'curriculum and',
        'what all', 'how to', 'process for'
    ]
    
    is_complex = any(kw in query_lower for kw in complexity_indicators)
    
    # Specific program queries
    programs = ['btech', 'mtech', 'mba', 'bba', 'bca', 'mca', 'bsc', 'msc', 
                'phd', 'diploma', 'bnys', 'bams', 'bpt']
    has_program = any(prog in query_lower for prog in programs)
    
    if has_program:
        if is_complex:
            return 'program_detailed', config.MAX_DOCS_SPECIFIC, 'quality'
        return 'program', config.MAX_DOCS_GENERAL, 'fast'
    
    # Comparison queries - need more context
    if any(kw in query_lower for kw in ['compare', 'difference', 'vs', 'versus']):
        return 'comparison', config.MAX_DOCS_COMPREHENSIVE, 'quality'
    
    # Admission/fees queries - often need multiple docs
    if any(kw in query_lower for kw in ['admission', 'eligibility', 'fee', 'cost', 'apply']):
        return 'admission', config.MAX_DOCS_GENERAL, 'fast'
    
    # General queries
    if is_complex:
        return 'general_complex', config.MAX_DOCS_GENERAL, 'quality'
    
    return 'general', config.MAX_DOCS_GENERAL, 'fast'

# -------------------------------------------------------------------------
# Enhanced Retrieval Engine
# -------------------------------------------------------------------------
class EnhancedRetriever:
    """Comprehensive retrieval with multi-stage search"""
    
    def __init__(self, vector_store_manager: VectorStore):
        self.vector_store = vector_store_manager
        
        # Expanded topic mapping
        self.topic_map = {
            'engineering': ['btech', 'b.tech', 'mtech', 'm.tech', 'engineering', 
                          'cse', 'ece', 'mechanical', 'electrical', 'civil'],
            'management': ['mba', 'bba', 'management', 'business', 'agribusiness', 'fintech'],
            'computer': ['bca', 'mca', 'computer', 'data science', 'cyber security', 
                        'artificial intelligence', 'machine learning'],
            'science': ['bsc', 'msc', 'science', 'physics', 'chemistry', 'environmental'],
            'medical': ['bams', 'bnys', 'bpt', 'nursing', 'optometry', 'physiotherapy', 
                       'ayurveda', 'naturopathy'],
            'arts': ['ba', 'ma', 'english', 'sanskrit', 'performing arts', 'fine arts', 
                    'interior design'],
            'agriculture': ['agriculture', 'horticulture', 'agronomy', 'agribusiness', 
                           'soil science', 'plant breeding'],
            'yoga': ['yoga', 'yogic science', 'contemplative'],
            'admissions': ['admission', 'apply', 'eligibility', 'entrance', 'application'],
            'fees': ['fee', 'cost', 'tuition', 'scholarship', 'financial'],
            'campus': ['campus', 'facility', 'hostel', 'library', 'sports'],
            'placements': ['placement', 'career', 'job', 'recruitment', 'salary']
        }
    
    def identify_topics_comprehensive(self, query: str) -> List[str]:
        """Identify all relevant topics from query"""
        query_lower = query.lower()
        topics = []
        
        for topic, keywords in self.topic_map.items():
            if any(kw in query_lower for kw in keywords):
                topics.append(topic)
        
        # If no topics found, search broadly
        if not topics:
            topics = ['engineering', 'management', 'admissions', 'campus']
        
        return topics
    
    async def multi_stage_retrieval(self, query: str, k: int) -> List[Document]:
        """
        Multi-stage retrieval for comprehensive coverage
        Stage 1: Broad retrieval (50 docs)
        Stage 2: Rerank and filter (top 30)
        Stage 3: Deduplicate and select final k
        """
        if k == 0:
            return []
        
        if not config.ENABLE_MULTI_STAGE:
            return await self.retrieve_parallel(query, k)
        
        try:
            # Stage 1: Broad retrieval
            logger.info(f"🔍 Stage 1: Broad retrieval (k={config.INITIAL_BROAD_K})")
            initial_docs = await self.retrieve_parallel(query, config.INITIAL_BROAD_K)
            
            if not initial_docs:
                return []
            
            # Stage 2: Score and rerank
            logger.info(f"📊 Stage 2: Reranking {len(initial_docs)} documents")
            scored_docs = await self.score_documents(query, initial_docs)
            
            # Filter by relevance threshold
            if config.ENABLE_RELEVANCE_THRESHOLD:
                scored_docs = [
                    (doc, score) for doc, score in scored_docs 
                    if score >= config.MIN_RELEVANCE_SCORE
                ]
            
            # Sort by score and take top RERANK_TOP_K
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            reranked_docs = [doc for doc, score in scored_docs[:config.RERANK_TOP_K]]
            
            # Stage 3: Smart deduplication and selection
            logger.info(f"✂️ Stage 3: Deduplication and selection (target k={k})")
            final_docs = self.smart_deduplication(reranked_docs, k)
            
            logger.info(f"✅ Retrieved {len(final_docs)} high-quality documents")
            return final_docs
            
        except Exception as e:
            logger.error(f"Multi-stage retrieval error: {e}")
            # Fallback to simple retrieval
            return await self.retrieve_parallel(query, k)
    
    async def retrieve_parallel(self, query: str, k: int) -> List[Document]:
        """Parallel retrieval across multiple topics"""
        if k == 0:
            return []
        
        topics = self.identify_topics_comprehensive(query)
        
        if len(topics) == 1:
            # Single topic - direct search
            docs = await asyncio.to_thread(
                self.vector_store.similarity_search,
                query,
                k=k
            )
            return docs
        
        # Multiple topics - parallel search with more docs per topic
        docs_per_topic = max(5, k // len(topics) + 2)
        
        tasks = []
        for topic in topics:
            topic_query = f"{topic} {query}"
            task = asyncio.to_thread(
                self.vector_store.similarity_search,
                topic_query,
                k=docs_per_topic
            )
            tasks.append(task)
        
        # Also search with original query
        tasks.append(
            asyncio.to_thread(
                self.vector_store.similarity_search,
                query,
                k=docs_per_topic
            )
        )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine and deduplicate
        all_docs = []
        for result in results:
            if isinstance(result, list):
                all_docs.extend(result)
        
        return self.smart_deduplication(all_docs, k)
    
    async def score_documents(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        """
        Score documents based on relevance with temporal awareness
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Detect if query is asking for "current" information
        current_keywords = ['current', 'who is', 'present', 'now', 'today', 'latest', 'currently']
        is_current_query = any(kw in query_lower for kw in current_keywords)
        
        # Keywords that indicate historical/old information
        historical_keywords = ['previous', 'former', 'ex-', 'was', 'retired', 'until', 'past', 
                              'old', 'earlier', 'before', '2020', '2021', '2022', '2023']
        
        scored = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            content_words = set(content_lower.split())
            
            # Base scoring: word overlap
            overlap = len(query_words.intersection(content_words))
            score = overlap / len(query_words) if query_words else 0
            
            # Boost if exact phrase match
            if query_lower in content_lower:
                score += 0.5
            
            # TEMPORAL FILTERING for "current" queries
            if is_current_query:
                # Heavily penalize historical indicators
                has_historical = any(kw in content_lower for kw in historical_keywords)
                if has_historical:
                    score *= 0.1  # Reduce score by 90%
                    logger.debug(f"Penalized historical document: {content_lower[:100]}")
                
                # Boost documents with "current" indicators
                current_boost_keywords = ['current', 'present', 'currently', '2024', '2025', 'now']
                has_current = any(kw in content_lower for kw in current_boost_keywords)
                if has_current:
                    score *= 2.0  # Double the score
                    logger.debug(f"Boosted current document: {content_lower[:100]}")
            
            scored.append((doc, score))
        
        return scored
    
    def smart_deduplication(self, docs: List[Document], target_k: int) -> List[Document]:
        """
        Smart deduplication with semantic similarity
        """
        if not docs:
            return []
        
        unique_docs = []
        seen_signatures = set()
        
        for doc in docs:
            # Create signature from first 200 chars
            signature = doc.page_content[:200].strip()
            
            # More lenient deduplication - check similarity, not exact match
            is_duplicate = False
            for seen_sig in seen_signatures:
                # Simple similarity: check if 80% of signature is in seen
                if len(signature) > 50:  # Only check meaningful signatures
                    overlap = sum(1 for i, char in enumerate(signature[:100]) 
                                 if i < len(seen_sig) and seen_sig[i] == char)
                    similarity = overlap / min(len(signature[:100]), len(seen_sig[:100]))
                    if similarity > 0.8:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                # Expand context if enabled
                if config.ENABLE_CONTEXT_EXPANSION:
                    doc = self.expand_document_context(doc)
                
                unique_docs.append(doc)
                seen_signatures.add(signature)
                
                if len(unique_docs) >= target_k:
                    break
        
        return unique_docs
    
    def expand_document_context(self, doc: Document) -> Document:
        """
        Expand document with surrounding context (if available in metadata)
        This helps maintain semantic coherence
        """
        # This is a placeholder - in practice, you'd fetch surrounding chunks
        # from the original document based on metadata like chunk_id
        return doc

# -------------------------------------------------------------------------
# Hardcoded Responses
# -------------------------------------------------------------------------
HARDCODED_COURSE_LIST = """**Programs at Sri Sri University**

**🎓 Undergraduate Programs (40 programs):**
• Bachelor of Architecture
• Bachelor of Business Administration (BBA)
• BBA Logistics Management
• Bachelor of Ayurveda, Medicine and Surgery (BAMS)
• Bachelor in Computer Application (BCA)
• BCA in Data Science and Analytics
• BCA Cyber Security
• B.Com (Hons.)
• B.Sc. Osteopathy
• B.Sc. Yoga
• B.Sc. (Hons.) in Agriculture
• B.Sc. (Hons.) Horticulture
• B.Sc. (Hons.) Agribusiness
• B.Sc. (Hons.) Food Nutrition and Dietetics
• B.Sc. (Hons.) Physics
• B.Sc. (Hons.) Computer Science
• B.Sc. (Hons.) Environmental Science
• B.Sc. (Hons.) in Psychology and Contemplative Studies
• B.Sc. Forensic Science
• B.Sc. in Forensic Science (Specialization in Digital Forensics & Cyber Forensics)
• B.Sc. Nursing
• B.Sc. in Optometry
• Bachelor of Fine Arts
• Bachelor in Interior Design
• Bachelor of Performing Arts (Odissi Dance)
• Bachelor of Performing Arts (Hindustani Vocal)
• B.A. (Hons.) English
• B.Tech. Computer Science Engineering
• B.Tech CSE (AI and Machine Learning)
• B.Tech CSE (Data Science)
• B.Tech CSE (Cyber Security and Cyber Defence)
• B.Tech. in CSE – Cyber Security & Cyber Defense (Apprentices Embedded)
• B.Tech. Electrical Engineering (EV, AI and Renewable Energy)
• B.Tech CSE in Animation and VFX
• B.Tech. Electronics & Communication Engineering (VLSI Chip Design, EV and AI)
• B.Tech. in Mechanical Engg. (Robotics & Automation, EV and AI)
• Bachelor in Physical Education & Sports (BPES)
• General Nursing and Midwifery
• Bachelor of Naturopathy and Yogic Science (BNYS)
• Bachelor of Physiotherapy (BPT)

**🎓 Postgraduate Programs (32 programs):**
• Master of Business Administration (MBA)
• MBA Agribusiness
• MBA Fintech
• Global MBA
• M.Sc. Osteopathy
• M.Sc. Yoga
• MA Yoga
• M.A. (Sanskrit)
• MPA (Odissi Dance)
• M.A. (English)
• M.A. in Hindu Studies
• M.Sc in Psychology and Contemplative Studies
• Master of Computer Application (MCA)
• MCA – Cyber Security (Apprentices Embedded)
• MCA with specialization in Cyber Security & Incident Management
• MCA in Data Science and Analytics
• M.Sc. Agronomy
• M.Sc. Soil Science and Agricultural Chemistry
• M.Sc. Genetics and Plant Breeding
• M.Sc. Entomology
• M.Sc. Horticulture
• M.Sc. Agricultural Statistics
• M.Sc. Agricultural Economics
• M.Sc. Agricultural Extension
• M.Sc. Plant Pathology
• MSc in Exercise & Sports Physiology
• M.Sc. in Cyber Security & Digital Forensic
• M.Sc. in Artificial Intelligence & Machine Learning
• M.Tech. – Cyber Security (Apprentices Embedded)
• M.Tech. in Cyber Security & Digital Forensic
• M.Tech. in Artificial Intelligence & Machine Learning
• MTech in Cybersecurity & Incident Management

**🎓 Other Programs (7 programs):**
• Post Graduate Programme in Startup Development
• Post Graduate Diploma in Disaster Management
• PG Diploma in AI Application for Disaster Risk Reduction
• Integrated M.Sc. in Osteopathy
• PG Diploma in Geospatial Technologies
• Advanced Certification in Geospatial Technologies
• Ph.D. (Research & Development Cell) programs

**Total: 79 programs across all categories**"""

CASUAL_RESPONSES = {
    'greeting': "Jay Gurudev! 🙏 How can I assist you today?",
    'thanks': "Jay Gurudev! Happy to help! Feel free to ask more questions. 🙏"
}

# -------------------------------------------------------------------------
# Initialize System with Adaptive Models
# -------------------------------------------------------------------------
if not OPENAI_API_KEY:
    raise EnvironmentError("❌ OPENAI_API_KEY not found")

# Initialize both models
fast_llm = ChatOpenAI(
    model=config.FAST_MODEL,
    temperature=0.2,
    api_key=OPENAI_API_KEY,
    streaming=True,
    request_timeout=30
)

quality_llm = ChatOpenAI(
    model=config.QUALITY_MODEL,
    temperature=0.2,
    api_key=OPENAI_API_KEY,
    streaming=True,
    request_timeout=60
)

logger.info(f"🚀 Models initialized: Fast={config.FAST_MODEL}, Quality={config.QUALITY_MODEL}")

vector_store_manager = VectorStore()
vector_store = vector_store_manager.load_store()

retriever = EnhancedRetriever(vector_store_manager)
cache = UltraFastCache(cache_dir="./fast_cache", max_memory_items=100)

# -------------------------------------------------------------------------
# Enhanced RAG Prompt with Current Information Focus
# -------------------------------------------------------------------------
balanced_prompt = PromptTemplate.from_template("""
You are Vedika, a comprehensive AI assistant for Sri Sri University with access to extensive university information.

You have retrieved {num_sources} relevant documents to answer this question comprehensively.

Context from multiple sources:
{context}

Question: {question}

**INSTRUCTIONS:**

1. **For "current" or "who is" questions (dean, HOD, director, etc.):**
   - ONLY provide the CURRENT person's name
   - If multiple names appear in context, prioritize information marked as "current", "present", or with recent dates (2024-2025)
   - IGNORE any names marked as "previous", "former", "ex-", "retired", or with old dates
   - Give a direct answer: "The current [position] is [Name]"
   - Add brief additional details if available (qualifications, background)
   - DO NOT list historical holders of the position

2. **Provide a COMPREHENSIVE answer** using ALL relevant information from the context
3. **Maintain professional formatting** with proper structure:
   - Use **section headers** for different topics
   - Use bullet points for lists
   - Use emojis sparingly (🎓 📚 💰 ⏱️ 📋 🏫)
   - Break into readable paragraphs (2-4 sentences each)

4. **For program information, include:**
   - **About the Program** - Overview and key features
   - **📋 Eligibility Criteria** - Entry requirements
   - **⏱️ Duration** - Program length
   - **💰 Fee Structure** - Cost information (or mention to contact admissions)
   - **📚 Curriculum Highlights** - Key subjects/specializations
   - **🎯 Career Opportunities** - Future prospects

5. **For comprehensive queries:**
   - Cover all aspects mentioned in the context
   - Group related information together
   - Provide examples when available

6. **Critical rules:**
   - DO NOT say "based on the provided context" or similar phrases
   - DO NOT add closing lines like "feel free to ask more"
   - If specific details are missing, say "Please contact the admissions office for specific details"
   - Be direct, clear, and well-organized
   - For current information queries, prioritize recent/current data and ignore historical data

**Tone:** Professional, thorough, helpful, and easy to read.

Comprehensive Answer:
""")

# -------------------------------------------------------------------------
# Balanced RAG Chain
# -------------------------------------------------------------------------
async def balanced_rag(query: str) -> Tuple[str, Dict[str, Any]]:
    """
    Balanced RAG: Comprehensive retrieval with reasonable speed (3-5s)
    """
    start_time = time.time()
    metadata = {'steps': [], 'comprehensive': True}
    
    try:
        # Step 1: Check cache
        t1 = time.time()
        cached = cache.get(query)
        if cached:
            metadata['cache_hit'] = True
            metadata['total_time'] = time.time() - start_time
            metadata['steps'].append(f"cache_hit: {time.time()-t1:.3f}s")
            logger.info(f"⚡ CACHED: {metadata['total_time']*1000:.1f}ms")
            return cached, metadata
        metadata['steps'].append(f"cache_miss: {time.time()-t1:.3f}s")
        
        # Step 2: Enhanced query classification
        t2 = time.time()
        query_type, k, model_pref = classify_query_enhanced(query)
        metadata['query_type'] = query_type
        metadata['k'] = k
        metadata['model_preference'] = model_pref
        metadata['steps'].append(f"classify: {time.time()-t2:.3f}s")
        
        # Handle special cases
        if query_type == 'list':
            cache.set(query, HARDCODED_COURSE_LIST)
            metadata['total_time'] = time.time() - start_time
            return HARDCODED_COURSE_LIST, metadata
        
        if query_type == 'casual':
            response = CASUAL_RESPONSES.get('greeting', "Hello!")
            metadata['total_time'] = time.time() - start_time
            return response, metadata
        
        # Step 3: Multi-stage retrieval
        t3 = time.time()
        docs = await retriever.multi_stage_retrieval(query, k)
        retrieval_time = time.time() - t3
        metadata['steps'].append(f"retrieval: {retrieval_time:.3f}s")
        metadata['docs_retrieved'] = len(docs)
        
        if not docs:
            response = "I couldn't find specific information. Please contact the university."
            metadata['total_time'] = time.time() - start_time
            return response, metadata
        
        # Step 4: Build comprehensive context
        t4 = time.time()
        context_parts = []
        total_chars = 0
        
        for i, doc in enumerate(docs, 1):
            content = doc.page_content
            
            # Filter tiny snippets
            if len(content) < config.MIN_CHARS_PER_DOC:
                continue
            
            # Truncate to max chars per doc
            content = content[:config.MAX_CHARS_PER_DOC]
            
            # Check if adding this would exceed total limit
            if total_chars + len(content) > config.MAX_TOTAL_CONTEXT:
                # Try to fit partial content
                remaining = config.MAX_TOTAL_CONTEXT - total_chars
                if remaining > config.MIN_CHARS_PER_DOC:
                    content = content[:remaining]
                else:
                    break
            
            context_parts.append(f"[Source {i}]\n{content}")
            total_chars += len(content)
        
        context = "\n\n---\n\n".join(context_parts)
        metadata['steps'].append(f"context: {time.time()-t4:.3f}s")
        metadata['context_length'] = len(context)
        metadata['num_sources'] = len(context_parts)
        
        # Step 5: Select appropriate model and generate
        t5 = time.time()
        selected_llm = quality_llm if model_pref == 'quality' else fast_llm
        metadata['model_used'] = config.QUALITY_MODEL if model_pref == 'quality' else config.FAST_MODEL
        
        formatted_prompt = balanced_prompt.format(
            num_sources=len(context_parts),
            context=context,
            question=query
        )
        
        response = await selected_llm.ainvoke(formatted_prompt)
        response_text = response.content
        llm_time = time.time() - t5
        metadata['steps'].append(f"llm: {llm_time:.3f}s")
        
        # Step 6: Post-process for "current" queries
        if query_type == 'current_person':
            response_text = filter_current_information(response_text, query)
        
        # Step 7: Cache
        cache.set(query, response_text)
        
        metadata['total_time'] = time.time() - start_time
        
        logger.info(f"✅ Balanced RAG: {metadata['total_time']:.2f}s | "
                   f"Docs: {len(docs)} | Model: {metadata['model_used']}")
        
        return response_text, metadata
        
    except Exception as e:
        logger.error(f"❌ Balanced RAG error: {e}", exc_info=True)
        metadata['error'] = str(e)
        metadata['total_time'] = time.time() - start_time
        return "Error processing your question. Please try again.", metadata

# -------------------------------------------------------------------------
# Export Functions
# -------------------------------------------------------------------------
def get_balanced_chain():
    """Get the balanced RAG chain"""
    return balanced_rag

def clear_cache():
    """Clear all caches"""
    cache.memory_cache.clear()
    cache.access_count.clear()
    logger.info("✅ Cache cleared")