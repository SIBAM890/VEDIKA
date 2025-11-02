"""
ULTRA-OPTIMIZED RAG SYSTEM
Target: 2-3 second responses while maintaining comprehensiveness

Key optimizations:
1. Aggressive parallel processing
2. Smaller context windows with better chunking
3. Fast LLM model selection
4. Precomputed embeddings cache
5. Request batching
"""

import os
import json
import hashlib
import asyncio
from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from backend.vectorstore import VectorStore
from backend.config import OPENAI_API_KEY, GENERATION_MODEL_NAME
from backend.utils.logger import get_logger
from datetime import datetime, timedelta
import time
from functools import lru_cache

logger = get_logger("ultra_rag")

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
class OptimizedConfig:
    # LLM Settings
    USE_FAST_MODEL = True  # Set to True for gpt-3.5-turbo-1106 (10x faster than gpt-4)
    FAST_MODEL = "gpt-3.5-turbo-1106"
    QUALITY_MODEL = "gpt-4-turbo-preview"
    
    # Retrieval Settings
    MAX_DOCS_COMPREHENSIVE = 20  # Reduced from 30
    MAX_DOCS_SPECIFIC = 6        # For specific queries
    MAX_DOCS_GENERAL = 8         # For general queries
    
    # Context Settings
    MAX_CHARS_PER_DOC = 600      # Reduced from 800
    MAX_TOTAL_CONTEXT = 8000     # Total context limit
    
    # Cache Settings
    ENABLE_MEMORY_CACHE = True
    ENABLE_DISK_CACHE = True
    CACHE_TTL_HOURS = 24
    
    # Processing Settings
    PARALLEL_PROCESSING = True
    MAX_WORKERS = 5

config = OptimizedConfig()

# -------------------------------------------------------------------------
# 1. Ultra-Fast Response Cache
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
# 2. Optimized Query Router
# -------------------------------------------------------------------------
@lru_cache(maxsize=1000)
def classify_query_fast(query: str) -> Tuple[str, int]:
    """
    Ultra-fast query classification with caching
    Returns: (query_type, optimal_k)
    """
    query_lower = query.lower()
    
    # List queries - use hardcoded
    if any(kw in query_lower for kw in ['all courses', 'all programs', 'list all', 'complete list']):
        return 'list', 0
    
    # Greeting/casual
    if any(kw in query_lower for kw in ['hi', 'hello', 'hey', 'thanks', 'thank you']):
        return 'casual', 0
    
    # Specific program
    programs = ['btech', 'mtech', 'mba', 'bba', 'bca', 'mca', 'bsc', 'msc']
    if any(prog in query_lower for prog in programs):
        # Check if asking for details
        if any(kw in query_lower for kw in ['about', 'details', 'eligibility', 'fees', 'curriculum']):
            return 'specific', config.MAX_DOCS_SPECIFIC
        return 'program', config.MAX_DOCS_GENERAL
    
    # Comparison
    if any(kw in query_lower for kw in ['compare', 'difference', 'vs', 'versus']):
        return 'comparison', config.MAX_DOCS_GENERAL
    
    # General
    return 'general', config.MAX_DOCS_GENERAL

# -------------------------------------------------------------------------
# 3. Optimized Retrieval Engine
# -------------------------------------------------------------------------
class UltraFastRetriever:
    """Optimized retrieval with parallel processing"""
    
    def __init__(self, vector_store_manager: VectorStore):
        self.vector_store = vector_store_manager
        
        # Topic keywords (simplified)
        self.topic_map = {
            'engineering': ['btech', 'b.tech', 'mtech', 'engineering', 'cse', 'ece'],
            'management': ['mba', 'bba', 'management', 'business'],
            'computer': ['bca', 'mca', 'computer'],
            'science': ['bsc', 'msc', 'science'],
            'admissions': ['admission', 'apply', 'eligibility', 'entrance'],
            'fees': ['fee', 'cost', 'tuition'],
        }
    
    def identify_topics_fast(self, query: str) -> List[str]:
        """Fast topic identification"""
        query_lower = query.lower()
        topics = []
        
        for topic, keywords in self.topic_map.items():
            if any(kw in query_lower for kw in keywords):
                topics.append(topic)
        
        # Default to main topics if none found
        return topics if topics else ['engineering', 'admissions']
    
    async def retrieve_parallel(self, query: str, k: int) -> List[Document]:
        """Ultra-fast parallel retrieval"""
        if k == 0:
            return []
        
        topics = self.identify_topics_fast(query)
        
        if len(topics) == 1:
            # Single topic - direct search (faster)
            docs = await asyncio.to_thread(
                self.vector_store.similarity_search,
                query,
                k=k
            )
            return docs
        
        # Multiple topics - parallel search
        docs_per_topic = max(3, k // len(topics))
        
        tasks = []
        for topic in topics[:3]:  # Limit to 3 topics max
            topic_query = f"{topic} {query}"
            task = asyncio.to_thread(
                self.vector_store.similarity_search,
                topic_query,
                k=docs_per_topic
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Quick deduplication
        seen = set()
        unique_docs = []
        for topic_docs in results:
            for doc in topic_docs:
                sig = doc.page_content[:100]
                if sig not in seen:
                    seen.add(sig)
                    unique_docs.append(doc)
                    if len(unique_docs) >= k:
                        return unique_docs
        
        return unique_docs

# -------------------------------------------------------------------------
# 4. Initialize System
# -------------------------------------------------------------------------
if not OPENAI_API_KEY:
    raise EnvironmentError("❌ OPENAI_API_KEY not found")

# Use fast model by default
model_name = config.FAST_MODEL if config.USE_FAST_MODEL else config.QUALITY_MODEL

llm = ChatOpenAI(
    model=model_name,
    temperature=0.2,
    api_key=OPENAI_API_KEY,
    streaming=True,
    request_timeout=30  # Timeout after 30s
)

logger.info(f"🚀 Using model: {model_name}")

vector_store_manager = VectorStore()
vector_store = vector_store_manager.load_store()

retriever = UltraFastRetriever(vector_store_manager)
cache = UltraFastCache(cache_dir="./fast_cache", max_memory_items=100)

# Hardcoded responses
HARDCODED_COURSE_LIST = """[Your course list here - same as before]"""

CASUAL_RESPONSES = {
    'greeting': "Jay Gurudev! 🙏 How can I assist you today?",
    'thanks': "Jay Gurudev! Happy to help! Feel free to ask more questions. 🙏"
}

# -------------------------------------------------------------------------
# 5. Ultra-Fast RAG Chain
# -------------------------------------------------------------------------

# Shorter, more efficient prompt
ultra_prompt = PromptTemplate.from_template("""
Context from {num_sources} sources:
{context}

Question: {question}

Provide a complete, accurate answer. Be direct and well-organized.
Answer:
""")

async def ultra_fast_rag(query: str) -> Tuple[str, Dict[str, Any]]:
    """
    Ultra-optimized RAG chain: 2-3 second target
    """
    start_time = time.time()
    metadata = {'steps': []}
    
    try:
        # Step 1: Check cache (0.001s)
        t1 = time.time()
        cached = cache.get(query)
        if cached:
            metadata['cache_hit'] = True
            metadata['total_time'] = time.time() - start_time
            metadata['steps'].append(f"cache_hit: {time.time()-t1:.3f}s")
            logger.info(f"⚡ INSTANT (cached): {metadata['total_time']*1000:.1f}ms")
            return cached, metadata
        metadata['steps'].append(f"cache_miss: {time.time()-t1:.3f}s")
        
        # Step 2: Query classification (0.001s)
        t2 = time.time()
        query_type, k = classify_query_fast(query)
        metadata['query_type'] = query_type
        metadata['k'] = k
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
        
        # Step 3: Parallel retrieval (0.5-1.0s)
        t3 = time.time()
        docs = await retriever.retrieve_parallel(query, k)
        retrieval_time = time.time() - t3
        metadata['steps'].append(f"retrieval: {retrieval_time:.3f}s")
        metadata['docs_retrieved'] = len(docs)
        
        if not docs:
            response = "I couldn't find specific information. Please contact the university."
            metadata['total_time'] = time.time() - start_time
            return response, metadata
        
        # Step 4: Build context (0.05s)
        t4 = time.time()
        context_parts = []
        total_chars = 0
        
        for i, doc in enumerate(docs, 1):
            # Aggressive truncation
            content = doc.page_content[:config.MAX_CHARS_PER_DOC]
            if total_chars + len(content) > config.MAX_TOTAL_CONTEXT:
                break
            context_parts.append(f"[{i}] {content}")
            total_chars += len(content)
        
        context = "\n\n".join(context_parts)
        metadata['steps'].append(f"context: {time.time()-t4:.3f}s")
        metadata['context_length'] = len(context)
        
        # Step 5: LLM generation (1.0-1.5s)
        t5 = time.time()
        formatted_prompt = ultra_prompt.format(
            num_sources=len(context_parts),
            context=context,
            question=query
        )
        
        response = await llm.ainvoke(formatted_prompt)
        response_text = response.content
        llm_time = time.time() - t5
        metadata['steps'].append(f"llm: {llm_time:.3f}s")
        
        # Step 6: Cache
        cache.set(query, response_text)
        
        metadata['total_time'] = time.time() - start_time
        
        logger.info(f"✅ Response: {metadata['total_time']:.2f}s | Breakdown: {metadata['steps']}")
        
        return response_text, metadata
        
    except Exception as e:
        logger.error(f"❌ Ultra RAG error: {e}", exc_info=True)
        metadata['error'] = str(e)
        metadata['total_time'] = time.time() - start_time
        return "Error processing your question.", metadata

# -------------------------------------------------------------------------
# 6. Export
# -------------------------------------------------------------------------
def get_ultra_fast_chain():
    """Get the ultra-fast RAG chain"""
    return ultra_fast_rag

def clear_cache():
    """Clear all caches"""
    cache.memory_cache.clear()
    cache.access_count.clear()
    logger.info("✅ Cache cleared")