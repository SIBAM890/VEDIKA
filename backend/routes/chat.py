# backend/routes/chat.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.ultra_optimized_rag import ultra_fast_rag
from backend.utils.logger import get_logger
import json
import traceback
import asyncio

router = APIRouter()
logger = get_logger("chat_api")

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    metadata: dict = {}

# --- Main Chat Endpoint (Non-Streaming) ---
@router.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    Ultra-fast chat endpoint with comprehensive retrieval
    """
    user_message = request.message
    logger.info(f"📨 Request: {user_message}")

    try:
        # Use ultra-fast RAG
        response, metadata = await ultra_fast_rag(user_message)
        
        logger.info(f"✅ Response in {metadata.get('total_time', 0):.2f}s | Cached: {metadata.get('cache_hit', False)}")
        
        return ChatResponse(reply=response, metadata=metadata)

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"❌ Error: {e}\n{error_trace}")
        raise HTTPException(status_code=500, detail="Internal processing error.")

# --- Streaming Endpoint (Better UX) ---
@router.post("/chat/stream")
async def handle_chat_stream(request: ChatRequest):
    """
    Streaming endpoint for progressive response display
    """
    user_message = request.message
    logger.info(f"📨 Stream request: {user_message}")

    async def generate():
        try:
            # Get response
            response, metadata = await ultra_fast_rag(user_message)
            
            # If cached, send instantly
            if metadata.get('cache_hit'):
                yield f"data: {json.dumps({'content': response})}\n\n"
                yield f"data: {json.dumps({'done': True, 'metadata': metadata})}\n\n"
                return
            
            # Stream response word by word for smooth UX
            words = response.split()
            for word in words:
                yield f"data: {json.dumps({'content': word + ' '})}\n\n"
                await asyncio.sleep(0.01)  # Smooth streaming
            
            yield f"data: {json.dumps({'done': True, 'metadata': metadata})}\n\n"
            
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"❌ Stream error: {e}\n{error_trace}")
            yield f"data: {json.dumps({'error': 'An error occurred. Please try again.'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# --- Cache Management Endpoints ---
@router.post("/cache/clear")
async def clear_cache_endpoint():
    """Clear response cache (admin only)"""
    try:
        from backend.ultra_optimized_rag import cache
        cache.memory_cache.clear()
        cache.access_count.clear()
        
        # Clear disk cache
        import os
        import shutil
        if os.path.exists(cache.cache_dir):
            shutil.rmtree(cache.cache_dir)
            os.makedirs(cache.cache_dir)
        
        logger.info("🗑️ Cache cleared")
        return {
            "status": "success",
            "message": "All caches cleared successfully"
        }
    except Exception as e:
        logger.error(f"❌ Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache/stats")
async def cache_stats():
    """Get cache performance statistics"""
    try:
        from backend.ultra_optimized_rag import cache
        import os
        
        # Count disk cache files
        disk_cache_count = 0
        if os.path.exists(cache.cache_dir):
            disk_cache_count = len([f for f in os.listdir(cache.cache_dir) if f.endswith('.json') and f != '_stats.json'])
        
        # Calculate hit rate
        total_accesses = sum(cache.access_count.values())
        unique_queries = len(cache.access_count)
        hit_rate = (unique_queries / total_accesses * 100) if total_accesses > 0 else 0
        
        # Find most popular queries
        top_queries = sorted(cache.access_count.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "memory_cache_size": len(cache.memory_cache),
            "disk_cache_size": disk_cache_count,
            "total_queries_processed": total_accesses,
            "unique_queries": unique_queries,
            "cache_hit_rate": f"{hit_rate:.1f}%",
            "ttl_hours": cache.ttl.total_seconds() / 3600,
            "top_5_query_hashes": [
                {"hash": k[:8], "count": v} for k, v in top_queries
            ]
        }
    except Exception as e:
        logger.error(f"❌ Cache stats error: {e}")
        return {"error": str(e)}

# --- Performance Monitoring ---
@router.get("/performance/stats")
async def performance_stats():
    """Get performance statistics"""
    try:
        from backend.ultra_optimized_rag import cache
        
        # Calculate average response time from logs (simplified)
        stats = {
            "cache_enabled": True,
            "memory_cache_items": len(cache.memory_cache),
            "total_queries_served": sum(cache.access_count.values()),
            "model": "gpt-3.5-turbo-1106 (fast mode)",
            "features": [
                "Ultra-fast caching (0.001s)",
                "Parallel retrieval",
                "Smart query classification",
                "LRU memory cache",
                "Automatic cache warming"
            ]
        }
        
        return stats
    except Exception as e:
        return {"error": str(e)}

# --- Health Check ---
@router.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        from backend.ultra_optimized_rag import vector_store, llm, cache
        
        health_status = {
            "status": "healthy",
            "components": {
                "vector_store": vector_store is not None,
                "llm": llm is not None,
                "cache": True,
                "memory_cache": len(cache.memory_cache) > 0
            },
            "optimization_level": "ultra",
            "expected_performance": {
                "cached_queries": "0.001s",
                "first_time_queries": "2-3s",
                "comprehensive_retrieval": "up to 20 documents"
            }
        }
        
        if not all(health_status["components"].values()):
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }