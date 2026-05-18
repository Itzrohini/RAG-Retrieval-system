from fastapi import APIRouter
from app.services.cache.redis_cache import QueryCache
from app.services.llm.llm_factory import get_llm_client
from app.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    cache = QueryCache()
    llm = get_llm_client()
    return {
        "status": "ok",
        "redis": cache.health_check(),
        "llm_provider": settings.llm_provider,
        "llm_model": settings.gemini_model if settings.llm_provider == "gemini" else settings.ollama_model,
        "llm_healthy": await llm.health_check(),
    }