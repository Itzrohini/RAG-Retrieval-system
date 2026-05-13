from fastapi import APIRouter
from app.services.cache.redis_cache import QueryCache
from app.services.llm.ollama_client import OllamaClient

router = APIRouter()


@router.get("/health")
async def health_check():
    cache = QueryCache()
    llm = OllamaClient()
    return {
        "status": "ok",
        "redis": cache.health_check(),
        "ollama": await llm.health_check(),
    }