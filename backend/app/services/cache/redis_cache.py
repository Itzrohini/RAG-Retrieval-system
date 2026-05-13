import redis
import json
import hashlib
from loguru import logger
from typing import Optional, Any
from app.config import settings


class QueryCache:
    """
    Caches query results in Redis.
    Key = SHA256 hash of the question text.
    TTL = configurable (default 1 hour).

    Why cache? Re-running the full pipeline for "What is Q3 revenue?"
    asked twice costs 500ms. Cache returns it in <5ms.
    """

    def __init__(self):
        self.client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
        self.ttl = settings.redis_ttl_seconds

    def _make_key(self, question: str) -> str:
        return "rag:query:" + hashlib.sha256(question.encode()).hexdigest()

    def get(self, question: str) -> Optional[Any]:
        key = self._make_key(question)
        val = self.client.get(key)
        if val:
            logger.info("Cache HIT")
            return json.loads(val)
        logger.debug("Cache MISS")
        return None

    def set(self, question: str, result: Any):
        key = self._make_key(question)
        self.client.setex(key, self.ttl, json.dumps(result))
        logger.debug(f"Cached result for: {question[:50]}")

    def invalidate(self, question: str):
        self.client.delete(self._make_key(question))

    def health_check(self) -> bool:
        try:
            return self.client.ping()
        except Exception:
            return False