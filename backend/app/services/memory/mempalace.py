import json
import redis
import networkx as nx
from loguru import logger
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.config import settings


class MemPalace:
    """
    Structured episodic memory for iterative Q&A.

    What it does:
    1. Stores every (question, answer, chunk_ids) triple per session
    2. Builds a graph of entities mentioned across turns
    3. On new queries, retrieves prior context to enrich the search
    4. Boosts chunk IDs that were useful in prior turns

    Why this matters for iterative mode:
    Without this, question 3 has no idea what questions 1 and 2 were about.
    MemPalace gives the retrieval layer memory across turns.
    """

    def __init__(self):
        self.redis = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
        # In-memory graph of entities per session (rebuilt from Redis on load)
        self._session_graphs: Dict[str, nx.DiGraph] = {}

    def _session_key(self, session_id: str) -> str:
        return f"mempalace:session:{session_id}"

    def save_episode(
        self,
        session_id: str,
        question: str,
        answer: str,
        chunk_ids: List[str],
        entities: Optional[List[str]] = None,
    ):
        """Store one Q&A turn as an episode in Redis."""
        episode = {
            "question": question,
            "answer": answer,
            "chunk_ids": chunk_ids,
            "entities": entities or [],
            "timestamp": datetime.utcnow().isoformat(),
        }
        key = self._session_key(session_id)
        self.redis.rpush(key, json.dumps(episode))
        self.redis.expire(key, 86400)  # 24-hour session TTL
        logger.debug(f"Saved episode for session {session_id}")

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all episodes for a session."""
        key = self._session_key(session_id)
        raw = self.redis.lrange(key, 0, -1)
        return [json.loads(r) for r in raw]

    def get_boosted_chunk_ids(self, session_id: str) -> List[str]:
        """
        Return chunk IDs that appeared in previous answers.
        The hybrid retriever will boost these in the RRF merge step.
        """
        history = self.get_session_history(session_id)
        seen = {}
        for episode in history:
            for cid in episode.get("chunk_ids", []):
                seen[cid] = seen.get(cid, 0) + 1
        # Sort by frequency — most-reused chunks get the biggest boost
        sorted_ids = sorted(seen, key=seen.get, reverse=True)
        return sorted_ids[:10]

    def build_context_summary(self, session_id: str, last_n: int = 3) -> str:
        """
        Returns a text summary of the last N turns.
        This is injected into the LLM prompt as conversation history.
        """
        history = self.get_session_history(session_id)
        recent = history[-last_n:] if len(history) >= last_n else history
        if not recent:
            return ""
        lines = ["Prior conversation context:"]
        for ep in recent:
            lines.append(f"Q: {ep['question']}")
            lines.append(f"A: {ep['answer'][:300]}...")
        return "\n".join(lines)

    def enrich_query(self, session_id: str, query: str) -> str:
        """
        Expands a follow-up query with entities from prior turns.
        Example: "What about Q2?" → "What about Q2? [context: revenue, finance, Q3]"
        """
        history = self.get_session_history(session_id)
        if not history:
            return query
        recent_entities = []
        for ep in history[-2:]:
            recent_entities.extend(ep.get("entities", []))
        if recent_entities:
            entity_str = ", ".join(set(recent_entities[:5]))
            return f"{query} [context: {entity_str}]"
        return query

    def clear_session(self, session_id: str):
        self.redis.delete(self._session_key(session_id))
        self._session_graphs.pop(session_id, None)
        logger.info(f"Cleared session {session_id}")