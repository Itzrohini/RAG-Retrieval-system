import re
import asyncio
from loguru import logger
from typing import List, Dict, Any
from app.config import settings
from app.services.vectorstore.qdrant_store import QdrantStore
from app.services.database.mongo_store import MongoStore
from app.services.embedding.embedder import BGEEmbedder
from app.services.memory.mempalace import MemPalace


# Patterns that indicate exact identifier queries
EXACT_PATTERNS = [
    r'problem\s+\d+[\.\-]\d+',
    r'example\s+\d+[\.\-]\d+',
    r'equation\s+\d+[\.\-]\d+',
    r'figure\s+\d+[\.\-]\d+',
    r'table\s+\d+[\.\-]\d+',
    r'exercise\s+\d+[\.\-]\d+',
    r'theorem\s+\d+[\.\-]\d+',
    r'section\s+\d+[\.\-]\d+',
    r'chapter\s+\d+[\.\-]\d+',
    r'q\.\s*\d+',
    r'^\d+[\.\-]\d+$',
]


def _extract_exact_term(query: str):
    q_lower = query.lower().strip()
    for pattern in EXACT_PATTERNS:
        match = re.search(pattern, q_lower)
        if match:
            return match.group(0)
    return None


class HybridRetriever:
    RRF_K = 60
    RELEVANCE_THRESHOLD = 0.01

    def __init__(self, qdrant: QdrantStore, mongo: MongoStore,
                 embedder: BGEEmbedder, memory: MemPalace):
        self.qdrant = qdrant
        self.mongo = mongo
        self.embedder = embedder
        self.memory = memory

    async def retrieve(self, query: str, session_id: str,
                       top_k: int = 50) -> List[Dict[str, Any]]:

        enriched_query = self.memory.enrich_query(session_id, query)
        logger.info(f"Enriched query: {enriched_query[:100]}")

        exact_term = _extract_exact_term(query)
        if exact_term:
            logger.info(f"Exact identifier detected: '{exact_term}'")

        embedding = self.embedder.embed_query(enriched_query)
        dense_vec = embedding["dense"]
        sparse_vec = embedding.get("sparse")

        async def empty():
            return []

        sparse_task = (
            asyncio.get_event_loop().run_in_executor(
                None, self.qdrant.search_sparse,
                sparse_vec["indices"], sparse_vec["values"], settings.top_k_sparse,
            )
            if sparse_vec and sparse_vec.get("indices")
            else empty()
        )

        exact_task = (
            self.mongo.exact_search(exact_term, 10)
            if exact_term else empty()
        )

        vector_results, sparse_results, bm25_results, exact_results = \
            await asyncio.gather(
                asyncio.get_event_loop().run_in_executor(
                    None, self.qdrant.search_dense, dense_vec, settings.top_k_dense,
                ),
                sparse_task,
                self.mongo.keyword_search(enriched_query, settings.top_k_bm25),
                exact_task,
            )

        logger.info(
            f"Retrieved: dense={len(vector_results)}, sparse={len(sparse_results)}, "
            f"bm25={len(bm25_results)}, exact={len(exact_results)}"
        )

        if exact_results:
            for r in exact_results:
                r["score"] = 1.0
                r["source"] = "exact"
            semantic_fused = self._rrf_merge(
                [vector_results, sparse_results, bm25_results], top_k
            )
            exact_ids = {r["chunk_id"] for r in exact_results}
            semantic_only = [r for r in semantic_fused if r["chunk_id"] not in exact_ids]
            fused = exact_results + semantic_only
        else:
            fused = self._rrf_merge(
                [vector_results, sparse_results, bm25_results], top_k
            )

        before = len(fused)
        if exact_results:
            exact_ids = {r["chunk_id"] for r in exact_results}
            fused = exact_results + [
                c for c in fused
                if c["chunk_id"] not in exact_ids
                and c.get("score", 0) >= self.RELEVANCE_THRESHOLD
            ]
        else:
            fused = [c for c in fused if c.get("score", 0) >= self.RELEVANCE_THRESHOLD]

        logger.info(f"Threshold filter: {before} -> {len(fused)} chunks")

        boosted_ids = self.memory.get_boosted_chunk_ids(session_id)
        fused = self._apply_memory_boost(fused, boosted_ids)

        return fused[:top_k]

    def _rrf_merge(self, result_lists, top_k):
        scores = {}
        payloads = {}
        text_seen = set()

        for results in result_lists:
            for rank, item in enumerate(results):
                cid = item["chunk_id"]
                text_fp = item["text"][:100].strip().lower()
                if text_fp in text_seen and cid not in scores:
                    continue
                text_seen.add(text_fp)
                rrf_score = 1.0 / (self.RRF_K + rank + 1)
                scores[cid] = scores.get(cid, 0.0) + rrf_score
                payloads[cid] = item

        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        merged = []
        for cid in sorted_ids[:top_k]:
            item = payloads[cid].copy()
            item["score"] = scores[cid]
            merged.append(item)
        return merged

    def _apply_memory_boost(self, results, boosted_ids):
        if not boosted_ids:
            return results
        boost_map = {cid: 0.1 / (i + 1) for i, cid in enumerate(boosted_ids)}
        for item in results:
            if item["chunk_id"] in boost_map:
                item["score"] += boost_map[item["chunk_id"]]
                item["memory_boosted"] = True
        return sorted(results, key=lambda x: x["score"], reverse=True)