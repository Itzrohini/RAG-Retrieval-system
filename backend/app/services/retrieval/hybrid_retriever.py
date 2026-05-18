from loguru import logger
from typing import List, Dict, Any

from app.config import settings
from app.services.vectorstore.qdrant_store import QdrantStore
from app.services.database.mongo_store import MongoStore
from app.services.embedding.embedder import BGEEmbedder
from app.services.memory.mempalace import MemPalace

import asyncio

RELEVANCE_THRESHOLD = 0.02  # chunks below this score are discarded

async def retrieve(
    self,
    query: str,
    session_id: str,
    top_k: int = 50,
) -> List[Dict[str, Any]]:
    enriched_query = self.memory.enrich_query(session_id, query)
    logger.info(f"Enriched query: {enriched_query[:100]}")

    embedding = self.embedder.embed_query(enriched_query)
    dense_vec = embedding["dense"]
    sparse_indices = embedding["sparse"]["indices"]
    sparse_values = embedding["sparse"]["values"]

    dense_results, sparse_results, bm25_results = await asyncio.gather(
        asyncio.to_thread(
            self.qdrant.search_dense, dense_vec, settings.top_k_dense
        ),
        asyncio.to_thread(
            self.qdrant.search_sparse, sparse_indices, sparse_values, settings.top_k_sparse
        ),
        self.mongo.keyword_search(query, settings.top_k_bm25),
    )

    logger.info(
        f"Retrieved: dense={len(dense_results)}, "
        f"sparse={len(sparse_results)}, bm25={len(bm25_results)}"
    )

    fused = self._rrf_merge([dense_results, sparse_results, bm25_results], top_k)

    # Apply relevance threshold — discard chunks below minimum score
    before_filter = len(fused)
    fused = [c for c in fused if c.get("score", 0) >= self.RELEVANCE_THRESHOLD]
    logger.info(f"Threshold filter: {before_filter} → {len(fused)} chunks (threshold={self.RELEVANCE_THRESHOLD})")

    boosted_ids = self.memory.get_boosted_chunk_ids(session_id)
    fused = self._apply_memory_boost(fused, boosted_ids)

    return fused[:top_k]


class HybridRetriever:
    """
    Fuses three retrieval signals using Reciprocal Rank Fusion (RRF):

    1. Dense semantic search (Qdrant HNSW)
    2. Sparse keyword search (Qdrant BM42)
    3. BM25 full-text search (MongoDB Atlas Search)

    Plus MemPalace episode boost for repeated query patterns.

    RRF formula:
        score(d) = Σ 1 / (k + rank(d))

    k=60 is standard — dampens the impact of very high ranks.
    """

    RRF_K = 60
    RELEVANCE_THRESHOLD = 0.02

    def __init__(
        self,
        qdrant: QdrantStore,
        mongo: MongoStore,
        embedder: BGEEmbedder,
        memory: MemPalace,
    ):
        self.qdrant = qdrant
        self.mongo = mongo
        self.embedder = embedder
        self.memory = memory

    async def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Full hybrid retrieval pipeline.
        Returns up to top_k candidates for the re-ranker.
        """

        # Step 1 — Enrich query with MemPalace context
        enriched_query = self.memory.enrich_query(
            session_id,
            query
        )

        logger.info(
            f"Enriched query: {enriched_query[:100]}"
        )

        # Step 2 — Embed query
        embedding = self.embedder.embed_query(
            enriched_query
        )

        dense_vec = embedding["dense"]

        sparse_indices = embedding["sparse"]["indices"]
        sparse_values = embedding["sparse"]["values"]

        # Step 3 — Run retrievals in parallel

        dense_task = asyncio.to_thread(
            self.qdrant.search_dense,
            dense_vec,
            settings.top_k_dense,
        )

        sparse_task = (
            asyncio.to_thread(
                self.qdrant.search_sparse,
                sparse_indices,
                sparse_values,
                settings.top_k_sparse,
            )
            if sparse_indices
            else asyncio.to_thread(lambda: [])
        )

        bm25_task = self.mongo.keyword_search(
            query,
            settings.top_k_bm25,
        )

        dense_results, sparse_results, bm25_results = await asyncio.gather(
            dense_task,
            sparse_task,
            bm25_task,
        )

        logger.info(
            f"Retrieved: "
            f"dense={len(dense_results)}, "
            f"sparse={len(sparse_results)}, "
            f"bm25={len(bm25_results)}"
        )

        # Step 4 — RRF merge
        fused = self._rrf_merge(
            [
                dense_results,
                sparse_results,
                bm25_results,
            ],
            top_k,
        )

        # Step 5 — Memory boost
        boosted_ids = self.memory.get_boosted_chunk_ids(
            session_id
        )

        fused = self._apply_memory_boost(
            fused,
            boosted_ids,
        )

        return fused[:top_k]

    def _rrf_merge(
        self,
        result_lists: List[List[Dict]],
        top_k: int,
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF)
        with deduplication by text fingerprint.
        """

        scores: Dict[str, float] = {}
        payloads: Dict[str, Dict] = {}
        text_seen: set = set()

        for results in result_lists:

            for rank, item in enumerate(results):

                # Handle Qdrant ScoredPoint objects
                if hasattr(item, "payload"):
                    item = item.payload

                chunk_id = item.get("chunk_id")

                if not chunk_id:
                    continue

                text = item.get("text", "")

                # Deduplicate by text fingerprint
                text_fp = text[:100].strip().lower()

                if text_fp in text_seen and chunk_id not in scores:
                    continue

                text_seen.add(text_fp)

                # RRF scoring
                rrf_score = 1.0 / (
                    self.RRF_K + rank + 1
                )

                scores[chunk_id] = (
                    scores.get(chunk_id, 0.0)
                    + rrf_score
                )

                payloads[chunk_id] = item

        # Sort by descending RRF score
        sorted_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        merged = []

        for chunk_id in sorted_ids[:top_k]:

            item = payloads[chunk_id].copy()

            item["rrf_score"] = scores[chunk_id]
            item["score"] = scores[chunk_id]

            merged.append(item)

        return merged

    def _apply_memory_boost(
        self,
        results: List[Dict],
        boosted_ids: List[str],
    ) -> List[Dict]:
        """
        Give small score boosts to chunks
        previously used in the conversation.
        """

        if not boosted_ids:
            return results

        boost_map = {
            cid: 0.1 / (i + 1)
            for i, cid in enumerate(boosted_ids)
        }

        for item in results:

            chunk_id = item.get("chunk_id")

            if chunk_id in boost_map:

                item["score"] += boost_map[chunk_id]
                item["memory_boosted"] = True

        # Re-sort after boosting
        return sorted(
            results,
            key=lambda x: x["score"],
            reverse=True,
        )