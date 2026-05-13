from loguru import logger
from typing import List, Dict, Any
from app.config import settings
from app.services.vectorstore.qdrant_store import QdrantStore
from app.services.database.mongo_store import MongoStore
from app.services.embedding.embedder import BGEEmbedder
from app.services.memory.mempalace import MemPalace
import asyncio


class HybridRetriever:
    """
    Fuses three retrieval signals using Reciprocal Rank Fusion (RRF):
    1. Dense semantic search (Qdrant HNSW)
    2. Sparse keyword search (Qdrant BM42)
    3. BM25 full-text search (MongoDB Atlas Search)

    Plus MemPalace episode boost for repeated query patterns.

    RRF formula: score(d) = Σ 1 / (k + rank(d))
    k=60 is standard — dampens the impact of very high ranks.
    This lets us merge ranked lists without normalizing scores.
    """

    RRF_K = 60

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
        # Step 1: Enrich query with MemPalace context
        enriched_query = self.memory.enrich_query(session_id, query)
        logger.info(f"Enriched query: {enriched_query[:100]}")

        # Step 2: Embed query (dense + sparse in one call)
        embedding = self.embedder.embed_query(enriched_query)
        dense_vec = embedding["dense"]
        sparse_indices = embedding["sparse"]["indices"]
        sparse_values = embedding["sparse"]["values"]

        # Step 3: Fire all three searches in parallel
        sparse_task = (
            asyncio.to_thread(self.qdrant.search_sparse, sparse_indices, sparse_values, settings.top_k_sparse)
            if sparse_indices
            else asyncio.to_thread(lambda: [])
        )
        
        dense_results, sparse_results, bm25_results = await asyncio.gather(
            asyncio.to_thread(
                self.qdrant.search_dense, dense_vec, settings.top_k_dense
            ),
            sparse_task,
            self.mongo.keyword_search(query, settings.top_k_bm25),
        )

        logger.info(
            f"Retrieved: dense={len(dense_results)}, "
            f"sparse={len(sparse_results)}, bm25={len(bm25_results)}"
        )

        # Step 4: RRF merge
        fused = self._rrf_merge([dense_results, sparse_results, bm25_results], top_k)

        # Step 5: MemPalace boost — bump chunks seen in prior answers
        boosted_ids = self.memory.get_boosted_chunk_ids(session_id)
        fused = self._apply_memory_boost(fused, boosted_ids)

        return fused[:top_k]

    def _rrf_merge(
        self,
        result_lists: List[List[Dict]],
        top_k: int,
    ) -> List[Dict]:
        """Apply Reciprocal Rank Fusion across multiple ranked lists."""
        scores: Dict[str, float] = {}
        payloads: Dict[str, Dict] = {}

        for results in result_lists:
            for rank, item in enumerate(results):
                cid = item["chunk_id"]
                rrf_score = 1.0 / (self.RRF_K + rank + 1)
                scores[cid] = scores.get(cid, 0.0) + rrf_score
                payloads[cid] = item  # keep latest payload

        # Sort by fused RRF score
        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        merged = []
        for cid in sorted_ids[:top_k]:
            item = payloads[cid].copy()
            item["rrf_score"] = scores[cid]
            item["score"] = scores[cid]
            merged.append(item)

        return merged

    def _apply_memory_boost(
        self,
        results: List[Dict],
        boosted_ids: List[str],
    ) -> List[Dict]:
        """
        Give a small score boost to chunks seen in prior session turns.
        Boost decays by position in the boosted list (most frequent = biggest boost).
        """
        if not boosted_ids:
            return results
        boost_map = {cid: 0.1 / (i + 1) for i, cid in enumerate(boosted_ids)}
        for item in results:
            if item["chunk_id"] in boost_map:
                item["score"] += boost_map[item["chunk_id"]]
                item["memory_boosted"] = True
        # Re-sort after boost
        return sorted(results, key=lambda x: x["score"], reverse=True)