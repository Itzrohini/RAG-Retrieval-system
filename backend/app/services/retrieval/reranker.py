from sentence_transformers import CrossEncoder
from loguru import logger
from typing import List, Dict, Any
from app.config import settings
import gc
import psutil


class CrossEncoderReranker:
    """
    Re-scores the top-50 hybrid results using a cross-encoder.

    Difference from bi-encoder (embedding):
    - Bi-encoder embeds query and doc separately, then compares vectors.
      Fast but loses query-document interaction.
    - Cross-encoder sees (query + document) together, scoring relevance jointly.
      Slower but far more accurate — knows exactly what the query needs from the doc.

    We run this only on top-50 candidates (not all 10,000 docs) to keep it fast.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # Lazy loading - model will be loaded on first use
        self.model = None
        self._initialized = True

    def _load_model(self):
        """Lazy load the model only when needed"""
        if self.model is None:
            # Check available memory before loading
            available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
            if available_memory < 3.0:  # Less than 3GB available
                logger.warning(f"Low memory detected: {available_memory:.1f}GB available. Reranker loading may fail.")
                if available_memory < 1.5:  # Critical memory
                    raise MemoryError(f"Insufficient memory to load reranker model. Available: {available_memory:.1f}GB")

            logger.info(f"Loading re-ranker: {settings.reranker_model}")
            try:
                self.model = CrossEncoder(settings.reranker_model, max_length=512)
                logger.info("Re-ranker loaded")
            except Exception as e:
                logger.error(f"Failed to load re-ranker: {e}")
                raise

    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            logger.info("Unloading re-ranker model to free memory")
            del self.model
            self.model = None
            gc.collect()

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Score each (query, candidate_text) pair and return top_k.
        """
        if not candidates:
            return []

        # Lazy load model on first use
        self._load_model()

        top_k = top_k or settings.top_k_rerank
        pairs = [(query, c["text"]) for c in candidates]

        logger.info(f"Re-ranking {len(pairs)} candidates")
        scores = self.model.predict(pairs, show_progress_bar=False)

        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        top = reranked[:top_k]

        logger.info(
            f"Re-rank scores: top={top[0]['rerank_score']:.3f}, "
            f"bottom={top[-1]['rerank_score']:.3f}"
        )
        return top