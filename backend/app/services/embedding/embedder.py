from FlagEmbedding import BGEM3FlagModel
from loguru import logger
from typing import List, Dict, Any
from app.config import settings
import torch


class BGEEmbedder:
    """
    Always uses BAAI/bge-m3 — 1024-dim dense + sparse vectors.
    Singleton pattern — model loads once and reuses across all requests.
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
        self._load_model()
        self._initialized = True

    def _load_model(self):
        logger.info(f"Loading BGE-M3 model: BAAI/bge-m3")
        logger.info("First run takes 2-3 minutes to download ~2GB model")
        self.model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=False,  # CPU safe
        )
        self.dense_dim = 1024
        logger.info("BGE-M3 loaded — dense_dim=1024")

    def embed_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []
        logger.debug(f"Embedding {len(texts)} texts")
        outputs = self.model.encode(
            texts,
            batch_size=4,
            max_length=512,
            return_dense=True,
            return_sparse=True,
        )
        results = []
        for i in range(len(texts)):
            dense = outputs["dense_vecs"][i].tolist()
            sparse_dict = outputs["lexical_weights"][i]
            results.append({
                "dense": dense,
                "sparse": {
                    "indices": [int(k) for k in sparse_dict.keys()],
                    "values": [float(v) for v in sparse_dict.values()],
                },
            })
        return results

    def embed_query(self, query: str) -> Dict[str, Any]:
        return self.embed_texts([query])[0]
    
    def unload(self):
        """Free model from RAM so reranker can load."""
        import gc
        if hasattr(self, 'model') and self.model is not None:
            logger.info("Unloading BGE-M3 to free RAM for reranker")
            del self.model
            self.model = None
            BGEEmbedder._instance = None  # reset singleton so next call reloads
            BGEEmbedder._initialized = False
            self._initialized = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("BGE-M3 unloaded")