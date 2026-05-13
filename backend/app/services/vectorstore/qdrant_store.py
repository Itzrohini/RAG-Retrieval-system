from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
    NamedVector,
    NamedSparseVector,
    SparseIndexParams,
)
from loguru import logger
from typing import List, Dict, Any
from app.config import settings


class QdrantStore:
    """
    Manages the Qdrant vector collection.
    Stores both dense (HNSW) and sparse (BM42) vectors per chunk.
    Enables semantic + lexical search from a single store.
    """

    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        self.collection = settings.qdrant_collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""

        collections = [
            c.name for c in self.client.get_collections().collections
        ]

        if self.collection in collections:
            logger.info(
                f"Qdrant collection '{self.collection}' already exists"
            )
            return

        logger.info(
            f"Creating Qdrant collection '{self.collection}'"
        )

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": VectorParams(
                    size=1024,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )

        logger.info(
            "Qdrant collection created with dense + sparse vectors"
        )

    def upsert_chunks(
        self,
        chunks_with_embeddings: List[Dict[str, Any]]
    ):
        """
        Batch upsert with smaller batches and retry for reliability at scale.
        """

        points = []

        for item in chunks_with_embeddings:

            point = PointStruct(
                id=item["chunk_id"],
                vector={
                    "dense": item["dense"],
                    "sparse": SparseVector(
                        indices=item["sparse"]["indices"],
                        values=item["sparse"]["values"],
                    ),
                },
                payload={
                    "text": item["text"],
                    "doc_id": item["metadata"]["doc_id"],
                    "file_name": item["metadata"]["file_name"],
                    "doc_type": item["metadata"]["doc_type"],
                    "page_number": item["metadata"].get("page_number"),
                    "sheet_name": item["metadata"].get("sheet_name"),
                    "chunk_index": item["metadata"]["chunk_index"],
                },
            )

            points.append(point)

        batch_size = 50

        total_batches = (
            len(points) + batch_size - 1
        ) // batch_size

        for i in range(0, len(points), batch_size):

            batch = points[i:i + batch_size]
            batch_num = i // batch_size + 1

            for attempt in range(3):

                try:
                    self.client.upsert(
                        collection_name=self.collection,
                        points=batch,
                        wait=True,
                    )

                    logger.debug(
                        f"Upserted batch {batch_num}/{total_batches}"
                    )

                    break

                except Exception as e:

                    if attempt == 2:
                        logger.error(
                            f"Batch {batch_num} failed after 3 attempts: {e}"
                        )
                        raise

                    logger.warning(
                        f"Batch {batch_num} attempt {attempt + 1} failed, retrying..."
                    )

                    import time
                    time.sleep(1)

        logger.info(
            f"Upserted {len(points)} chunks to Qdrant in {total_batches} batches"
        )

    def search_dense(
        self,
        dense_vector: List[float],
        top_k: int
    ) -> List[Dict]:

        results = self.client.search(
            collection_name=self.collection,
            query_vector=NamedVector(
                name="dense",
                vector=dense_vector,
            ),
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "chunk_id": str(r.id),
                "text": r.payload["text"],
                "score": r.score,
                "metadata": {
                    k: v
                    for k, v in r.payload.items()
                    if k != "text"
                },
                "source": "dense",
            }
            for r in results
        ]

    def search_sparse(
        self,
        sparse_indices: List[int],
        sparse_values: List[float],
        top_k: int,
    ) -> List[Dict]:

        results = self.client.search(
            collection_name=self.collection,
            query_vector=NamedSparseVector(
                name="sparse",
                vector=SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
            ),
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "chunk_id": str(r.id),
                "text": r.payload["text"],
                "score": r.score,
                "metadata": {
                    k: v
                    for k, v in r.payload.items()
                    if k != "text"
                },
                "source": "sparse",
            }
            for r in results
        ]

    def delete_by_doc_id(self, doc_id: str):

        from qdrant_client.models import (
            Filter,
            FieldCondition,
            MatchValue,
        )

        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            ),
        )

        logger.info(
            f"Deleted all chunks for doc_id: {doc_id}"
        )