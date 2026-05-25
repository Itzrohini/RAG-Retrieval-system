import re
import motor.motor_asyncio
from loguru import logger
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.config import settings


class MongoStore:
    """
    Stores document metadata and chunk text in MongoDB.
    Priority search order:
    1. Exact phrase match (for "Problem 6.10", "Example 5.9" etc.)
    2. Atlas Search BM25 (if on Atlas)
    3. MongoDB $text index fallback
    4. Regex fallback
    """

    def __init__(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(
            settings.mongodb_url,
        )
        self.db = self.client[settings.mongodb_db_name]
        self.docs = self.db["documents"]
        self.chunks = self.db["chunks"]

    async def ensure_indexes(self):
        """Call once at startup to create required indexes."""
        await self.docs.create_index("doc_id", unique=True)
        await self.chunks.create_index("chunk_id", unique=True)
        await self.chunks.create_index("doc_id")
        await self.chunks.create_index([("text", "text")])
        logger.info("MongoDB indexes ensured")

    async def save_document(self, doc_meta: Dict[str, Any]):
        doc_meta["created_at"] = datetime.utcnow()
        await self.docs.update_one(
            {"doc_id": doc_meta["doc_id"]},
            {"$set": doc_meta},
            upsert=True,
        )
        logger.info(f"Saved document metadata: {doc_meta['doc_id']}")

    async def save_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
        try:
            await self.chunks.insert_many(chunks, ordered=False)
        except Exception as e:
            logger.warning(f"Some chunks already existed (skipped): {e}")
        logger.info(f"Saved {len(chunks)} chunks to MongoDB")

    async def exact_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Exact phrase search — highest priority.
        Used for queries like 'Problem 6.10', 'Example 5.9', 'Table 6.9'.
        Returns chunks that contain the exact phrase (case-insensitive).
        """
        try:
            escaped = re.escape(query.strip())
            cursor = self.chunks.find(
                {"text": {"$regex": escaped, "$options": "i"}},
                {"_id": 0}
            ).limit(top_k)
            results = await cursor.to_list(length=top_k)
            if results:
                logger.info(f"Exact search found {len(results)} chunks for: '{query}'")
            return [
                {
                    "chunk_id": r.get("chunk_id", ""),
                    "text": r.get("text", ""),
                    "score": 1.0,  # exact match gets highest score
                    "metadata": {
                        "doc_id": r.get("doc_id"),
                        "file_name": r.get("file_name"),
                        "page_number": r.get("page_number"),
                        "sheet_name": r.get("sheet_name"),
                    },
                    "source": "exact",
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Exact search failed: {e}")
            return []

    async def keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """
        BM25 full-text search. Tries Atlas Search first, falls back to $text index.
        """
        try:
            pipeline = [
                {
                    "$search": {
                        "index": "default",
                        "text": {
                            "query": query,
                            "path": "text",
                        },
                    }
                },
                {"$limit": top_k},
                {
                    "$project": {
                        "chunk_id": 1, "text": 1, "doc_id": 1,
                        "file_name": 1, "page_number": 1, "sheet_name": 1,
                        "score": {"$meta": "searchScore"}, "_id": 0,
                    }
                },
            ]
            cursor = self.chunks.aggregate(pipeline)
            results = await cursor.to_list(length=top_k)
            if not results:
                return await self._text_index_search(query, top_k)
            return [
                {
                    "chunk_id": r["chunk_id"],
                    "text": r["text"],
                    "score": r.get("score", 0.5),
                    "metadata": {
                        "doc_id": r.get("doc_id"),
                        "file_name": r.get("file_name"),
                        "page_number": r.get("page_number"),
                        "sheet_name": r.get("sheet_name"),
                    },
                    "source": "bm25",
                }
                for r in results
            ]
        except Exception:
            return await self._text_index_search(query, top_k)

    async def _text_index_search(self, query: str, top_k: int) -> List[Dict]:
        """MongoDB $text index search fallback."""
        try:
            cursor = self.chunks.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}, "_id": 0},
            ).sort([("score", {"$meta": "textScore"})]).limit(top_k)
            results = await cursor.to_list(length=top_k)
            if results:
                return [
                    {
                        "chunk_id": r.get("chunk_id", ""),
                        "text": r.get("text", ""),
                        "score": r.get("score", 0.3),
                        "metadata": {
                            "doc_id": r.get("doc_id"),
                            "file_name": r.get("file_name"),
                            "page_number": r.get("page_number"),
                            "sheet_name": r.get("sheet_name"),
                        },
                        "source": "text_index",
                    }
                    for r in results
                ]
        except Exception:
            pass

        # Final fallback — regex
        try:
            cursor = self.chunks.find(
                {"text": {"$regex": query, "$options": "i"}},
                {"_id": 0}
            ).limit(top_k)
            results = await cursor.to_list(length=top_k)
            return [
                {
                    "chunk_id": r.get("chunk_id", ""),
                    "text": r.get("text", ""),
                    "score": 0.3,
                    "metadata": {
                        "doc_id": r.get("doc_id"),
                        "file_name": r.get("file_name"),
                        "page_number": r.get("page_number"),
                        "sheet_name": r.get("sheet_name"),
                    },
                    "source": "regex",
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"All keyword search methods failed: {e}")
            return []

    async def get_document(self, doc_id: str) -> Optional[Dict]:
        return await self.docs.find_one({"doc_id": doc_id}, {"_id": 0})

    async def list_documents(self) -> List[Dict]:
        cursor = self.docs.find({}, {"_id": 0})
        return await cursor.to_list(length=1000)

    async def delete_document(self, doc_id: str):
        await self.docs.delete_one({"doc_id": doc_id})
        result = await self.chunks.delete_many({"doc_id": doc_id})
        logger.info(f"Deleted document {doc_id} and {result.deleted_count} chunks")