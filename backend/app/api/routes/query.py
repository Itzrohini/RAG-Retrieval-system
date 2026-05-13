import time
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
from loguru import logger
from app.config import settings
from app.models.document import QueryRequest, QueryResponse, SourceChunk
from app.services.embedding.embedder import BGEEmbedder
from app.services.vectorstore.qdrant_store import QdrantStore
from app.services.database.mongo_store import MongoStore
from app.services.memory.mempalace import MemPalace
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.reranker import CrossEncoderReranker
from app.services.llm.ollama_client import OllamaClient
from app.services.cache.redis_cache import QueryCache
import psutil

router = APIRouter()


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Streaming endpoint — returns answer tokens as Server-Sent Events.
    Frontend reads this with EventSource or fetch + ReadableStream.
    """
    async def event_generator():
        try:
            start = time.time()
            session_id = request.session_id or str(uuid.uuid4())

            # Services
            embedder = BGEEmbedder()
            qdrant = QdrantStore()
            mongo = MongoStore()
            memory = MemPalace()
            retriever = HybridRetriever(qdrant, mongo, embedder, memory)
            reranker = CrossEncoderReranker()
            llm = OllamaClient()
            cache = QueryCache()

            # Cache check
            if request.use_cache:
                cached = cache.get(request.question)
                if cached:
                    yield f"data: {json.dumps({'type': 'cached', 'data': cached})}\n\n"
                    return

            # Hybrid retrieval
            candidates = await retriever.retrieve(
                request.question, session_id, top_k=50
            )

            # Re-rank (optional for memory optimization)
            if settings.enable_reranking:
                try:
                    # Check if we have enough memory for reranking
                    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
                    if available_memory < 3.0:
                        logger.warning("Not enough free RAM for reranking; using unranked candidate order.")
                        top_chunks = candidates[:request.top_k] if request.top_k else candidates[:settings.top_k_rerank]
                    else:
                        top_chunks = reranker.rerank(request.question, candidates, top_k=request.top_k)
                except MemoryError as e:
                    logger.warning(f"Reranking failed due to memory: {e}. Using candidates without reranking.")
                    top_chunks = candidates[:request.top_k] if request.top_k else candidates[:settings.top_k_rerank]
            else:
                # Skip reranking - use original candidates
                logger.info("Reranking disabled for memory optimization")
                top_chunks = candidates[:request.top_k] if request.top_k else candidates[:settings.top_k_rerank]

            # Build sources for UI
            sources = [
                {
                    "chunk_id": c["chunk_id"],
                    "text": c["text"][:300],
                    "score": round(c.get("rerank_score", c.get("score", 0)), 4),
                    "file_name": c["metadata"].get("file_name", "unknown"),
                    "page_number": c["metadata"].get("page_number"),
                    "sheet_name": c["metadata"].get("sheet_name"),
                }
                for c in top_chunks
            ]

            # Send sources first so UI can show them while answer streams
            yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

            # MemPalace context
            memory_ctx = memory.build_context_summary(session_id)

            # Stream LLM answer
            full_answer = []
            async for token in llm.stream_answer(request.question, top_chunks, memory_ctx):
                full_answer.append(token)
                yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

            answer_text = "".join(full_answer)
            latency = round((time.time() - start) * 1000, 1)

            # Save to MemPalace
            memory.save_episode(
                session_id,
                request.question,
                answer_text,
                [c["chunk_id"] for c in top_chunks],
            )

            # Cache result
            result_to_cache = {
                "answer": answer_text,
                "sources": sources,
                "latency_ms": latency,
            }
            cache.set(request.question, result_to_cache)

            yield f"data: {json.dumps({'type': 'done', 'data': {'latency_ms': latency, 'session_id': session_id}})}\n\n"

        except Exception as e:
            logger.error(f"Query error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )