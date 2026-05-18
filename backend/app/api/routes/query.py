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
from app.services.cache.redis_cache import QueryCache
import psutil
from app.services.llm.llm_factory import get_llm_client

router = APIRouter()


@router.post("/query/stream")
async def query_stream(request: QueryRequest):

    async def event_generator():
        try:
            start = time.time()
            session_id = request.session_id or str(uuid.uuid4())

            embedder = BGEEmbedder()
            qdrant = QdrantStore()
            mongo = MongoStore()
            memory = MemPalace()
            retriever = HybridRetriever(qdrant, mongo, embedder, memory)
            llm = get_llm_client()
            cache = QueryCache()

            # Check cache first
            if request.use_cache:
                cached = cache.get(request.question)
                if cached:
                    t1 = time.time()
                    logger.info(f"LATENCY | total={round((t1-start)*1000)}ms (CACHE HIT)")
                    yield f"data: {json.dumps({'type': 'cached', 'data': cached})}\n\n"
                    return

            # Retrieve candidates
            t0 = time.time()
            candidates = await retriever.retrieve(
                request.question, session_id, top_k=50
            )
            t1 = time.time()
            logger.info(f"LATENCY | retrieval={round((t1-t0)*1000)}ms")

            # Detect if this is a general question
            # General = no candidates OR all candidates below threshold
            is_general = len(candidates) == 0
            top_score = candidates[0].get("score", 0) if candidates else 0
            if top_score < 0.02:
                is_general = True

            # Check if MemPalace has prior context for this session
            session_history = memory.get_session_history(session_id)
            from_mempalace = len(session_history) > 0
            boosted_ids = memory.get_boosted_chunk_ids(session_id)
            mempalace_boost_used = any(
                c.get("chunk_id") in boosted_ids for c in candidates
            )

            # Rerank only if we have relevant chunks
            if not is_general and candidates:
                try:
                    from app.services.retrieval.reranker import CrossEncoderReranker
                    reranker = CrossEncoderReranker()
                    top_chunks = reranker.rerank(
                        request.question, candidates, top_k=request.top_k
                    )
                    t2 = time.time()
                    logger.info(f"LATENCY | rerank={round((t2-t1)*1000)}ms")
                except Exception as e:
                    logger.warning(f"Reranker skipped: {e}")
                    top_chunks = candidates[:request.top_k]
                    t2 = time.time()
            else:
                top_chunks = []
                t2 = time.time()

            # Build sources — only send if document-based question
            sources = []
            if not is_general and top_chunks:
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
                yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

            # Send MemPalace status to UI
            yield f"data: {json.dumps({'type': 'mempalace', 'data': {'from_memory': from_mempalace, 'boost_used': mempalace_boost_used, 'prior_turns': len(session_history)}})}\n\n"

            # Stream LLM answer
            memory_ctx = memory.build_context_summary(session_id)
            full_answer = []

            t3 = time.time()
            async for token in llm.stream_answer(
                request.question, top_chunks, memory_ctx
            ):
                full_answer.append(token)
                yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

            t4 = time.time()
            answer_text = "".join(full_answer)
            latency = round((t4 - start) * 1000, 1)

            logger.info(f"LATENCY | llm={round((t4-t3)*1000)}ms | total={latency}ms")

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
                "is_general": is_general,
            }
            cache.set(request.question, result_to_cache)

            yield f"data: {json.dumps({'type': 'done', 'data': {'latency_ms': latency, 'session_id': session_id, 'is_general': is_general, 'threshold': retriever.RELEVANCE_THRESHOLD, 'chunks_used': len(top_chunks)}})}\n\n"

        except Exception as e:
            logger.error(f"Query error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )