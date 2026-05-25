import time
import uuid
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import json
from loguru import logger

from app.models.document import QueryRequest
from app.services.embedding.embedder import BGEEmbedder
from app.services.memory.mempalace import MemPalace, extract_entities
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.cache.redis_cache import QueryCache
from app.services.llm.llm_factory import get_llm_client
from app.services.vectorstore.qdrant_store import QdrantStore
from app.services.database.mongo_store import MongoStore
from app.services.retrieval.reranker import CrossEncoderReranker

router = APIRouter()

# ── Minimum rerank score to consider a chunk genuinely relevant ──────────────
# CrossEncoder scores are raw logits (NOT 0-1). Typical ranges:
#   > 1.0  → strongly relevant
#   0 to 1 → possibly relevant
#  -1 to 0 → weak match
#  < -1.0  → clearly irrelevant
RERANK_RELEVANCE_THRESHOLD = -1.0


@router.post("/query/stream")
async def query_stream(request: QueryRequest):

    async def event_generator():
        try:
            start = time.time()
            session_id = request.session_id or str(uuid.uuid4())

            embedder = BGEEmbedder()
            memory = MemPalace()
            llm = get_llm_client()
            cache = QueryCache()
            qdrant = QdrantStore()
            mongo = MongoStore()

            retriever = HybridRetriever(qdrant, mongo, embedder, memory)

            # ── CACHE CHECK ──────────────────────────────────────────────────
            if request.use_cache:
                cached = cache.get(request.question)
                if cached:
                    t1 = time.time()
                    logger.info(f"LATENCY | total={round((t1 - start) * 1000)}ms (CACHE HIT)")

                    # Even on cache hit, save episode so memory graph grows
                    memory.save_episode(
                        session_id=session_id,
                        question=request.question,
                        answer=cached["answer"],
                        chunk_ids=[c["chunk_id"] for c in cached.get("sources", [])],
                        # entities auto-extracted inside save_episode
                    )

                    actual_latency = round((time.time() - start) * 1000, 1)
                    cached_with_real_latency = {
                        **cached,
                        "latency_ms": actual_latency,
                        "cache_hit": True,
                        "is_general": cached.get("is_general", False),
                    }
                    yield f"data: {json.dumps({'type': 'cached', 'data': cached_with_real_latency})}\n\n"
                    return

            # ── RETRIEVAL ────────────────────────────────────────────────────
            t0 = time.time()
            candidates = await retriever.retrieve(
                request.question, session_id, top_k=50
            )
            t1 = time.time()
            logger.info(f"LATENCY | retrieval={round((t1 - t0) * 1000)}ms")

            # ── RERANKING ────────────────────────────────────────────────────
            top_chunks = []
            t2 = time.time()

            if candidates:
                try:
                    # Free embedder RAM before loading reranker
                    embedder.unload()

                    reranker = CrossEncoderReranker()
                    top_chunks = reranker.rerank(
                        request.question,
                        candidates,
                        top_k=request.top_k,
                    )
                    t2 = time.time()
                    logger.info(f"LATENCY | rerank={round((t2 - t1) * 1000)}ms")

                    # Unload reranker — embedder reloads fresh on next request
                    reranker.unload_model()

                except Exception as e:
                    logger.warning(f"Reranker skipped: {e}")
                    top_chunks = candidates[:request.top_k]
                    t2 = time.time()

            # ── RELEVANCE GATE ───────────────────────────────────────────────
            if top_chunks:
                best_score = top_chunks[0].get("rerank_score", 0)
                worst_score = top_chunks[-1].get("rerank_score", 0)
                logger.info(f"Rerank scores: best={best_score:.3f}, worst={worst_score:.3f}")

            relevant_chunks = [
                c for c in top_chunks
                if c.get("rerank_score", c.get("score", 0)) >= RERANK_RELEVANCE_THRESHOLD
            ]

            is_general = len(relevant_chunks) == 0

            logger.info(
                f"Relevance gate: {len(top_chunks)} reranked → "
                f"{len(relevant_chunks)} relevant | is_general={is_general}"
            )

            final_chunks = relevant_chunks if not is_general else []

            # ── SESSION MEMORY ───────────────────────────────────────────────
            session_history = memory.get_session_history(session_id)
            from_mempalace = len(session_history) > 0
            boosted_ids = memory.get_boosted_chunk_ids(session_id)
            mempalace_boost_used = any(
                c.get("chunk_id") in boosted_ids for c in final_chunks
            )

            # ── SOURCES ─────────────────────────────────────────────────────
            sources = []
            if not is_general and final_chunks:
                sources = [
                    {
                        "chunk_id": c["chunk_id"],
                        "text": c["text"][:300],
                        "score": round(
                            c.get("rerank_score", c.get("score", 0)), 4
                        ),
                        "file_name": c["metadata"].get("file_name", "unknown"),
                        "page_number": c["metadata"].get("page_number"),
                        "sheet_name": c["metadata"].get("sheet_name"),
                    }
                    for c in final_chunks
                ]
                yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

            # ── MEMPALACE STATUS ─────────────────────────────────────────────
            yield f"data: {json.dumps({'type': 'mempalace', 'data': {'from_memory': from_mempalace, 'boost_used': mempalace_boost_used, 'prior_turns': len(session_history)}})}\n\n"

            # ── LLM STREAMING ────────────────────────────────────────────────
            memory_ctx = memory.build_context_summary(session_id)
            full_answer = []
            t3 = time.time()

            async for token in llm.stream_answer(
                request.question, final_chunks, memory_ctx
            ):
                full_answer.append(token)
                yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

            t4 = time.time()
            answer_text = "".join(full_answer)
            latency = round((t4 - start) * 1000, 1)

            logger.info(f"LATENCY | llm={round((t4 - t3) * 1000)}ms | total={latency}ms")

            # ── SAVE MEMORY WITH ENTITY EXTRACTION ──────────────────────────
            # Extract entities from Q+A and store in graph (rooms + drawers)
            combined_text = f"{request.question} {answer_text[:500]}"
            extracted_entities = extract_entities(combined_text)

            logger.info(f"MemPalace entities: {extracted_entities}")

            memory.save_episode(
                session_id,
                request.question,
                answer_text,
                [c["chunk_id"] for c in final_chunks],
                entities=extracted_entities,
            )

            # Log graph state for debugging
            graph_summary = memory.get_graph_summary(session_id)
            logger.info(
                f"MemPalace graph | "
                f"rooms={len(graph_summary['rooms'])} | "
                f"drawers={len(graph_summary['drawers'])} | "
                f"items={len(graph_summary['items'])}"
            )

            # ── CACHE RESULT ─────────────────────────────────────────────────
            result_to_cache = {
                "answer": answer_text,
                "sources": sources,
                "latency_ms": latency,
                "is_general": is_general,
            }
            cache.set(request.question, result_to_cache)

            # ── FINAL EVENT ──────────────────────────────────────────────────
            yield f"data: {json.dumps({'type': 'done', 'data': {'latency_ms': latency, 'session_id': session_id, 'is_general': is_general, 'threshold': retriever.RELEVANCE_THRESHOLD, 'chunks_used': len(final_chunks)}})}\n\n"

        except Exception as e:
            logger.error(f"Query error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )