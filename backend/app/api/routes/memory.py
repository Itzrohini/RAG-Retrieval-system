from fastapi import APIRouter
from loguru import logger
from app.services.memory.mempalace import MemPalace

router = APIRouter()


@router.get("/memory/{session_id}")
async def get_memory_graph(session_id: str):
    """
    Returns the full MemPalace graph summary for a session.
    Used by the frontend debug panel to show rooms, drawers, items.
    """
    try:
        memory = MemPalace()
        summary = memory.get_graph_summary(session_id)
        history = memory.get_session_history(session_id)

        return {
            "ok": True,
            "session_id": session_id,
            "graph": summary,
            "turns": len(history),
            "episodes": [
                {
                    "question": ep["question"],
                    "entities": ep.get("entities", []),
                    "chunk_ids": ep.get("chunk_ids", []),
                    "timestamp": ep.get("timestamp", ""),
                }
                for ep in history[-5:]  # last 5 turns only
            ],
        }
    except Exception as e:
        logger.error(f"Memory graph fetch failed: {e}")
        return {"ok": False, "error": str(e)}


@router.delete("/memory/{session_id}")
async def clear_memory(session_id: str):
    """Clear all memory for a session."""
    try:
        memory = MemPalace()
        memory.clear_session(session_id)
        return {"ok": True, "message": f"Session {session_id} cleared"}
    except Exception as e:
        return {"ok": False, "error": str(e)}