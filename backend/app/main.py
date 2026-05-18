from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from app.config import settings
from app.api.routes import health, ingest, query
from app.services.database.mongo_store import MongoStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name}")

    mongo = MongoStore()
    await mongo.ensure_indexes()

    # Pre-load embedding model at startup so first query is fast
    logger.info("Pre-loading BGE-M3 embedding model...")

    try:
        from app.services.embedding.embedder import BGEEmbedder

        embedder = BGEEmbedder()
        _ = embedder.embed_query("warmup")

        logger.info("BGE-M3 model ready")

    except Exception as e:
        logger.warning(f"Model pre-load failed: {e}")

    logger.info("All services initialized")

    yield

    logger.info("Shutting down")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan
)

# CORS — allows the React frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(
    health.router,
    tags=["Health"]
)

app.include_router(
    ingest.router,
    prefix="/api/v1",
    tags=["Ingest"]
)

app.include_router(
    query.router,
    prefix="/api/v1",
    tags=["Query"]
)


@app.get("/")
async def root():
    return {
        "message": "RAG System API is running",
        "docs": "/docs"
    }