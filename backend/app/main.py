import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from app.config import settings
from app.api.routes import ingest, query, health
from app.api.routes import ingest, query, health, memory


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name}")

    from app.services.vectorstore.qdrant_store import QdrantStore
    from app.services.database.mongo_store import MongoStore

    qdrant = QdrantStore()

    mongo = MongoStore()
    await mongo.ensure_indexes()

    logger.info("Pre-loading embedding model...")
    try:
        from app.services.embedding.embedder import BGEEmbedder
        embedder = BGEEmbedder()
        _ = embedder.embed_query("warmup")  
        embedder.unload()  # free RAM — reloads automatically on first query
        logger.info("Embedding model warmed up and unloaded")   
    except Exception as e:
        logger.warning(f"Model pre-load failed: {e}")

        embedder = BGEEmbedder()
        _ = embedder.embed_query("warmup")
        embedder.unload()  # free RAM — reloads automatically on first query
        logger.info("Embedding model warmed up and unloaded")
    except Exception as e:
        logger.warning(f"Model pre-load failed: {e}")

    logger.info("All services initialized")
    try:
        yield
    except asyncio.CancelledError:
        logger.info("Lifespan cancelled")
    finally:
        logger.info("Shutting down")


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")
app.include_router(memory.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"status": "ok", "app": settings.app_name}