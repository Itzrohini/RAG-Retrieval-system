from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from app.config import settings
from app.api.routes import health, ingest, query
from app.services.database.mongo_store import MongoStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info(f"Starting {settings.app_name} in {settings.app_env} mode")

    # Ensure MongoDB indexes on startup
    mongo = MongoStore()
    await mongo.ensure_indexes()

    logger.info("All services initialized")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="RAG System API",
    description="Production RAG with Hybrid Retrieval + MemPalace",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allows the React frontend (port 5173) to call this backend (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health.router, tags=["Health"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])


@app.get("/")
async def root():
    return {"message": "RAG System API is running", "docs": "/docs"}