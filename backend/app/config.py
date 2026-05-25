from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # App
    app_name: str = "RAG System"
    app_env: str = "development"
    debug: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # LLM Provider — switch between ollama and gemini
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # MongoDB
    mongodb_url: str = Field(..., alias="MONGODB_URL")
    mongodb_db_name: str = "rag_system"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "rag_documents"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_ttl_seconds: int = 3600

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "tinyllama:latest"  # Smaller model for lower memory usage

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 100
    max_chunks_per_doc: int = 500

    # Retrieval
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_bm25: int = 20
    top_k_rerank: int = 5

    # Memory optimization
    enable_reranking: bool = True  # Disabled to save memory
    memory_optimization: bool = True  # Enable lazy loading and memory management

    # Upload
    max_file_size_mb: int = 50
    upload_dir: str = "./uploads"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
print("Loaded Mongo URL:", settings.mongodb_url)

# Ensure upload dir exists
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)