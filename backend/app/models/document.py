from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    PDF = "pdf"
    EXCEL = "excel"
    DOCX = "docx"


class ChunkMetadata(BaseModel):
    doc_id: str
    chunk_index: int
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None
    row_range: Optional[str] = None
    file_name: str
    doc_type: DocumentType
    total_chunks: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    session_id: Optional[str] = None


class IngestResponse(BaseModel):
    doc_id: str
    file_name: str
    doc_type: DocumentType
    total_chunks: int
    status: str
    message: str


class QueryRequest(BaseModel):
    question: str
    session_id: str
    top_k: int = 5
    use_cache: bool = True


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    file_name: str
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    session_id: str
    cached: bool = False
    latency_ms: float