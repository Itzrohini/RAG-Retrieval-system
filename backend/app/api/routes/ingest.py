import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from loguru import logger
from app.config import settings
from app.models.document import IngestResponse, DocumentType
from app.services.ingestion.pdf_parser import PDFParser
from app.services.ingestion.excel_parser import ExcelParser
from app.services.ingestion.chunker import SemanticChunker
from app.services.embedding.embedder import BGEEmbedder
from app.services.vectorstore.qdrant_store import QdrantStore
from app.services.database.mongo_store import MongoStore

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".csv"}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a PDF or Excel file.
    The file is parsed, chunked, embedded, and stored in Qdrant + MongoDB.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    file_bytes = await file.read()
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_file_size_mb}MB",
        )

    doc_id = str(uuid.uuid4())
    save_path = Path(settings.upload_dir) / f"{doc_id}{suffix}"
    with open(save_path, "wb") as f:
        f.write(file_bytes)

    logger.info(f"Saved upload: {save_path}")

    # Run heavy processing in background
    background_tasks.add_task(
        process_document,
        str(save_path),
        doc_id,
        file.filename,
        suffix,
    )

    doc_type = DocumentType.PDF if suffix == ".pdf" else DocumentType.EXCEL
    return IngestResponse(
        doc_id=doc_id,
        file_name=file.filename,
        doc_type=doc_type,
        total_chunks=0,  # updated when background task completes
        status="processing",
        message="File received. Processing in background. Check /health for status.",
    )


async def process_document(
    file_path: str,
    doc_id: str,
    file_name: str,
    suffix: str,
):
    """Background task: parse → chunk → embed → store."""
    try:
        chunker = SemanticChunker()
        embedder = BGEEmbedder()
        qdrant = QdrantStore()
        mongo = MongoStore()

        # Parse
        if suffix == ".pdf":
            parser = PDFParser()
            pages = parser.parse(file_path)
            chunks = chunker.chunk_pdf_pages(pages, doc_id, file_name)
        else:
            parser = ExcelParser()
            sheets = parser.parse(file_path)
            chunks = chunker.chunk_excel_sheets(sheets, doc_id, file_name)

        if not chunks:
            logger.warning(f"No chunks extracted from {file_name}")
            return

        # Embed
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_texts(texts)

        # Prepare for Qdrant
        qdrant_items = []
        mongo_chunks = []
        for chunk, emb in zip(chunks, embeddings):
            qdrant_items.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "dense": emb["dense"],
                "sparse": emb["sparse"],
                "metadata": chunk.metadata.model_dump(),
            })
            mongo_chunks.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": doc_id,
                "text": chunk.text,
                "file_name": file_name,
                "page_number": chunk.metadata.page_number,
                "sheet_name": chunk.metadata.sheet_name,
            })

        # Store
        qdrant.upsert_chunks(qdrant_items)
        await mongo.save_chunks(mongo_chunks)
        await mongo.save_document({
            "doc_id": doc_id,
            "file_name": file_name,
            "doc_type": suffix.lstrip("."),
            "total_chunks": len(chunks),
            "status": "ready",
        })

        logger.info(f"Document {file_name} fully processed: {len(chunks)} chunks")

    except Exception as e:
        logger.error(f"Failed to process {file_name}: {e}")