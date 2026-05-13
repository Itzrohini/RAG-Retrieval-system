import uuid
from typing import List, Dict, Any
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings
from app.models.document import DocumentChunk, ChunkMetadata, DocumentType


class SemanticChunker:
    """
    Splits parsed pages/sheets into overlapping chunks.
    Attaches rich metadata to every chunk for downstream filtering.
    """

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_pdf_pages(
        self,
        pages: List[Dict[str, Any]],
        doc_id: str,
        file_name: str,
    ) -> List[DocumentChunk]:
        chunks = []
        for page in pages:
            text = page.get("text", "").strip()
            if not text:
                continue
            splits = self.splitter.split_text(text)
            for i, split in enumerate(splits):
                if len(split.strip()) < 20:  # skip tiny fragments
                    continue
                chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=split.strip(),
                    metadata=ChunkMetadata(
                        doc_id=doc_id,
                        chunk_index=len(chunks),
                        page_number=page["page_number"],
                        file_name=file_name,
                        doc_type=DocumentType.PDF,
                        total_chunks=0,  # updated after
                    ),
                )
                chunks.append(chunk)

        # Update total_chunks now that we know
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)

        logger.info(f"PDF chunked into {len(chunks)} chunks")
        return chunks[:settings.max_chunks_per_doc]

    def chunk_excel_sheets(
        self,
        sheets: List[Dict[str, Any]],
        doc_id: str,
        file_name: str,
    ) -> List[DocumentChunk]:
        chunks = []
        for sheet in sheets:
            text = sheet.get("text", "").strip()
            if not text:
                continue
            splits = self.splitter.split_text(text)
            for split in splits:
                if len(split.strip()) < 20:
                    continue
                chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=split.strip(),
                    metadata=ChunkMetadata(
                        doc_id=doc_id,
                        chunk_index=len(chunks),
                        sheet_name=sheet["sheet_name"],
                        row_range=sheet.get("row_range"),
                        file_name=file_name,
                        doc_type=DocumentType.EXCEL,
                        total_chunks=0,
                    ),
                )
                chunks.append(chunk)

        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)

        logger.info(f"Excel chunked into {len(chunks)} chunks")
        return chunks[:settings.max_chunks_per_doc]