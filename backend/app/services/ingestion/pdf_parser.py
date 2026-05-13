import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any


class PDFParser:
    """
    Parses PDF files extracting text page by page.
    Uses PyMuPDF for speed, falls back to pdfplumber for complex layouts.
    """

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Returns a list of page dicts:
        [{"page_number": 1, "text": "...", "has_tables": False}, ...]
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        logger.info(f"Parsing PDF: {path.name}")
        pages = []

        try:
            pages = self._parse_with_pymupdf(file_path)
            # If many pages have very short text, retry with pdfplumber
            short_pages = sum(1 for p in pages if len(p["text"]) < 50)
            if short_pages > len(pages) * 0.3:
                logger.warning("Many short pages detected, switching to pdfplumber")
                pages = self._parse_with_pdfplumber(file_path)
        except Exception as e:
            logger.error(f"PyMuPDF failed: {e}, trying pdfplumber")
            pages = self._parse_with_pdfplumber(file_path)

        logger.info(f"Parsed {len(pages)} pages from {path.name}")
        return pages

    def _parse_with_pymupdf(self, file_path: str) -> List[Dict[str, Any]]:
        pages = []
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            pages.append({
                "page_number": page_num + 1,
                "text": text,
                "has_tables": False,
            })
        doc.close()
        return pages

    def _parse_with_pdfplumber(self, file_path: str) -> List[Dict[str, Any]]:
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                tables = page.extract_tables()
                table_text = ""
                for table in tables:
                    for row in table:
                        cleaned = [str(cell) if cell else "" for cell in row]
                        table_text += " | ".join(cleaned) + "\n"
                full_text = (text + "\n" + table_text).strip()
                pages.append({
                    "page_number": page_num + 1,
                    "text": full_text,
                    "has_tables": len(tables) > 0,
                })
        return pages