import fitz
import pdfplumber
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any


class PDFParser:
    """
    Parses PDF files extracting text page by page.
    Stores the ACTUAL page number printed in the PDF
    by reading the page label, not just the index.
    """

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        logger.info(f"Parsing PDF: {path.name}")
        pages = []

        try:
            pages = self._parse_with_pymupdf(file_path)
            short_pages = sum(1 for p in pages if len(p["text"]) < 50)
            if short_pages > len(pages) * 0.3:
                logger.warning("Many short pages, switching to pdfplumber")
                pages = self._parse_with_pdfplumber(file_path)
        except Exception as e:
            logger.error(f"PyMuPDF failed: {e}, trying pdfplumber")
            pages = self._parse_with_pdfplumber(file_path)

        logger.info(f"Parsed {len(pages)} pages from {path.name}")
        return pages

    def _parse_with_pymupdf(self, file_path: str) -> List[Dict[str, Any]]:
        pages = []
        doc = fitz.open(file_path)

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text").strip()

            # Get the printed page number from PDF labels
            # This matches what you see printed in the book
            printed_page = self._get_printed_page_number(doc, page_idx)

            pages.append({
                "page_number": printed_page,       # printed number (e.g. 90)
                "page_index": page_idx + 1,        # file position (e.g. 95)
                "text": text,
                "has_tables": False,
            })
        doc.close()
        return pages

    def _get_printed_page_number(self, doc: fitz.Document, page_idx: int) -> int:
        """
        Try to get the actual printed page number from the PDF.
        NCERT books have Roman numerals for preface, then Arabic for chapters.
        We want the Arabic numeral page numbers.
        """
        try:
            # Method 1: Read page label from PDF metadata
            page_label = doc.get_page_labels()
            if page_label:
                for label_info in page_label:
                    start = label_info.get("startpage", 0)
                    style = label_info.get("style", "")
                    prefix = label_info.get("prefix", "")
                    first = label_info.get("firstpagenum", 1)

                    if style in ("D", "") and page_idx >= start:
                        return first + (page_idx - start)

            # Method 2: Look for printed number in page text
            page = doc[page_idx]
            text = page.get_text("text").strip()
            lines = [l.strip() for l in text.split("\n") if l.strip()]

            # Page numbers are usually on first or last line, short, numeric
            for line in [lines[0], lines[-1]] if len(lines) > 1 else lines:
                if line.isdigit() and 1 <= int(line) <= 9999:
                    return int(line)

        except Exception:
            pass

        # Fallback: use 1-indexed position in file
        return page_idx + 1

    def _parse_with_pdfplumber(self, file_path: str) -> List[Dict[str, Any]]:
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                tables = page.extract_tables()
                table_text = ""
                for table in tables:
                    for row in table:
                        cleaned = [str(cell) if cell else "" for cell in row]
                        table_text += " | ".join(cleaned) + "\n"
                full_text = (text + "\n" + table_text).strip()

                pages.append({
                    "page_number": page_idx + 1,
                    "page_index": page_idx + 1,
                    "text": full_text,
                    "has_tables": len(tables) > 0,
                })
        return pages