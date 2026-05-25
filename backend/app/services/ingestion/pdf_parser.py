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

        # Build the full page label map once for all pages
        label_map = self._build_label_map(doc)

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text").strip()
            printed_page = label_map.get(page_idx, page_idx + 1)

            pages.append({
                "page_number": printed_page,
                "page_index": page_idx + 1,
                "text": text,
                "has_tables": False,
            })
        doc.close()
        return pages

    def _build_label_map(self, doc: fitz.Document) -> dict:
        """
        Build a complete map of {page_index: printed_page_number} for all pages.
        Handles NCERT books which use Roman numerals for preface pages,
        then switch to Arabic numerals for chapter pages.
        Only maps Arabic numeral pages — Roman numeral pages get their file index.
        """
        label_map = {}
        total_pages = len(doc)

        try:
            page_labels = doc.get_page_labels()
            if not page_labels:
                # No labels — use 1-based index for all
                return {i: i + 1 for i in range(total_pages)}

            # Sort by startpage to process in order
            page_labels = sorted(page_labels, key=lambda x: x.get("startpage", 0))

            for label_idx, label_info in enumerate(page_labels):
                start = label_info.get("startpage", 0)
                style = label_info.get("style", "D")
                first = label_info.get("firstpagenum", 1)

                # Find end of this label range
                if label_idx + 1 < len(page_labels):
                    end = page_labels[label_idx + 1].get("startpage", total_pages)
                else:
                    end = total_pages

                for page_idx in range(start, end):
                    if style == "D" or style == "":
                        # Arabic numerals — this is the real chapter page number
                        label_map[page_idx] = first + (page_idx - start)
                    else:
                        # Roman numerals (style="r" or "R") or letters — use file index
                        label_map[page_idx] = page_idx + 1

        except Exception as e:
            logger.warning(f"Could not read page labels: {e}, using 1-based index")
            return {i: i + 1 for i in range(total_pages)}

        return label_map

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