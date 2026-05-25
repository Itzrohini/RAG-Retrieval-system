import json
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any


class JSONParser:
    """
    Parses JSON and JSONL files for RAG ingestion.

    Supported formats:
    - .json  : array of objects, or single object
    - .jsonl : one JSON object per line (common for datasets, chat logs, QA pairs)

    Each object is converted to readable text for embedding.
    Nested objects are flattened to key: value format.
    """

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        path = Path(file_path)
        suffix = path.suffix.lower()
        logger.info(f"Parsing {suffix.upper()}: {path.name}")

        if suffix == ".jsonl":
            return self._parse_jsonl(file_path)
        return self._parse_json(file_path)

    def _parse_json(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both array and single object
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = [data]
        else:
            records = [{"value": str(data)}]

        return self._records_to_pages(records)

    def _parse_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")

        logger.info(f"Parsed {len(records)} records from JSONL")
        return self._records_to_pages(records)

    def _records_to_pages(self, records: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert JSON records to page-like dicts for the chunker.
        Groups records into blocks of 10 per 'page' to avoid
        creating thousands of tiny chunks.
        """
        pages = []
        block_size = 10

        for i in range(0, len(records), block_size):
            block = records[i:i + block_size]
            text_lines = []

            for j, record in enumerate(block):
                text_lines.append(f"--- Record {i + j + 1} ---")
                text_lines.append(self._flatten(record))

            pages.append({
                "page_number": (i // block_size) + 1,
                "text": "\n".join(text_lines),
                "record_range": f"{i+1}-{min(i+block_size, len(records))}",
                "has_tables": False,
            })

        logger.info(f"Converted {len(records)} records into {len(pages)} page blocks")
        return pages

    def _flatten(self, obj: Any, prefix: str = "", depth: int = 0) -> str:
        """Recursively flatten nested JSON to readable text."""
        if depth > 5:  # prevent infinite recursion on deeply nested data
            return f"{prefix}: {str(obj)[:200]}"

        lines = []
        if isinstance(obj, dict):
            for key, val in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(val, (dict, list)):
                    lines.append(self._flatten(val, full_key, depth + 1))
                else:
                    lines.append(f"{full_key}: {val}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj[:20]):  # limit list items
                full_key = f"{prefix}[{i}]" if prefix else f"[{i}]"
                if isinstance(item, (dict, list)):
                    lines.append(self._flatten(item, full_key, depth + 1))
                else:
                    lines.append(f"{full_key}: {item}")
            if len(obj) > 20:
                lines.append(f"{prefix}: ... ({len(obj) - 20} more items)")
        else:
            lines.append(f"{prefix}: {obj}" if prefix else str(obj))

        return "\n".join(lines)