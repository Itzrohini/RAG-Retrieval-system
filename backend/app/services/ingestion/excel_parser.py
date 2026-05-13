import pandas as pd
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any


class ExcelParser:
    """
    Parses Excel files (.xlsx, .xls, .csv).
    Each sheet becomes a separate block of text with row context.
    """

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        path = Path(file_path)
        suffix = path.suffix.lower()
        logger.info(f"Parsing Excel/CSV: {path.name}")

        if suffix == ".csv":
            return self._parse_csv(file_path)
        return self._parse_excel(file_path)

    def _parse_excel(self, file_path: str) -> List[Dict[str, Any]]:
        sheets = []
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            df = df.dropna(how="all").fillna("")
            if df.empty:
                continue
            text = self._df_to_text(df, sheet_name)
            sheets.append({
                "sheet_name": sheet_name,
                "text": text,
                "row_count": len(df),
                "col_count": len(df.columns),
                "row_range": f"1-{len(df)}",
            })
        logger.info(f"Parsed {len(sheets)} sheets")
        return sheets

    def _parse_csv(self, file_path: str) -> List[Dict[str, Any]]:
        df = pd.read_csv(file_path).fillna("")
        text = self._df_to_text(df, "Sheet1")
        return [{
            "sheet_name": "Sheet1",
            "text": text,
            "row_count": len(df),
            "col_count": len(df.columns),
            "row_range": f"1-{len(df)}",
        }]

    def _df_to_text(self, df: pd.DataFrame, sheet_name: str) -> str:
        """Convert dataframe to a readable text block."""
        lines = [f"Sheet: {sheet_name}"]
        lines.append("Columns: " + " | ".join(str(c) for c in df.columns))
        lines.append("---")
        # Chunk rows into groups of 50 for readability
        for i in range(0, len(df), 50):
            chunk = df.iloc[i:i+50]
            lines.append(f"Rows {i+1}-{min(i+50, len(df))}:")
            for _, row in chunk.iterrows():
                row_text = " | ".join(f"{col}: {val}" for col, val in row.items() if str(val).strip())
                lines.append(row_text)
        return "\n".join(lines)