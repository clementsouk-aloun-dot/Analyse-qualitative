from __future__ import annotations

from io import BytesIO
from typing import List

import pandas as pd


SUPPORTED_EXTENSIONS = ["xlsx", "xls"]


def list_sheet_names(file_bytes: bytes) -> List[str]:
    """Return all sheet names from an Excel file loaded in memory."""
    excel = pd.ExcelFile(BytesIO(file_bytes))
    return excel.sheet_names


def read_excel_sheet(file_bytes: bytes, sheet_name: str | int) -> pd.DataFrame:
    """Read one sheet from an Excel file stored as bytes."""
    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)
    df.columns = [str(col).strip() for col in df.columns]
    return df
