from __future__ import annotations

from io import BytesIO
from typing import Dict

import pandas as pd


EXPORT_SHEET_ORDER = [
    "input_clean",
    "top_words",
    "top_bigrams",
    "k_evaluation",
    "clusters_auto",
    "cluster_examples",
    "themes_final",
    "verbatims_thematiques",
    "verbatims_codes",
    "semantic_summary",
]


def workbook_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name in EXPORT_SHEET_ORDER:
            if sheet_name in sheets and sheets[sheet_name] is not None:
                df = sheets[sheet_name]
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    output.seek(0)
    return output.getvalue()
