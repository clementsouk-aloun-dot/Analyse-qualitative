from __future__ import annotations

import re
from typing import Dict, Iterable, List

import pandas as pd
from unidecode import unidecode


DEFAULT_MATCH_MODE = "lemma"
DEFAULT_LOGIC = "any"


def sanitize_theme_code(label: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(label).strip().upper())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "THEME"


def build_default_themes_df(clusters_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = clusters_df.sort_values("cluster_auto_id").copy()
    for idx, row in ordered.iterrows():
        top_terms = [t.strip() for t in str(row["top_terms"]).split("|") if t.strip()]
        default_label = top_terms[0] if top_terms else f"Theme_{int(row['cluster_auto_id'])}"
        rows.append(
            {
                "final_theme_id": f"T{idx + 1}",
                "final_theme_label": str(default_label).strip().title(),
                "source_clusters": str(int(row["cluster_auto_id"])),
                "tags": " | ".join(top_terms[:6]),
                "match_mode": DEFAULT_MATCH_MODE,
                "logic": DEFAULT_LOGIC,
                "active": True,
            }
        )
    return pd.DataFrame(rows)


def parse_source_clusters(value: object) -> List[int]:
    if pd.isna(value):
        return []
    text = str(value).replace(",", "|").replace(";", "|")
    out = []
    for part in text.split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return sorted(set(out))


def clean_themes_df(themes_df: pd.DataFrame) -> pd.DataFrame:
    work = themes_df.copy()
    expected_cols = [
        "final_theme_id",
        "final_theme_label",
        "source_clusters",
        "tags",
        "match_mode",
        "logic",
        "active",
    ]
    for col in expected_cols:
        if col not in work.columns:
            work[col] = ""
    work = work[expected_cols].copy()
    work["final_theme_id"] = work["final_theme_id"].fillna("").astype(str).str.strip()
    work["final_theme_label"] = work["final_theme_label"].fillna("").astype(str).str.strip()
    work["source_clusters"] = work["source_clusters"].fillna("").astype(str).str.strip()
    work["tags"] = work["tags"].fillna("").astype(str).str.strip()
    work["match_mode"] = work["match_mode"].fillna(DEFAULT_MATCH_MODE).astype(str).str.strip().str.lower()
    work["logic"] = work["logic"].fillna(DEFAULT_LOGIC).astype(str).str.strip().str.lower()
    work["active"] = work["active"].fillna(False).astype(bool)
    work = work[(work["active"]) & (work["final_theme_label"] != "")].reset_index(drop=True)
    return work


def assign_themes_from_clusters(processed_df: pd.DataFrame, themes_df: pd.DataFrame) -> pd.DataFrame:
    out = processed_df.copy()
    cluster_to_theme: Dict[int, str] = {}
    cluster_to_theme_id: Dict[int, str] = {}

    for _, row in clean_themes_df(themes_df).iterrows():
        for cluster_id in parse_source_clusters(row["source_clusters"]):
            cluster_to_theme[cluster_id] = str(row["final_theme_label"])
            cluster_to_theme_id[cluster_id] = str(row["final_theme_id"])

    out["theme_final_id"] = out["cluster_auto_id"].map(cluster_to_theme_id).fillna("")
    out["theme_final_label"] = out["cluster_auto_id"].map(cluster_to_theme).fillna("Non attribué")
    return out


def match_record(row: pd.Series, triggers: List[str], match_mode: str, logic: str) -> bool:
    flags: List[bool] = []

    if match_mode == "lemma":
        for trig in triggers:
            toks = [t for t in trig.strip().split() if t]
            if not toks:
                flags.append(False)
                continue
            if len(toks) == 1:
                flags.append(toks[0] in row["lemmas_set"])
            else:
                flags.append((" " + trig + " ") in (" " + row["lemmas_str"] + " "))
    elif match_mode == "contains":
        tx = row["texte_noaccent"]
        for trig in triggers:
            trigx = unidecode(trig.strip().lower())
            flags.append(trigx in tx)
    elif match_mode == "regex":
        tx = row["texte_norm"]
        for trig in triggers:
            try:
                flags.append(bool(re.search(trig, tx, flags=re.IGNORECASE)))
            except re.error:
                flags.append(False)
    else:
        flags = [False] * len(triggers)

    return any(flags) if logic == "any" else all(flags)


def build_binary_coding(processed_df: pd.DataFrame, themes_df: pd.DataFrame):
    out = processed_df.copy()
    summary_rows = []
    coded_columns = []

    for _, row in clean_themes_df(themes_df).iterrows():
        label = sanitize_theme_code(row["final_theme_label"])
        triggers = [t.strip() for t in str(row["tags"]).split("|") if t.strip()]
        match_mode = str(row.get("match_mode", DEFAULT_MATCH_MODE)).strip().lower() or DEFAULT_MATCH_MODE
        logic = str(row.get("logic", DEFAULT_LOGIC)).strip().lower() or DEFAULT_LOGIC
        out[label] = out.apply(
            lambda r: int(match_record(r, triggers, match_mode, logic)) if triggers else 0,
            axis=1,
        )
        coded_columns.append(label)

    n = len(out)
    for col in coded_columns:
        k = int(out[col].sum())
        summary_rows.append(
            {
                "theme_code": col,
                "n": k,
                "pct": round(100 * k / n, 2) if n else 0.0,
            }
        )

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["n", "theme_code"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return out, summary_df, coded_columns
