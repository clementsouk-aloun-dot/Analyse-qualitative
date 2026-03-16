from __future__ import annotations

import importlib
import re
from typing import Iterable, List, Sequence

import pandas as pd
from unidecode import unidecode

DEFAULT_SPACY_MODELS = [
    "fr_core_news_sm",
    "fr_core_news_md",
    "fr_core_news_lg",
]


def discover_available_spacy_models() -> List[str]:
    available = []
    for model_name in DEFAULT_SPACY_MODELS:
        try:
            importlib.import_module(model_name)
            available.append(model_name)
        except Exception:
            continue
    return available


def normalize_text(
    text: object,
    *,
    strip_accents: bool = False,
    remove_numbers: bool = False,
) -> str:
    if pd.isna(text):
        return ""
    value = str(text).replace("\xa0", " ").replace("’", "'")
    value = re.sub(r"\s+", " ", value.strip())
    if remove_numbers:
        value = re.sub(r"\d+", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
    if strip_accents:
        value = unidecode(value)
    return value


def spacy_lemmas(doc, keep_pos: Sequence[str]) -> List[str]:
    keep_pos_set = set(keep_pos)
    tokens: List[str] = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.like_num or tok.like_url or tok.like_email:
            continue
        if tok.pos_ in keep_pos_set:
            lemma = tok.lemma_.lower().strip()
            lemma = re.sub(r"[^a-zàâçéèêëîïôùûüÿœ'\- ]", " ", lemma)
            lemma = re.sub(r"\s+", " ", lemma).strip()
            if lemma:
                tokens.append(lemma)
    return tokens


def preprocess_dataframe(
    df: pd.DataFrame,
    *,
    id_col: str,
    text_col: str,
    nlp,
    keep_pos: Sequence[str],
    strip_accents: bool = False,
    remove_numbers: bool = False,
    batch_size: int = 128,
) -> pd.DataFrame:
    out = df.copy()
    out["__source_id"] = out[id_col].astype(str)
    out["__source_text"] = out[text_col].fillna("").astype(str)
    out["texte_norm"] = out["__source_text"].map(
        lambda x: normalize_text(
            x,
            strip_accents=strip_accents,
            remove_numbers=remove_numbers,
        )
    )
    docs = list(nlp.pipe(out["texte_norm"].tolist(), batch_size=batch_size, n_process=1))
    out["lemmas"] = [spacy_lemmas(doc, keep_pos) for doc in docs]
    out["lemmas_str"] = out["lemmas"].apply(lambda xs: " ".join(xs))
    out["lemmas_set"] = out["lemmas"].apply(set)
    out["texte_lower"] = out["texte_norm"].str.lower()
    out["texte_noaccent"] = out["texte_lower"].map(unidecode)
    out["texte_vide"] = out["texte_norm"].eq("")
    out["nb_lemmas"] = out["lemmas"].apply(len)
    return out
