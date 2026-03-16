from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def compute_frequencies(
    lemmas_str: pd.Series,
    *,
    min_df_uni: int = 3,
    min_df_bi: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    series = lemmas_str.fillna("")

    top_words = pd.DataFrame(columns=["term", "freq"])
    top_bigrams = pd.DataFrame(columns=["bigram", "freq"])

    if (series.str.strip() != "").sum() == 0:
        return top_words, top_bigrams

    try:
        cv_uni = CountVectorizer(
            token_pattern=r"(?u)\b\w[\w'\-]+\b",
            lowercase=True,
            min_df=max(1, int(min_df_uni)),
        )
        x_uni = cv_uni.fit_transform(series)
        terms = np.array(cv_uni.get_feature_names_out())
        freqs = np.asarray(x_uni.sum(axis=0)).ravel()
        top_words = (
            pd.DataFrame({"term": terms, "freq": freqs})
            .sort_values("freq", ascending=False)
            .reset_index(drop=True)
        )
    except ValueError:
        pass

    try:
        cv_bi = CountVectorizer(
            ngram_range=(2, 2),
            token_pattern=r"(?u)\b\w[\w'\-]+\b",
            lowercase=True,
            min_df=max(1, int(min_df_bi)),
        )
        x_bi = cv_bi.fit_transform(series)
        bigrams = np.array(cv_bi.get_feature_names_out())
        bfreqs = np.asarray(x_bi.sum(axis=0)).ravel()
        top_bigrams = (
            pd.DataFrame({"bigram": bigrams, "freq": bfreqs})
            .sort_values("freq", ascending=False)
            .reset_index(drop=True)
        )
    except ValueError:
        pass

    return top_words, top_bigrams


def build_tfidf_matrix(
    lemmas_str: pd.Series,
    *,
    min_df_tfidf: int = 3,
    ngram_range: tuple[int, int] = (1, 2),
):
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w[\w'\-]+\b",
        lowercase=True,
        min_df=max(1, int(min_df_tfidf)),
        ngram_range=ngram_range,
    )
    x = vectorizer.fit_transform(lemmas_str.fillna(""))
    vocab = np.array(vectorizer.get_feature_names_out())
    return x, vocab, vectorizer


def top_features_for_cluster(
    x,
    labels: np.ndarray,
    vocab: np.ndarray,
    *,
    k: int,
    topn: int = 12,
) -> List[List[str]]:
    out: List[List[str]] = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            out.append([])
            continue
        centroid = x[idx].mean(axis=0)
        arr = np.asarray(centroid).ravel()
        top_idx = arr.argsort()[::-1][:topn]
        out.append([str(vocab[i]) for i in top_idx])
    return out


def choose_k_auto(
    x,
    *,
    k_min: int = 3,
    k_max: int = 10,
    method: str = "composite",
    random_state: int = 42,
    n_init: int = 20,
    penalty_lambda: float = 0.12,
    min_cluster_size_ratio: float = 0.02,
    svd_components: int = 50,
):
    n, p = x.shape
    if n < 3:
        raise ValueError("Pas assez de lignes pour faire un clustering automatique.")

    k_min = max(2, min(k_min, n - 1))
    k_max = max(k_min, min(k_max, n - 1))

    if method == "silhouette":
        best = {"k": None, "score": -1, "labels": None, "model": None}
        rows = []
        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
            labels = km.fit_predict(x)
            if len(set(labels)) < 2:
                continue
            try:
                score = silhouette_score(x, labels, sample_size=min(10000, n))
            except Exception:
                score = -1
            rows.append({"k": k, "silhouette": score})
            if score > best["score"]:
                best = {"k": k, "score": score, "labels": labels, "model": km}
        eval_df = pd.DataFrame(rows)
        if best["k"] is None:
            raise ValueError("Impossible de sélectionner automatiquement K.")
        return best["k"], best["labels"], best["model"], eval_df

    max_svd_dim = min(n - 1, p - 1)
    if max_svd_dim >= 2:
        svd_dim = max(2, min(svd_components, max_svd_dim))
        xr = TruncatedSVD(n_components=svd_dim, random_state=random_state).fit_transform(x)
    else:
        xr = x.toarray()

    eval_rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(xr)
        if len(set(labels)) < 2:
            continue
        sizes = np.bincount(labels, minlength=k)
        too_small = (sizes < max(2, int(min_cluster_size_ratio * n))).sum()
        size_penalty = too_small / k
        sil = silhouette_score(xr, labels, sample_size=min(10000, n))
        ch = calinski_harabasz_score(xr, labels)
        db = davies_bouldin_score(xr, labels)
        eval_rows.append(
            {
                "k": k,
                "silhouette": sil,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
                "size_penalty": size_penalty,
            }
        )

    eval_df = pd.DataFrame(eval_rows)
    if eval_df.empty:
        raise ValueError("Impossible de sélectionner automatiquement K.")

    def _norm(values, invert: bool = False):
        arr = np.array(values, dtype=float)
        if invert:
            arr = -arr
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        if mx - mn < 1e-9:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn + 1e-9)

    eval_df["sil_n"] = _norm(eval_df["silhouette"])
    eval_df["ch_n"] = _norm(eval_df["calinski_harabasz"])
    eval_df["db_n"] = _norm(eval_df["davies_bouldin"], invert=True)
    eval_df["complexity_penalty"] = penalty_lambda * (eval_df["k"] - k_min) / max(1, (k_max - k_min))
    eval_df["final_score"] = (
        0.50 * eval_df["sil_n"]
        + 0.35 * eval_df["ch_n"]
        + 0.15 * eval_df["db_n"]
        - 0.30 * eval_df["size_penalty"]
        - eval_df["complexity_penalty"]
    )
    eval_df = eval_df.sort_values("k").reset_index(drop=True)
    best_idx = eval_df["final_score"].idxmax()
    best_k = int(eval_df.loc[best_idx, "k"])
    if best_k == k_max:
        prev = eval_df[eval_df["k"] == (k_max - 1)]
        if not prev.empty and (eval_df.loc[best_idx, "final_score"] - prev["final_score"].iloc[0]) < 0.02:
            best_k = k_max - 1

    final_model = KMeans(n_clusters=best_k, n_init=n_init, random_state=random_state).fit(xr)
    labels = final_model.labels_
    return best_k, labels, final_model, eval_df


def fit_manual_k(x, *, k: int, random_state: int = 42, n_init: int = 20):
    model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(x)
    return model.labels_, model


def build_clusters_dataframe(
    processed_df: pd.DataFrame,
    labels: np.ndarray,
    x,
    vocab: np.ndarray,
    *,
    topn: int = 12,
    examples_per_cluster: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    k = int(len(np.unique(labels)))
    cluster_tops = top_features_for_cluster(x, labels, vocab, k=k, topn=topn)
    work = processed_df.copy()
    work["cluster_auto_id"] = labels

    cluster_rows = []
    example_rows = []
    for c in range(k):
        tops = cluster_tops[c]
        label_auto = ", ".join(tops[:5]) if tops else f"Cluster {c}"
        members = work[work["cluster_auto_id"] == c].copy()
        cluster_rows.append(
            {
                "cluster_auto_id": c,
                "label_auto": label_auto,
                "top_terms": " | ".join(tops),
                "taille": int(len(members)),
            }
        )
        for _, row in members.head(examples_per_cluster).iterrows():
            example_rows.append(
                {
                    "cluster_auto_id": c,
                    "id": row["__source_id"],
                    "texte": row["__source_text"],
                }
            )

    clusters_df = pd.DataFrame(cluster_rows).sort_values("taille", ascending=False).reset_index(drop=True)
    examples_df = pd.DataFrame(example_rows)
    return clusters_df, examples_df
