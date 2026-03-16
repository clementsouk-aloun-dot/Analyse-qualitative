
# streamlit_app.py
# -------------------------------------------------------------
# Analyse qualitative assistée - champs sémantiques + recodage + sentiment
# Interface simple pour néophytes (upload de fichier, exploration,
# proposition de champs, mapping utilisateur, recodage 0/1, sentiment).
# -------------------------------------------------------------

import io
import os
import re
import json
import time
import base64
import chardet
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from unidecode import unidecode

# ML / NLP
import warnings
warnings.filterwarnings("ignore")

# Lazy imports (accélère le chargement initial)
_spacy = None
_nlp = None
def lazy_import_spacy():
    global _spacy
    if _spacy is None:
        import spacy as _sp
        return _sp
    return _spacy

def get_nlp():
    """Charge spaCy fr_core_news_sm, avec fallback si nécessaire."""
    global _nlp, _spacy
    if _nlp is not None:
        return _nlp
    _spacy = lazy_import_spacy()
    try:
        _nlp = _spacy.load("fr_core_news_sm", disable=["ner"])
    except Exception:
        # tentative d'installation en local si possible
        try:
            from spacy.cli import download as spacy_download
            spacy_download("fr_core_news_sm")
            _nlp = _spacy.load("fr_core_news_sm", disable=["ner"])
        except Exception:
            _nlp = None  # on gèrera un mode dégradé
    if _nlp is not None and "sentencizer" not in _nlp.pipe_names:
        _nlp.add_pipe("sentencizer")
    return _nlp

# Vectorisation & clustering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Optionnel: HDBSCAN (si l'utilisateur veut une alternative sans K)
try:
    import hdbscan
    _has_hdbscan = True
except Exception:
    _has_hdbscan = False

# Sentiment (facultatif)
_hf_pipe = None
def get_sentiment_pipeline(model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment"):
    global _hf_pipe
    if _hf_pipe is not None:
        return _hf_pipe
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        _hf_pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            return_all_scores=True
        )
        return _hf_pipe
    except Exception as e:
        st.warning("Le module 'transformers' n'est pas disponible ou le modèle n'a pas pu être chargé. "
                   "Désactive le sentiment ou installe les dépendances.")
        return None

# -------------------- UI --------------------

st.set_page_config(page_title="Analyse quali assistée", layout="wide")
st.title("🧠 Analyse qualitative assistée (champs sémantiques + recodage + sentiment)")

with st.sidebar:
    st.header("⚙️ Paramètres")

    st.markdown("**Étape 1 — Données**")
    up = st.file_uploader("Dépose ton CSV/XLSX avec au moins 'id' & 'texte' :", type=["csv","xlsx","xls"])
    sep_guess = st.selectbox("Séparateur CSV (si CSV)", [None, ",",";","\t","|"], index=0, help="Laisse None pour auto-détection.")
    sample_n = st.number_input("Échantillon (0 = tout)", min_value=0, value=0, step=100)
    st.markdown("---")

    st.markdown("**Étape 2 — Prétraitement**")
    use_spacy = st.checkbox("Lemmatizer spaCy (FR) + POS (NOUN/PROPN/ADJ)", value=True,
                            help="Si décoché : mode léger (sans lemmatisation).")
    remove_accents = st.checkbox("Normaliser sans accents pour les correspondances", value=True)
    min_df_uni = st.slider("min_df unigrams (fréquence min)", 1, 20, 3)
    min_df_bi  = st.slider("min_df bigrams (fréquence min)", 1, 20, 2)
    st.markdown("---")

    st.markdown("**Étape 3 — Thèmes proposés**")
    method = st.radio("Méthode de regroupement", ["Auto-K (KMeans)", "HDBSCAN (sans K)"] if _has_hdbscan else ["Auto-K (KMeans)"])
    K_MIN = st.slider("K min", 2, 10, 3)
    K_MAX = st.slider("K max", K_MIN+1, 15, max(10, K_MIN+2))
    SVD_COMPONENTS = st.slider("Réduction SVD (dimension)", 10, 200, 50, step=10,
                               help="Stabilise les métriques de clustering.")
    MIN_CLUSTER_SIZE_RATIO = st.slider("Taille min cluster (%)", 1, 20, 2) / 100.0
    PENALTY_LAMBDA = st.slider("Pénalité complexité", 0.00, 0.50, 0.12, step=0.01)
    st.markdown("---")

    st.markdown("**Étape 4 — Sentiment (optionnel)**")
    do_sentiment = st.checkbox("Calculer le sentiment", value=False)
    st.caption("Modèle : cardiffnlp/twitter-xlm-roberta-base-sentiment (multilingue)")

st.info("**Parcours conseillé :** 1) Charge les données → 2) Prétraite → 3) Propose les champs → 4) Télécharge le template → 5) Réimporte le mapping édité → 6) Recodage 0/1 → 7) (option) Sentiment → 8) Exports.")

# -------------------- Fonctions utilitaires --------------------

def read_any_table(file, filename: str, sep_hint=None) -> pd.DataFrame:
    name = (filename or "").lower()
    if name.endswith((".xlsx",".xls")):
        return pd.read_excel(file)
    # CSV : détection encodage
    raw = file.getvalue() if hasattr(file,"getvalue") else file.read()
    # tentative sépa auto
    tried_encs = []
    for enc in ["utf-8","utf-8-sig","cp1252","iso-8859-1","latin1"]:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep_hint, engine="python")
            return df
        except UnicodeDecodeError:
            tried_encs.append(enc)
        except Exception:
            # recommence plus loin
            pass
    guess = chardet.detect(raw).get("encoding") or "cp1252"
    return pd.read_csv(io.BytesIO(raw), encoding=guess, sep=sep_hint, engine="python", errors="replace")

def normalize_text(t: str) -> str:
    if pd.isna(t): return ""
    t = str(t).replace("\xa0"," ").replace("’","'")
    t = re.sub(r"\s+"," ", t.strip())
    return t

def spacy_lemmas(doc):
    keep_pos = {"NOUN","PROPN","ADJ"}
    toks = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.like_num or tok.like_url or tok.like_email:
            continue
        if tok.pos_ in keep_pos:
            lemma = tok.lemma_.lower()
            lemma = re.sub(r"[^a-zàâçéèêëîïôùûüÿœ'-]", " ", lemma)
            lemma = re.sub(r"\s+"," ", lemma).strip()
            if lemma:
                toks.append(lemma)
    return toks

def sent_units_from_doc(doc):
    units = []
    for s in doc.sents:
        txt = s.text.strip()
        if not txt: 
            continue
        lem = spacy_lemmas(s) if doc is not None else []
        units.append({
            "text": txt,
            "lemmas_set": set(lem),
            "lemmas_str": " ".join(lem),
            "noaccent_lower": unidecode(txt.lower())
        })
    return units

def top_features_for_cluster(X, labels, vocab, k, topn=12):
    out = []
    for c in range(k):
        idx = np.where(labels==c)[0]
        if len(idx)==0:
            out.append([]); continue
        centroid = X[idx].mean(axis=0)
        arr = np.asarray(centroid).ravel()
        top_idx = arr.argsort()[::-1][:topn]
        out.append([vocab[i] for i in top_idx])
    return out

def choose_k_auto(X, K_MIN, K_MAX, SVD_COMPONENTS, MIN_CLUSTER_SIZE_RATIO, PENALTY_LAMBDA):
    n, p = X.shape
    svd_dim = max(2, min(SVD_COMPONENTS, n-1, p-1))
    svd = TruncatedSVD(n_components=svd_dim, random_state=42)
    Xr = svd.fit_transform(X)

    eval_rows = []
    for k in range(K_MIN, K_MAX+1):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(Xr)
        if len(set(labels)) < 2:
            continue

        sizes = np.bincount(labels, minlength=k)
        too_small = (sizes < max(2, int(MIN_CLUSTER_SIZE_RATIO*n))).sum()
        size_penalty = too_small / k

        sil = silhouette_score(Xr, labels, sample_size=min(10000, n))
        ch  = calinski_harabasz_score(Xr, labels)
        db  = davies_bouldin_score(Xr, labels)

        eval_rows.append({"k":k, "sil":sil, "ch":ch, "db":db, "size_penalty":size_penalty})

    eval_df = pd.DataFrame(eval_rows)
    if eval_df.empty:
        return 3, np.zeros((n,2))  # fallback

    def _norm(col, invert=False):
        x = eval_df[col].to_numpy(float)
        if invert: x = -x
        mn, mx = np.nanmin(x), np.nanmax(x)
        return np.ones_like(x)*0.5 if mx-mn<1e-9 else (x - mn) / (mx - mn + 1e-9)

    eval_df["sil_n"] = _norm("sil")
    eval_df["ch_n"]  = _norm("ch")
    eval_df["db_n"]  = _norm("db", invert=True)
    eval_df["complexity_penalty"] = PENALTY_LAMBDA * (eval_df["k"] - K_MIN) / max(1, (K_MAX - K_MIN))
    eval_df["final_score"] = (
        0.50 * eval_df["sil_n"]
      + 0.35 * eval_df["ch_n"]
      + 0.15 * eval_df["db_n"]
      - 0.30 * eval_df["size_penalty"]
      - eval_df["complexity_penalty"]
    )

    eval_df = eval_df.sort_values("k")
    best_idx = eval_df["final_score"].idxmax()
    best_k = int(eval_df.loc[best_idx, "k"])
    # anti-plafond
    if best_k == K_MAX:
        prev = eval_df[eval_df["k"]==(K_MAX-1)]
        if not prev.empty:
            if (eval_df.loc[best_idx,"final_score"] - prev["final_score"].iloc[0]) < 0.02:
                best_k = K_MAX-1

    return best_k, TruncatedSVD(n_components=svd_dim, random_state=42).fit_transform(X)

def mapping_template_from_proposed(proposed_df, take_top=6):
    rows = []
    for _, r in proposed_df.iterrows():
        label_sugg = str(r["label_auto"]).split(",")[0].strip().replace(" ", "_")[:30]
        triggers = " | ".join([t.strip() for t in str(r["top_terms"]).split(",")[:take_top]])
        rows.append({
            "champ_label": label_sugg.upper() or f"CHAMP_{int(r['champ_id'])}",
            "triggers": triggers,
            "match_mode": "lemma",
            "logic": "any"
        })
    return pd.DataFrame(rows, columns=["champ_label","triggers","match_mode","logic"])

def match_sentence(unit, triggers, match_mode, logic):
    if match_mode == "lemma":
        flags=[]
        for trig in triggers:
            toks = [t for t in trig.split() if t]
            if not toks: flags.append(False); continue
            if len(toks)==1:
                flags.append(toks[0] in unit["lemmas_set"])
            else:
                flags.append((" "+trig+" ") in (" "+unit["lemmas_str"]+" "))
        return any(flags) if logic=="any" else all(flags)
    elif match_mode == "contains":
        tx = unit["noaccent_lower"]
        checks = [unidecode(t.lower()) in tx for t in triggers]
        return any(checks) if logic=="any" else all(checks)
    elif match_mode == "regex":
        tx = unit["text"]
        try:
            checks = [bool(re.search(t, tx, flags=re.IGNORECASE)) for t in triggers]
            return any(checks) if logic=="any" else all(checks)
        except re.error:
            return False
    return False

def evidence_text_for_field(row, field_meta):
    triggers = [t.strip() for t in str(field_meta["triggers"]).split("|") if t.strip()]
    match_mode = str(field_meta.get("match_mode","lemma")).lower()
    logic = str(field_meta.get("logic","any")).lower()
    units = row["sent_units"]
    matches = [u["text"] for u in units if match_sentence(u, triggers, match_mode, logic)]
    if matches:
        return " ".join(matches)[:1000]
    return row["texte_norm"]

def sentiment_scores(pipe, text: str):
    if not text or pipe is None:
        return {"label":"", "pos":np.nan, "neu":np.nan, "neg":np.nan, "score":np.nan}
    res = pipe(text[:512])
    scores = {d["label"].lower(): float(d["score"]) for d in res[0]}
    pos = scores.get("positive", scores.get("label_2", 0.0))
    neu = scores.get("neutral",  scores.get("label_1", 0.0))
    neg = scores.get("negative", scores.get("label_0", 0.0))
    if pos >= max(neu, neg): label = "positive"
    elif neg >= max(pos, neu): label = "negative"
    else: label = "neutral"
    return {"label":label, "pos":pos, "neu":neu, "neg":neg, "score":pos - neg}

# -------------------- Corps principal --------------------

st.info("**Parcours conseillé :** 1) Charge les données → 2) Prétraite → 3) Propose les champs → 4) Télécharge le template → 5) Réimporte le mapping édité → 6) Recodage 0/1 → 7) (option) Sentiment → 8) Exports.")

with st.sidebar:
    pass

if "app_state" not in st.session_state:
    st.session_state.app_state = {}

# 1) Upload
up_container = st.container()
with up_container:
    if "uploaded" not in st.session_state:
        st.session_state.uploaded = None
    st.session_state.uploaded = up

if st.session_state.uploaded is None:
    st.info("➡️ Dépose un fichier pour commencer.")
    st.stop()

# Lecture
with st.spinner("Lecture du fichier..."):
    df = read_any_table(st.session_state.uploaded, st.session_state.uploaded.name, sep_hint=sep_guess if sep_guess not in (None,"\t") else None)
    if sep_guess == "\t":
        df = read_any_table(st.session_state.uploaded, st.session_state.uploaded.name, sep_hint="\t")

# Vérifs colonnes
cols = list(df.columns)
st.success(f"Fichier chargé ✅ ({len(df)} lignes, {len(cols)} colonnes)")
with st.expander("Aperçu des colonnes"):
    st.write(cols)

# Sélection des colonnes id / texte
c1, c2 = st.columns(2)
with c1:
    id_col = st.selectbox("Colonne identifiant", options=cols, index=0, key="idcol")
with c2:
    txt_col = st.selectbox("Colonne texte (verbatim)", options=cols, index=min(1, len(cols)-1), key="txtcol")

if sample_n and sample_n > 0:
    df = df.sample(n=min(sample_n, len(df)), random_state=42).reset_index(drop=True)
    st.caption(f"Échantillon utilisé : {len(df)} verbatims.")

df_work = df[[id_col, txt_col]].rename(columns={id_col:"id", txt_col:"texte"}).copy()
df_work["texte_norm"] = df_work["texte"].map(normalize_text)

# spaCy / lemmatization
if use_spacy:
    nlp = get_nlp()
    if nlp is None:
        st.warning("spaCy FR indisponible. Bascule en mode léger (sans lemmatisation).")
        use_spacy = False

if use_spacy:
    with st.spinner("Lemmatization spaCy en cours..."):
        docs = list(get_nlp().pipe(df_work["texte_norm"].tolist(), batch_size=512))
        df_work["lemmas"] = [spacy_lemmas(doc) for doc in docs]
        df_work["lemmas_str"] = df_work["lemmas"].apply(lambda xs: " ".join(xs))
        # unités de phrase pour sentiment par champ
        df_work["sent_units"] = [sent_units_from_doc(doc) for doc in docs]
else:
    # mode léger : tokenisation regex (moins précis)
    def light_tokens(t):
        t = t.lower()
        t = re.sub(r"[^a-zàâçéèêëîïôùûüÿœ'-]", " ", t)
        t = re.sub(r"\s+"," ", t).strip()
        return [w for w in t.split() if len(w)>2]
    df_work["lemmas"] = df_work["texte_norm"].apply(light_tokens)
    df_work["lemmas_str"] = df_work["lemmas"].apply(lambda xs: " ".join(xs))
    df_work["sent_units"] = df_work["texte_norm"].apply(lambda x: [{"text":x, "lemmas_set":set(light_tokens(x)), "lemmas_str":" ".join(light_tokens(x)), "noaccent_lower":unidecode(x.lower())}] )

# Fréquences
st.subheader("1) Mots & expressions fréquents")
c1, c2 = st.columns(2)
with c1:
    cv_uni = CountVectorizer(token_pattern=r"(?u)\b\w[\w'-]+\b", lowercase=True, min_df=min_df_uni)
    X_uni = cv_uni.fit_transform(df_work["lemmas_str"])
    terms = np.array(cv_uni.get_feature_names_out())
    freqs = np.asarray(X_uni.sum(axis=0)).ravel()
    top_words = pd.DataFrame({"term": terms, "freq": freqs}).sort_values("freq", ascending=False).head(50)
    st.write("Top mots (lemmes)")
    st.dataframe(top_words, use_container_width=True)

with c2:
    cv_bi = CountVectorizer(ngram_range=(2,2), token_pattern=r"(?u)\b\w[\w'-]+\b", lowercase=True, min_df=min_df_bi)
    X_bi = cv_bi.fit_transform(df_work["lemmas_str"])
    bigrams = np.array(cv_bi.get_feature_names_out())
    bfreqs = np.asarray(X_bi.sum(axis=0)).ravel()
    top_bigrams = pd.DataFrame({"bigram": bigrams, "freq": bfreqs}).sort_values("freq", ascending=False).head(50)
    st.write("Top expressions (bigrams)")
    st.dataframe(top_bigrams, use_container_width=True)

# Vectorisation TF-IDF (unigram+bigram)
tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w'-]+\b", lowercase=True, min_df=min_df_uni, ngram_range=(1,2))
X = tfidf.fit_transform(df_work["lemmas_str"])

# Propositions de champs (clusters)
st.subheader("2) Proposition de champs sémantiques")
if method == "Auto-K (KMeans)":
    with st.spinner("Sélection automatique de K..."):
        best_k, Xr = choose_k_auto(X, K_MIN, K_MAX, SVD_COMPONENTS, MIN_CLUSTER_SIZE_RATIO, PENALTY_LAMBDA)
        km_final = KMeans(n_clusters=best_k, n_init=20, random_state=42).fit(Xr)
        labels = km_final.labels_
        df_work["champ_auto_id"] = labels
        st.success(f"K retenu automatiquement : **{best_k}**")

    vocab = np.array(tfidf.get_feature_names_out())
    cluster_tops = top_features_for_cluster(X, labels, vocab, best_k, topn=12)
    proposed = []
    for c in range(best_k):
        tops = cluster_tops[c]
        label = ", ".join(tops[:5]) if tops else f"Cluster {c}"
        proposed.append({"champ_id": c, "label_auto": label, "top_terms": ", ".join(tops),
                         "taille": int((labels==c).sum())})
    proposed_df = pd.DataFrame(proposed).sort_values("taille", ascending=False)

else:  # HDBSCAN
    if not _has_hdbscan:
        st.error("HDBSCAN n'est pas disponible. Installe 'hdbscan' puis relance.")
        st.stop()
    with st.spinner("Clustering HDBSCAN..."):
        n = X.shape[0]
        svd = TruncatedSVD(n_components=min(SVD_COMPONENTS, X.shape[1]-1, max(2, X.shape[0]-1)), random_state=42)
        Xr = svd.fit_transform(X)
        min_cluster_size = max(5, int(MIN_CLUSTER_SIZE_RATIO * n))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=None, metric='euclidean')
        labels = clusterer.fit_predict(Xr)
        df_work["champ_auto_id"] = labels
        valid_clusters = sorted([c for c in set(labels) if c != -1])

    vocab = np.array(tfidf.get_feature_names_out())
    proposed = []
    for c in valid_clusters:
        idx = np.where(labels==c)[0]
        centroid = X[idx].mean(axis=0)
        arr = np.asarray(centroid).ravel()
        top_idx = arr.argsort()[::-1][:12]
        terms = [vocab[i] for i in top_idx]
        label = ", ".join(terms[:5]) if terms else f"Cluster {c}"
        proposed.append({"champ_id": int(c), "label_auto": label, "top_terms": ", ".join(terms),
                         "taille": int((labels==c).sum())})
    proposed_df = pd.DataFrame(proposed).sort_values("taille", ascending=False)

st.write("Champs proposés (auto)")
st.dataframe(proposed_df, use_container_width=True)

# Téléchargement proposed_fields.csv
csv_bytes = proposed_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Télécharger proposed_fields.csv", data=csv_bytes, file_name="proposed_fields.csv", mime="text/csv")

# Génère un template de mapping à partir de la proposition
st.subheader("3) Préparer / importer le mapping (triggers → champ)")
templ = mapping_template_from_proposed(proposed_df)
st.caption("Tu peux partir de ce template, le modifier dans Excel, puis le réimporter ci-dessous.")
st.dataframe(templ.head(10), use_container_width=True)
templ_bytes = templ.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Télécharger semantic_map_template.csv", data=templ_bytes, file_name="semantic_map_template.csv", mime="text/csv")

map_file = st.file_uploader("Réimporte ton mapping édité (CSV avec colonnes champ_label,triggers,match_mode,logic)", type=["csv"])

if map_file is not None:
    user_map = pd.read_csv(map_file).dropna(subset=["champ_label","triggers"]).copy()
    # normalisation des labels de colonnes
    user_map.columns = [c.strip().lower() for c in user_map.columns]
    required = {"champ_label","triggers","match_mode","logic"}
    if not required.issubset(set(user_map.columns)):
        st.error(f"Le CSV doit contenir les colonnes : {sorted(required)}.")
        st.stop()

    st.success(f"Mapping chargé ✅ ({len(user_map)} lignes)")
    st.dataframe(user_map.head(20), use_container_width=True)

    # 4) Recodage 0/1
    st.subheader("4) Recodage 0/1 par champ")
    df_work["texte_lower"] = df_work["texte_norm"].str.lower()
    df_work["texte_noaccent"] = df_work["texte_lower"].apply(unidecode)
    df_work["lemmas_set"] = df_work["lemmas"].apply(set)

    def _norm_lab(s): 
        return re.sub(r"[^A-Za-z0-9_]", "_", str(s).strip().upper())
    map_norm = { _norm_lab(r["champ_label"]): r for _, r in user_map.iterrows() }

    def match_record_row(row, triggers, match_mode, logic):
        flags = []
        if match_mode == "lemma":
            for trig in triggers:
                toks = [t for t in trig.strip().split() if t]
                if not toks:
                    flags.append(False); continue
                if len(toks)==1:
                    flags.append(toks[0] in row["lemmas_set"])
                else:
                    flags.append((" "+trig+" ") in (" "+row["lemmas_str"]+" "))
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
        return any(flags) if logic=="any" else all(flags)

    coded_cols = []
    for _, r in user_map.iterrows():
        label = _norm_lab(r["champ_label"])
        triggers = [t.strip() for t in str(r["triggers"]).split("|") if t.strip()]
        match_mode = str(r.get("match_mode","lemma")).strip().lower()
        logic = str(r.get("logic","any")).strip().lower()
        df_work[label] = df_work.apply(lambda row: int(match_record_row(row, triggers, match_mode, logic)), axis=1)
        coded_cols.append(label)

    # Synthèse n / %
    n = len(df_work)
    summary = []
    for c in coded_cols:
        k = int(df_work[c].sum())
        summary.append({"champ_label": c, "n": k, "pct": round(100*k/n,2)})
    summary_df = pd.DataFrame(summary).sort_values(["n","champ_label"], ascending=[False, True])

    c1, c2 = st.columns(2)
    with c1:
        st.write("Synthèse (n / %)")
        st.dataframe(summary_df, use_container_width=True)
    with c2:
        st.write("Aperçu des colonnes recodées")
        st.dataframe(df_work[["id","texte"] + coded_cols].head(20), use_container_width=True)

    # 5) Sentiment (optionnel)
    if do_sentiment:
        st.subheader("5) Sentiment (global + par champ)")
        pipe = get_sentiment_pipeline()
        if pipe is not None:
            with st.spinner("Calcul du sentiment global..."):
                glob = df_work["texte_norm"].apply(lambda x: sentiment_scores(pipe, x))
                df_work["sent_label"] = glob.apply(lambda x: x["label"])
                df_work["sent_pos"]   = glob.apply(lambda x: x["pos"])
                df_work["sent_neu"]   = glob.apply(lambda x: x["neu"])
                df_work["sent_neg"]   = glob.apply(lambda x: x["neg"])
                df_work["sent_score"] = glob.apply(lambda x: x["score"])

            with st.spinner("Calcul du sentiment par champ (sur phrases-évidence)..."):
                for c in coded_cols:
                    def _field_sent(row):
                        if int(row[c]) != 1:
                            return {"label":"", "pos":np.nan, "neu":np.nan, "neg":np.nan, "score":np.nan}
                        meta = map_norm.get(c)
                        txt = evidence_text_for_field(row, meta) if meta is not None else row["texte_norm"]
                        return sentiment_scores(pipe, txt)
                    tmp = df_work.apply(_field_sent, axis=1)
                    df_work[f"{c}_SENT_LABEL"] = tmp.apply(lambda x: x["label"])
                    df_work[f"{c}_SENT_POS"]   = tmp.apply(lambda x: x["pos"])
                    df_work[f"{c}_SENT_NEU"]   = tmp.apply(lambda x: x["neu"])
                    df_work[f"{c}_SENT_NEG"]   = tmp.apply(lambda x: x["neg"])
                    df_work[f"{c}_SENT_SCORE"] = tmp.apply(lambda x: x["score"])

            # Agrégats par champ
            aggs = []
            for c in coded_cols:
                sub = df_work.loc[df_work[c]==1, [f"{c}_SENT_LABEL", f"{c}_SENT_SCORE"]].copy()
                if len(sub)==0:
                    aggs.append({"champ_label": c, "n":0, "mean_score": np.nan, "%pos":0.0, "%neu":0.0, "%neg":0.0})
                    continue
                nsub = len(sub)
                pct_pos = round(100* (sub[f"{c}_SENT_LABEL"]=="positive").mean(), 2)
                pct_neu = round(100* (sub[f"{c}_SENT_LABEL"]=="neutral").mean(),  2)
                pct_neg = round(100* (sub[f"{c}_SENT_LABEL"]=="negative").mean(), 2)
                mean_sc = round(float(sub[f"{c}_SENT_SCORE"].mean()), 4)
                aggs.append({"champ_label": c, "n": nsub, "mean_score": mean_sc, "%pos": pct_pos, "%neu": pct_neu, "%neg": pct_neg})
            sent_summary = pd.DataFrame(aggs).sort_values(["n","mean_score"], ascending=[False, False])
            st.write("Résumé sentiment par champ")
            st.dataframe(sent_summary, use_container_width=True)

    # 6) Exports
    st.subheader("6) Exports")
    export_cols = ["id","texte","texte_norm","lemmas_str"] + coded_cols
    if do_sentiment and "sent_label" in df_work.columns:
        export_cols = ["id","texte","texte_norm","lemmas_str","sent_label","sent_score","sent_pos","sent_neu","sent_neg"] + coded_cols

    verb_csv = df_work[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ verbatims_codes.csv", data=verb_csv, file_name="verbatims_codes.csv", mime="text/csv")

    synth = df_work[coded_cols].sum().reset_index()
    synth.columns = ["champ_label","n"]
    synth["pct"] = (100 * synth["n"] / len(df_work)).round(2)
    synth_csv = synth.sort_values(["n","champ_label"], ascending=[False, True]).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ semantic_summary.csv", data=synth_csv, file_name="semantic_summary.csv", mime="text/csv")

else:
    st.info("💡 Télécharge le template, édite-le, puis réimporte-le ici pour lancer le recodage 0/1.")
