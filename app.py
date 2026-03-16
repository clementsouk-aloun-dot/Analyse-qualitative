from __future__ import annotations

from io import BytesIO
from typing import Dict, List

import pandas as pd
import streamlit as st
import spacy

from src.clustering import (
    build_clusters_dataframe,
    build_tfidf_matrix,
    choose_k_auto,
    compute_frequencies,
    fit_manual_k,
)
from src.exporter import workbook_bytes
from src.io_excel import list_sheet_names, read_excel_sheet
from src.nlp_pipeline import discover_available_spacy_models, preprocess_dataframe
from src.theming import (
    assign_themes_from_clusters,
    build_binary_coding,
    build_default_themes_df,
    clean_themes_df,
)

st.set_page_config(page_title="Analyse qualitative - Streamlit", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner=False)
def load_spacy_model(model_name: str):
    nlp = spacy.load(model_name, disable=["ner"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


@st.cache_data(show_spinner=False)
def cached_sheet_names(file_bytes: bytes) -> List[str]:
    return list_sheet_names(file_bytes)


@st.cache_data(show_spinner=False)
def cached_read_sheet(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return read_excel_sheet(file_bytes, sheet_name)


def run_analysis(
    df: pd.DataFrame,
    *,
    id_col: str,
    text_col: str,
    model_name: str,
    keep_pos: List[str],
    strip_accents: bool,
    remove_numbers: bool,
    min_df_uni: int,
    min_df_bi: int,
    min_df_tfidf: int,
    top_terms: int,
    examples_per_cluster: int,
    cluster_mode: str,
    auto_k_method: str,
    manual_k: int,
):
    nlp = load_spacy_model(model_name)
    processed_df = preprocess_dataframe(
        df,
        id_col=id_col,
        text_col=text_col,
        nlp=nlp,
        keep_pos=keep_pos,
        strip_accents=strip_accents,
        remove_numbers=remove_numbers,
    )

    non_empty = processed_df[processed_df["lemmas_str"].str.strip() != ""].copy()
    if len(non_empty) < 3:
        raise ValueError("Il faut au moins 3 verbatims non vides après prétraitement.")

    top_words_df, top_bigrams_df = compute_frequencies(
        non_empty["lemmas_str"],
        min_df_uni=min_df_uni,
        min_df_bi=min_df_bi,
    )

    try:
        x, vocab, _ = build_tfidf_matrix(non_empty["lemmas_str"], min_df_tfidf=min_df_tfidf)
    except ValueError as exc:
        raise ValueError("Impossible de construire la matrice TF-IDF. Réduis `min_df TF-IDF` ou vérifie le contenu textuel.") from exc

    if cluster_mode == "auto":
        best_k, labels, _model, eval_df = choose_k_auto(
            x,
            k_min=3,
            k_max=10,
            method=auto_k_method,
        )
    else:
        if len(non_empty) <= manual_k:
            raise ValueError("Le nombre de clusters manuel doit être strictement inférieur au nombre de verbatims analysés.")
        labels, _model = fit_manual_k(x, k=manual_k)
        best_k = manual_k
        eval_df = pd.DataFrame()

    clusters_df, examples_df = build_clusters_dataframe(
        non_empty,
        labels,
        x,
        vocab,
        topn=top_terms,
        examples_per_cluster=examples_per_cluster,
    )

    non_empty = non_empty.copy()
    non_empty["cluster_auto_id"] = labels
    themes_df = build_default_themes_df(clusters_df)

    verbatims_thematiques = assign_themes_from_clusters(non_empty, themes_df)
    coded_df, summary_df, coded_cols = build_binary_coding(verbatims_thematiques, themes_df)

    return {
        "processed_df": processed_df,
        "non_empty_df": non_empty,
        "top_words_df": top_words_df,
        "top_bigrams_df": top_bigrams_df,
        "clusters_df": clusters_df,
        "examples_df": examples_df,
        "themes_df": themes_df,
        "verbatims_thematiques": verbatims_thematiques,
        "coded_df": coded_df,
        "summary_df": summary_df,
        "coded_cols": coded_cols,
        "eval_df": eval_df,
        "best_k": best_k,
    }


def refresh_theming():
    analysis = st.session_state.get("analysis")
    edited_themes = st.session_state.get("themes_editor_df")
    if analysis is None or edited_themes is None:
        return

    cleaned = clean_themes_df(edited_themes)
    non_empty = analysis["non_empty_df"].copy()
    verbatims_thematiques = assign_themes_from_clusters(non_empty, cleaned)
    coded_df, summary_df, coded_cols = build_binary_coding(verbatims_thematiques, cleaned)

    analysis["themes_df"] = cleaned
    analysis["verbatims_thematiques"] = verbatims_thematiques
    analysis["coded_df"] = coded_df
    analysis["summary_df"] = summary_df
    analysis["coded_cols"] = coded_cols
    st.session_state["analysis"] = analysis


st.title("🧠 Analyse qualitative assistée")
st.caption(
    "Upload d'un fichier Excel, clustering automatique ou manuel, édition des thèmes/clusters et export Excel multi-onglets."
)

with st.sidebar:
    st.header("1) Fichier")
    uploaded_file = st.file_uploader("Charger un fichier Excel", type=["xlsx", "xls"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        st.session_state["file_bytes"] = file_bytes
        st.session_state["file_name"] = uploaded_file.name

    file_bytes = st.session_state.get("file_bytes")

    if file_bytes:
        sheet_names = cached_sheet_names(file_bytes)
        selected_sheet = st.selectbox("Feuille", sheet_names, key="selected_sheet")
        df_input = cached_read_sheet(file_bytes, selected_sheet)
        st.session_state["input_df"] = df_input

        st.header("2) Colonnes")
        if len(df_input.columns) >= 2:
            default_id = df_input.columns[0]
            default_text = df_input.columns[1]
        else:
            default_id = df_input.columns[0]
            default_text = df_input.columns[0]

        id_col = st.selectbox("Colonne identifiant", df_input.columns.tolist(), index=df_input.columns.tolist().index(default_id))
        text_col = st.selectbox("Colonne texte", df_input.columns.tolist(), index=df_input.columns.tolist().index(default_text))

        st.header("3) NLP")
        available_models = discover_available_spacy_models()
        if not available_models:
            st.error("Aucun modèle spaCy français n'est installé. Vérifie ton requirements.txt.")
            available_models = ["fr_core_news_sm"]
        model_name = st.selectbox("Modèle spaCy", available_models)
        keep_pos = st.multiselect(
            "Types grammaticaux conservés",
            ["NOUN", "PROPN", "ADJ", "VERB"],
            default=["NOUN", "PROPN", "ADJ"],
        )
        strip_accents = st.checkbox("Retirer les accents dans le texte normalisé", value=False)
        remove_numbers = st.checkbox("Retirer les chiffres", value=False)

        st.header("4) Clustering")
        cluster_mode = st.radio("Mode de choix de K", ["auto", "manuel"], horizontal=True)
        auto_k_method = st.selectbox(
            "Méthode auto",
            ["composite", "silhouette"],
            disabled=(cluster_mode != "auto"),
        )
        manual_k = st.slider("Nombre de clusters (manuel)", 3, 10, 4, disabled=(cluster_mode != "manuel"))

        with st.expander("Paramètres avancés"):
            min_df_uni = st.slider("min_df mots", 1, 10, 3)
            min_df_bi = st.slider("min_df bigrams", 1, 10, 2)
            min_df_tfidf = st.slider("min_df TF-IDF", 1, 10, 3)
            top_terms = st.slider("Tags par cluster", 3, 15, 8)
            examples_per_cluster = st.slider("Exemples par cluster", 2, 10, 5)

        run_clicked = st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)
        if run_clicked:
            try:
                with st.spinner("Analyse en cours..."):
                    analysis = run_analysis(
                        df_input,
                        id_col=id_col,
                        text_col=text_col,
                        model_name=model_name,
                        keep_pos=keep_pos,
                        strip_accents=strip_accents,
                        remove_numbers=remove_numbers,
                        min_df_uni=min_df_uni,
                        min_df_bi=min_df_bi,
                        min_df_tfidf=min_df_tfidf,
                        top_terms=top_terms,
                        examples_per_cluster=examples_per_cluster,
                        cluster_mode=cluster_mode,
                        auto_k_method=auto_k_method,
                        manual_k=manual_k,
                    )
                st.session_state["analysis"] = analysis
                st.session_state["analysis_config"] = {
                    "id_col": id_col,
                    "text_col": text_col,
                    "model_name": model_name,
                    "keep_pos": keep_pos,
                    "strip_accents": strip_accents,
                    "remove_numbers": remove_numbers,
                    "cluster_mode": cluster_mode,
                    "auto_k_method": auto_k_method,
                    "manual_k": manual_k,
                }
                st.session_state["themes_editor_df"] = analysis["themes_df"].copy()
                st.success("Analyse terminée.")
            except Exception as exc:
                st.exception(exc)

if "input_df" not in st.session_state:
    st.info("Charge d'abord un fichier Excel dans la barre latérale.")
    st.stop()

input_df = st.session_state["input_df"]
analysis = st.session_state.get("analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Aperçu fichier",
        "Résultats auto",
        "Édition thèmes",
        "Sorties finales",
        "Export",
    ]
)

with tab1:
    st.subheader("Aperçu du fichier")
    col1, col2, col3 = st.columns(3)
    col1.metric("Lignes", len(input_df))
    col2.metric("Colonnes", len(input_df.columns))
    col3.metric("Feuille", st.session_state.get("selected_sheet", "-"))
    st.dataframe(input_df.head(20), use_container_width=True)

with tab2:
    if analysis is None:
        st.info("Lance l'analyse depuis la barre latérale pour voir les résultats automatiques.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Verbatims analysés", len(analysis["non_empty_df"]))
        col2.metric("K retenu", analysis["best_k"])
        col3.metric("Textes vides / exclus", int(analysis["processed_df"]["texte_vide"].sum()))

        if not analysis["eval_df"].empty:
            st.subheader("Évaluation de K")
            st.dataframe(analysis["eval_df"], use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top mots")
            st.dataframe(analysis["top_words_df"].head(30), use_container_width=True)
        with c2:
            st.subheader("Top bigrams")
            st.dataframe(analysis["top_bigrams_df"].head(30), use_container_width=True)

        st.subheader("Clusters automatiques")
        st.dataframe(analysis["clusters_df"], use_container_width=True)

        st.subheader("Exemples par cluster")
        examples_df = analysis["examples_df"]
        if examples_df.empty:
            st.info("Pas d'exemples disponibles.")
        else:
            for cluster_id, sub in examples_df.groupby("cluster_auto_id"):
                with st.expander(f"Cluster {cluster_id}", expanded=False):
                    st.dataframe(sub, use_container_width=True)

with tab3:
    if analysis is None:
        st.info("Lance d'abord l'analyse.")
    else:
        st.subheader("Éditer les thèmes finaux")
        st.write(
            "Tu peux fusionner plusieurs clusters dans un même thème via `source_clusters` (ex. `0|2|3`), "
            "renommer les thèmes et modifier les tags associés."
        )

        edited_themes = st.data_editor(
            st.session_state.get("themes_editor_df", analysis["themes_df"]),
            use_container_width=True,
            num_rows="dynamic",
            key="themes_data_editor",
            column_config={
                "active": st.column_config.CheckboxColumn("Actif"),
                "match_mode": st.column_config.SelectboxColumn(
                    "Mode de match",
                    options=["lemma", "contains", "regex"],
                ),
                "logic": st.column_config.SelectboxColumn(
                    "Logique",
                    options=["any", "all"],
                ),
            },
        )
        st.session_state["themes_editor_df"] = edited_themes.copy()

        if st.button("✅ Appliquer les thèmes édités", type="primary"):
            refresh_theming()
            st.success("Thèmes mis à jour.")

        st.subheader("Prévisualisation des affectations")
        preview_cols = [
            "__source_id",
            "__source_text",
            "cluster_auto_id",
            "theme_final_id",
            "theme_final_label",
        ]
        st.dataframe(
            analysis.get("verbatims_thematiques", pd.DataFrame())[preview_cols].head(100),
            use_container_width=True,
            height=420,
        )

with tab4:
    if analysis is None:
        st.info("Lance d'abord l'analyse.")
    else:
        st.subheader("Verbatims thématisés")
        theme_cols = [
            "__source_id",
            "__source_text",
            "cluster_auto_id",
            "theme_final_id",
            "theme_final_label",
        ]
        st.dataframe(analysis["verbatims_thematiques"][theme_cols], use_container_width=True, height=420)

        st.subheader("Codage binaire par thème")
        coded_preview_cols = [
            "__source_id",
            "__source_text",
            "theme_final_label",
        ] + analysis["coded_cols"]
        st.dataframe(analysis["coded_df"][coded_preview_cols].head(200), use_container_width=True, height=420)

        st.subheader("Synthèse")
        st.dataframe(analysis["summary_df"], use_container_width=True)

with tab5:
    if analysis is None:
        st.info("Lance d'abord l'analyse.")
    else:
        export_sheets = {
            "input_clean": analysis["processed_df"],
            "top_words": analysis["top_words_df"],
            "top_bigrams": analysis["top_bigrams_df"],
            "k_evaluation": analysis["eval_df"],
            "clusters_auto": analysis["clusters_df"],
            "cluster_examples": analysis["examples_df"],
            "themes_final": analysis["themes_df"],
            "verbatims_thematiques": analysis["verbatims_thematiques"],
            "verbatims_codes": analysis["coded_df"],
            "semantic_summary": analysis["summary_df"],
        }
        export_bytes = workbook_bytes(export_sheets)
        base_name = st.session_state.get("file_name", "analyse_quali")
        base_name = base_name.rsplit(".", 1)[0]
        st.download_button(
            "📥 Télécharger le fichier Excel de sortie",
            data=export_bytes,
            file_name=f"{base_name}_analyse_qualitative.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.write("Le fichier exporté contient tous les onglets utiles pour retravailler l'analyse dans Excel.")
