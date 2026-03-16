# App Streamlit - Analyse qualitative

Cette application reprend la logique de ton notebook Colab et la transforme en interface Streamlit exploitable depuis GitHub.

## Fonctionnalités

- upload d'un fichier Excel (`.xlsx` / `.xls`)
- choix de la feuille
- choix de la colonne identifiant et de la colonne texte
- choix du modèle spaCy installé
- choix des POS conservés (`NOUN`, `PROPN`, `ADJ`, `VERB`)
- clustering automatique (`K` entre 3 et 10) ou manuel
- édition des thèmes finaux : fusion de clusters via `source_clusters`, renommage, édition des tags, `match_mode`, `logic`
- export Excel multi-onglets

## Structure

- `app.py` : interface Streamlit
- `src/io_excel.py` : lecture Excel
- `src/nlp_pipeline.py` : normalisation et lemmatisation
- `src/clustering.py` : TF-IDF, fréquences, choix de K, clustering
- `src/theming.py` : édition des thèmes et codage binaire
- `src/exporter.py` : export Excel

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Déployer avec GitHub + Streamlit Cloud

1. Créer un dépôt GitHub, par exemple `quali-streamlit`
2. Pousser le contenu de ce dossier dans le dépôt
3. Sur Streamlit Community Cloud, créer une nouvelle app
4. Sélectionner le dépôt GitHub, la branche `main` et le fichier `app.py`
5. Déployer

## Si tu veux proposer d'autres modèles spaCy

Par défaut, le dépôt installe `fr_core_news_sm`. Si tu veux aussi proposer `fr_core_news_md` ou `fr_core_news_lg`, ajoute leurs wheels dans `requirements.txt`. L'app n'affiche que les modèles réellement installés.
