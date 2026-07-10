import numpy as np 
import pandas as pd 
import joblib
from pathlib import Path 
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

NUMERIC = ["additions", "deletions", "changed_files", "commits",
           "comments", "review_comments", "num_files",
           "author_past_prs", "author_past_bug_rate", "is_first_pr"]

MODELS_DIR = Path("models")
_st_model = None  # loaded lazily, once

def get_st_model():
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def build_features(df, fit_pca=True):
    """DataFrame with numeric cols + 'title' -> (X, feature_names).
    fit_pca=True fits and saves PCA (training); False loads the saved PCA (serving).
    """
    embeddings = get_st_model().encode(df["title"].fillna("").tolist())

    if fit_pca:
        pca = PCA(n_components=30, random_state=42)
        emb_small = pca.fit_transform(embeddings)
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(pca, MODELS_DIR / "pca.pkl")
    else:
        pca = joblib.load(MODELS_DIR / "pca.pkl")
        emb_small = pca.transform(embeddings)

    X = np.hstack([df[NUMERIC].values, emb_small])
    feature_names = NUMERIC + [f"title_emb_{i}" for i in range(30)]
    return X, feature_names