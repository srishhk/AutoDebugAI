import numpy as np
import pandas as pd
import joblib
import shap

from src.model.features import build_features, NUMERIC

_model = None
_explainer = None


def get_model():
    global _model, _explainer
    if _model is None:
        _model = joblib.load("models/best_model.pkl")
        _explainer = shap.TreeExplainer(_model)
    return _model, _explainer


def score_pr(pr_dict):
    """pr_dict: numeric features + 'title'. Returns risk, reasons, and contributions."""
    model, explainer = get_model()
    df = pd.DataFrame([pr_dict])
    X, names = build_features(df, fit_pca=False)

    risk = float(model.predict_proba(X)[0, 1])

    sv = explainer.shap_values(X)
    sv = sv[:, :, 1] if np.array(sv).ndim == 3 else sv
    contrib = pd.Series(sv[0], index=names)

    # group all title embedding dims into one "PR title content" contribution
    title_total = contrib[[n for n in names if n.startswith("title_emb")]].sum()
    grouped = contrib[[n for n in names if not n.startswith("title_emb")]].copy()
    grouped["PR title content"] = title_total

    top = grouped.abs().sort_values(ascending=False).head(5)

    contributions = [
        {"feature": f, "value": round(float(grouped[f]) * 100, 1),
         "direction": "raises" if grouped[f] > 0 else "lowers"}
        for f in top.index
    ]
    reasons = [f"{c['feature']} {c['direction']} risk" for c in contributions[:3]]

    return {"risk_score": round(risk, 3), "reasons": reasons, "contributions": contributions}


if __name__ == "__main__":
    demo = {
        "additions": 850, "deletions": 40, "changed_files": 12,
        "commits": 6, "comments": 0, "review_comments": 0, "num_files": 12,
        "author_past_prs": 0, "author_past_bug_rate": 0, "is_first_pr": 1,
        "title": "quick hotfix for auth",
    }
    print(score_pr(demo))