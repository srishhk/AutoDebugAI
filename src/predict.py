import numpy as np
import pandas as pd
import joblib
import shap

from features import build_features, NUMERIC

_model = None
_explainer = None


def get_model():
    global _model, _explainer
    if _model is None:
        _model = joblib.load("models/best_model.pkl")
        _explainer = shap.TreeExplainer(_model)
    return _model, _explainer


def score_pr(pr_dict):
    """pr_dict: numeric features + 'title'. Returns risk + top 3 reasons."""
    model, explainer = get_model()
    df = pd.DataFrame([pr_dict])
    X, names = build_features(df, fit_pca=False)

    risk = float(model.predict_proba(X)[0, 1])

    sv = explainer.shap_values(X)
    sv = sv[:, :, 1] if np.array(sv).ndim == 3 else sv
    contrib = pd.Series(sv[0], index=names)
    top3 = contrib.abs().sort_values(ascending=False).head(3)

    reasons = []
    for feat in top3.index:
        label = "PR title content" if feat.startswith("title_emb") else feat
        direction = "raises" if contrib[feat] > 0 else "lowers"
        reasons.append(f"{label} {direction} risk")

    return {"risk_score": round(risk, 3), "reasons": reasons}


if __name__ == "__main__":
    demo = {
        "additions": 850, "deletions": 40, "changed_files": 12,
        "commits": 6, "comments": 0, "review_comments": 0, "num_files": 12,
        "author_past_prs": 0, "author_past_bug_rate": 0, "is_first_pr": 1,
        "title": "quick hotfix for auth",
    }
    print(score_pr(demo))