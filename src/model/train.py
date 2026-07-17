import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.model.features import build_features


def main():
    df = pd.read_csv("data/processed/labeled_prs_v2.csv")
    X, names = build_features(df, fit_pca=True)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"AUC: {auc:.3f}")   # expect ~0.71

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/best_model.pkl")
    print("Saved models/best_model.pkl and models/pca.pkl")


if __name__ == "__main__":
    main()