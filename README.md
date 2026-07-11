# AutoDebug AI

Predicts which GitHub pull requests are likely to introduce bugs, so reviewers know which ones need extra attention.

Work in progress — modeling complete, API and dashboard next.

## How it works

1. **Collect data** — pulls 500 merged PRs and 2,000 commits from the `psf/requests` repo using the GitHub API
2. **Label them** — a PR is marked "buggy" if a bug-fix commit later changed the same files within 90 days of the merge
3. **Train a model** — predicts the bug risk of new PRs from features like size, files changed, and title text

Dataset: **201 buggy / 299 safe PRs (40% buggy)**

## Results

Final feature set: 10 numeric features + PR title embeddings (`all-MiniLM-L6-v2`, compressed 384 → 30 dims with PCA).

| Model | AUC |
|---|---|
| **Random Forest** | **0.710** |
| Logistic Regression | 0.662 |
| XGBoost | 0.658 |
| LightGBM | 0.631 |

The winning Random Forest is saved to `models/best_model.pkl`.

Findings from the experiments along the way:

- **Author-history features didn't help** — 45% of PRs come from first-time contributors, so most authors have no usable history.
- **Full 384-dim title embeddings overfit** (394 features vs 400 training rows); PCA compression to 30 dims fixed it and produced the best result.
- **Gradient boosting underperformed** — expected with only 400 training rows, where lower-variance models generalize better.

## What drives risk?

![SHAP summary](output.png)

PR size dominates: more changed files and added lines push risk up. Title embeddings carry surprising signal — the strongest single feature is a title dimension, meaning how a PR is described correlates with bug risk.

## Project structure

```
AutoDebugAI/
├── data/                        # raw JSON + labeled CSVs (gitignored)
├── models/                      # best_model.pkl + pca.pkl
├── notebooks/                   # experiments: baseline → features → NLP → shootout → SHAP
├── src/
│   ├── collect.py               # fetch merged PRs
│   ├── collect_commits.py       # fetch commits
│   ├── label.py                 # create labeled dataset
│   ├── features.py              # build feature matrix (numeric + PCA title embeddings)
│   ├── train.py                 # train Random Forest, save model
│   └── predict.py               # score a PR: risk + top reasons
├── output.png                   # SHAP summary plot
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
```

Add a GitHub token to a `.env` file:

```
GITHUB_TOKEN=your_token_here
```

Build the dataset:

```bash
python src/collect.py
python src/collect_commits.py
python src/label.py
```

Train and predict:

```bash
python src/train.py      # trains the model (AUC ~0.71), saves models/
python src/predict.py    # scores a demo PR
```

Example prediction from `predict.py`:

```json
{"risk_score": 0.635, "reasons": ["num_files raises risk", "additions raises risk", "changed_files raises risk"]}
```

## API

```bash
uvicorn src.api:app --reload
```

Open http://127.0.0.1:8000/docs for interactive testing (local).

`POST /predict` takes PR features and returns:

```json
{"risk_score": 0.635, "risk_level": "medium", "reasons": ["num_files raises risk", "additions raises risk", "changed_files raises risk"]}
```


## Tech

Python · PyGithub · pandas · scikit-learn · sentence-transformers · XGBoost · LightGBM · SHAP — FastAPI and Streamlit coming next.
