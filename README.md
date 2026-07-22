# AutoDebug AI

Predicts which GitHub pull requests are likely to introduce bugs, so reviewers know which ones need extra attention — a "credit score for code changes."

A full end-to-end ML system: data collection, automated labeling, model training, explainability, a REST API, and an interactive dashboard.

**Live demo:** https://autodebugai.streamlit.app/

## How it works

1. **Collect data** — pulls 500 merged PRs and 2,000 commits from the `psf/requests` repo using the GitHub API
2. **Label them** — a PR is marked "buggy" if a bug-fix commit later changed the same files within 90 days of the merge
3. **Train a model** — predicts the bug risk of new PRs from features like size, files changed, and title text
4. **Serve it** — a FastAPI service and a Streamlit dashboard score any PR and explain why

Dataset: **201 buggy / 299 safe PRs (40% buggy)**

## Dashboard

![Dashboard](docs/dashboard.png)

```bash
streamlit run app/dashboard.py
```

Five pages, navigated from a collapsible sidebar:

| Page | What it does |
|---|---|
| **Overview** | KPI row, risk distribution histogram, risk trend against the high-risk threshold, and a "needs review first" table |
| **Score a PR** | Enter any `psf/requests` PR number, fetch it live, and see its risk score with SHAP feature contributions |
| **Review queue** | Every scored PR sorted riskiest-first, filterable by risk level and title search, exportable to CSV |
| **Model** | Model comparison table, ROC AUC dial, feature groups, and a plain-language limitations section |
| **About** | How the pipeline works and what it's built with |


## Results

Final feature set: 10 numeric features + PR title embeddings (`all-MiniLM-L6-v2`, compressed 384 → 30 dims with PCA).

| Model | AUC |
|---|---|
| **Random Forest** | **0.710** |
| Logistic Regression | 0.662 |
| XGBoost | 0.658 |
| LightGBM | 0.631 |

The winning Random Forest is saved to `models/best_model.pkl`.


## What drives risk?

![SHAP summary](docs/output.png)

PR size dominates: more changed files and added lines push risk up. Title embeddings carry surprising signal — the strongest single feature is a title dimension, meaning how a PR is described correlates with bug risk.

## API

```bash
uvicorn src.api:app --reload
```

Open http://127.0.0.1:8000/docs for interactive testing (local)

- `POST /predict` — score a PR from provided features
- `GET /predict/{pr_number}` — fetch a live PR from `psf/requests` and score it

Example — live scoring of a real PR:

```json
GET /predict/7555
{"risk_score": 0.665, "risk_level": "medium", "reasons": ["num_files raises risk", "PR title content raises risk", "additions raises risk"]}
```

## Run it yourself

```bash
pip install -r requirements.txt
```

Add a GitHub token to a `.env` file:

```
GITHUB_TOKEN=your_token_here
```

Build the dataset, then train:

```bash
python -m src.data.collect
python -m src.data.collect_commits
python -m src.data.label
python -m src.model.train      # trains the model (AUC ~0.71), saves to models/
```

Or run the API with Docker:

```bash
docker build -t autodebug-ai .
docker run -p 8000:8000 --env-file .env autodebug-ai
```

## Project structure

```
AutoDebugAI/
├── app/
│   └── dashboard.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── dashboard.png
│   └── output.png
├── models/
│   ├── best_model.pkl
│   └── pca.pkl
├── notebooks/
├── src/
│   ├── data/
│   │   ├── collect.py
│   │   ├── collect_commits.py
│   │   ├── label.py
│   │   └── fetch_pr.py
│   ├── model/
│   │   ├── features.py
│   │   ├── train.py
│   │   └── predict.py
│   └── api.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Tech

Python · PyGithub · pandas · scikit-learn · sentence-transformers · XGBoost · LightGBM · SHAP · FastAPI · Streamlit · Docker
