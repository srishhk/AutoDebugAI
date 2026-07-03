# AutoDebug AI

Predicts which GitHub pull requests are likely to introduce bugs, so reviewers know which ones need extra attention.

> 🚧 Work in progress — data pipeline done, model training next.

## How it works

1. **Collect data** — pulls 500 merged PRs and 2,000 commits from the `psf/requests` repo using the GitHub API
2. **Label them** — a PR is marked "buggy" if a bug-fix commit later changed the same files within 90 days of the merge
3. **Train a model** *(coming next)* — predict the risk of new PRs from features like size, files changed, and author history

Current dataset: **201 buggy / 299 safe PRs (40% buggy)**

## Project structure

```
AutoDebugAI/
├── data/                    # raw JSON + labeled CSV (gitignored)
├── notebooks/               # experiments
├── src/
│   ├── collect.py           # fetch merged PRs
│   ├── collect_commits.py   # fetch commits
│   └── label.py             # create labeled dataset
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

Then:

```bash
python src/collect.py
python src/collect_commits.py
python src/label.py
```

This creates `data/processed/labeled_prs.csv` — the training dataset.

## Tech

Python · PyGithub · pandas · scikit-learn — with XGBoost, SHAP, FastAPI, and Streamlit planned.
