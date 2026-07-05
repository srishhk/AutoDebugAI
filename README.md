# AutoDebug AI

Predicts which GitHub pull requests are likely to introduce bugs, so reviewers know which ones need extra attention.

Work in progress — baseline models trained, NLP features next.

## How it works

1. **Collect data** — pulls 500 merged PRs and 2,000 commits from the `psf/requests` repo using the GitHub API
2. **Label them** — a PR is marked "buggy" if a bug-fix commit later changed the same files within 90 days of the merge
3. **Train a model** *(coming next)* — predict the risk of new PRs from features like size, files changed, and author history

Current dataset: **201 buggy / 299 safe PRs (40% buggy)**


## Results so far 
| Model | Features | AUC |
|---|---|---|
| Logistic Regression | 7 numeric (PR size, review activity) | **0.705** |
| Random Forest | 7 numeric | 0.662 |
| Logistic Regression | + author history (10 features) | 0.695 |
| Random Forest | + author history | 0.609 |

Author-history features (past PRs, past bug rate) did not improve performance —
most PRs in this sample come from one-time contributors, so authors have little
usable history. Next step: NLP embeddings of PR titles.

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
