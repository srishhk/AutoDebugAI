import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import score_pr
from src.fetch_pr import fetch_pr_features

st.set_page_config(page_title="AutoDebug AI", page_icon="🔍", layout="wide")

st.title("🔍 AutoDebug AI")
st.caption("Predicts which pull requests are likely to introduce bugs — so reviewers know where to look first.")

RISK_COLORS = {"high": "🔴", "medium": "🟠", "low": "🟢"}


def risk_level(score):
    return "high" if score >= 0.7 else "medium" if score >= 0.4 else "low"


# ---------- section 1: score a live PR ----------
st.header("Score a live PR")
pr_number = st.number_input("PR number from psf/requests", min_value=1, value=7555, step=1)

if st.button("Score it"):
    with st.spinner("Fetching PR from GitHub and scoring..."):
        try:
            features = fetch_pr_features(int(pr_number))
            result = score_pr(features)
            level = risk_level(result["risk_score"])

            col1, col2 = st.columns(2)
            col1.metric("Risk score", f"{result['risk_score']:.0%}")
            col2.metric("Risk level", f"{RISK_COLORS[level]} {level}")

            st.subheader("Why?")
            for reason in result["reasons"]:
                st.write(f"- {reason}")

            st.subheader("PR details")
            st.json(features)
        except Exception as e:
            st.error(f"Could not score PR #{pr_number}: {e}")

# ---------- section 2: the review queue ----------
st.header("Review queue — recent PRs by risk")

DATA = Path("data/processed/labeled_prs_v2.csv")

@st.cache_data
def load_and_score(n=30):
    df = pd.read_csv(DATA).sort_values("merged_at", ascending=False).head(n)
    rows = []
    for _, r in df.iterrows():
        pr_dict = {
            "additions": r["additions"], "deletions": r["deletions"],
            "changed_files": r["changed_files"], "commits": r["commits"],
            "comments": r["comments"], "review_comments": r["review_comments"],
            "num_files": r["num_files"],
            "author_past_prs": r["author_past_prs"],
            "author_past_bug_rate": r["author_past_bug_rate"],
            "is_first_pr": r["is_first_pr"],
            "title": r["title"],
        }
        result = score_pr(pr_dict)
        level = risk_level(result["risk_score"])
        rows.append({
            "Risk": f"{RISK_COLORS[level]} {result['risk_score']:.0%}",
            "PR #": int(r["number"]),
            "Title": r["title"],
            "Files": int(r["num_files"]),
            "+Lines": int(r["additions"]),
            "_sort": result["risk_score"],
        })
    out = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort")
    return out

st.dataframe(load_and_score(), use_container_width=True, hide_index=True)
st.caption("30 most recent PRs from the dataset, scored by the model and sorted riskiest-first.")