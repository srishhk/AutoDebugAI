# import json
# from pathlib import Path

# import pandas as pd
# import streamlit as st

# from src.predict import score_pr
# from src.fetch_pr import fetch_pr_features

# st.set_page_config(page_title="AutoDebug AI", page_icon="🔍", layout="wide")

# st.title("🔍 AutoDebug AI")
# st.caption("Predicts which pull requests are likely to introduce bugs — so reviewers know where to look first.")

# RISK_COLORS = {"high": "🔴", "medium": "🟠", "low": "🟢"}


# def risk_level(score):
#     return "high" if score >= 0.7 else "medium" if score >= 0.4 else "low"


# # ---------- section 1: score a live PR ----------
# st.header("Score a live PR")
# pr_number = st.number_input("PR number from psf/requests", min_value=1, value=7555, step=1)

# if st.button("Score it"):
#     with st.spinner("Fetching PR from GitHub and scoring..."):
#         try:
#             features = fetch_pr_features(int(pr_number))
#             result = score_pr(features)
#             level = risk_level(result["risk_score"])

#             col1, col2 = st.columns(2)
#             col1.metric("Risk score", f"{result['risk_score']:.0%}")
#             col2.metric("Risk level", f"{RISK_COLORS[level]} {level}")

#             st.subheader("Why?")
#             for reason in result["reasons"]:
#                 st.write(f"- {reason}")

#             st.subheader("PR details")
#             st.json(features)
#         except Exception as e:
#             st.error(f"Could not score PR #{pr_number}: {e}")

# # ---------- section 2: the review queue ----------
# st.header("Review queue — recent PRs by risk")

# DATA = Path("data/processed/labeled_prs_v2.csv")

# @st.cache_data
# def load_and_score(n=30):
#     df = pd.read_csv(DATA).sort_values("merged_at", ascending=False).head(n)
#     rows = []
#     for _, r in df.iterrows():
#         pr_dict = {
#             "additions": r["additions"], "deletions": r["deletions"],
#             "changed_files": r["changed_files"], "commits": r["commits"],
#             "comments": r["comments"], "review_comments": r["review_comments"],
#             "num_files": r["num_files"],
#             "author_past_prs": r["author_past_prs"],
#             "author_past_bug_rate": r["author_past_bug_rate"],
#             "is_first_pr": r["is_first_pr"],
#             "title": r["title"],
#         }
#         result = score_pr(pr_dict)
#         level = risk_level(result["risk_score"])
#         rows.append({
#             "Risk": f"{RISK_COLORS[level]} {result['risk_score']:.0%}",
#             "PR #": int(r["number"]),
#             "Title": r["title"],
#             "Files": int(r["num_files"]),
#             "+Lines": int(r["additions"]),
#             "_sort": result["risk_score"],
#         })
#     out = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort")
#     return out

# st.dataframe(load_and_score(), use_container_width=True, hide_index=True)
# st.caption("30 most recent PRs from the dataset, scored by the model and sorted riskiest-first.")

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict import score_pr
from src.fetch_pr import fetch_pr_features

# ---------------- page config ----------------
st.set_page_config(page_title="AutoDebug AI", page_icon="🔍", layout="wide")

# ---------------- styling ----------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2a3142;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-label { color: #8b93a7; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 34px; font-weight: 700; margin-top: 6px; }
    .high { color: #ef4444; }
    .medium { color: #f59e0b; }
    .low { color: #22c55e; }
    h1, h2, h3 { color: #e6e9ef; }
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}
RISK_DOTS = {"high": "🔴", "medium": "🟠", "low": "🟢"}


def risk_level(score):
    return "high" if score >= 0.7 else "medium" if score >= 0.4 else "low"


def gauge(score):
    level = risk_level(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={"suffix": "%", "font": {"size": 40, "color": RISK_COLORS[level]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b93a7"},
            "bar": {"color": RISK_COLORS[level]},
            "bgcolor": "#1a1f2e",
            "steps": [
                {"range": [0, 40], "color": "#14321f"},
                {"range": [40, 70], "color": "#3a2e14"},
                {"range": [70, 100], "color": "#3a1717"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e6e9ef"})
    return fig


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("🔍 AutoDebug AI")
    st.caption("Bug-risk prediction for pull requests")
    page = st.radio("Navigate", ["Score a PR", "Review Queue", "About"])
    st.markdown("---")
    st.metric("Model AUC", "0.710")
    st.metric("Training PRs", "500")
    st.caption("Random Forest · numeric + NLP features")


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
            "num_files": r["num_files"], "author_past_prs": r["author_past_prs"],
            "author_past_bug_rate": r["author_past_bug_rate"], "is_first_pr": r["is_first_pr"],
            "title": r["title"],
        }
        score = score_pr(pr_dict)["risk_score"]
        level = risk_level(score)
        rows.append({
            "Risk": f"{RISK_DOTS[level]} {score:.0%}",
            "PR #": int(r["number"]), "Title": r["title"],
            "Files": int(r["num_files"]), "+Lines": int(r["additions"]),
            "_score": score,
        })
    return pd.DataFrame(rows).sort_values("_score", ascending=False)


# ---------------- page: score a PR ----------------
if page == "Score a PR":
    st.header("Score a live pull request")
    st.caption("Fetches a real PR from psf/requests and predicts its bug risk.")

    pr_number = st.number_input("PR number", min_value=1, value=7555, step=1)

    if st.button("Analyze PR", type="primary"):
        with st.spinner("Fetching from GitHub and scoring..."):
            try:
                features = fetch_pr_features(int(pr_number))
                result = score_pr(features)
                level = risk_level(result["risk_score"])

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.plotly_chart(gauge(result["risk_score"]), use_container_width=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Risk Level</div>
                        <div class="metric-value {level}">{RISK_DOTS[level]} {level.upper()}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("#### Why?")
                    for reason in result["reasons"]:
                        st.markdown(f"- {reason}")

                with st.expander("PR details"):
                    st.json(features)
            except Exception as e:
                st.error(f"Could not score PR #{pr_number}: {e}")


# ---------------- page: review queue ----------------
elif page == "Review Queue":
    st.header("Review queue")
    st.caption("30 most recent PRs, scored and sorted riskiest-first.")

    with st.spinner("Scoring recent PRs..."):
        df = load_and_score()

    high = (df["_score"] >= 0.7).sum()
    med = ((df["_score"] >= 0.4) & (df["_score"] < 0.7)).sum()
    low = (df["_score"] < 0.4).sum()

    c1, c2, c3 = st.columns(3)
    for col, label, val, cls in [(c1, "High risk", high, "high"),
                                  (c2, "Medium risk", med, "medium"),
                                  (c3, "Low risk", low, "low")]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {cls}">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("###")

    left, right = st.columns([2, 1])
    with left:
        st.dataframe(df.drop(columns="_score"), use_container_width=True, hide_index=True, height=460)
    with right:
        fig = go.Figure(go.Bar(
            x=[high, med, low], y=["High", "Medium", "Low"], orientation="h",
            marker_color=[RISK_COLORS["high"], RISK_COLORS["medium"], RISK_COLORS["low"]],
        ))
        fig.update_layout(title="Risk distribution", height=460,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"color": "#e6e9ef"}, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ---------------- page: about ----------------
else:
    st.header("About AutoDebug AI")
    st.markdown("""
    AutoDebug AI predicts which pull requests are likely to introduce bugs,
    so reviewers can prioritize the riskiest changes.

    **How it works**
    - Trained on 500 merged PRs and 2,000 commits from `psf/requests`
    - Features: PR size, review activity, and NLP embeddings of PR titles
    - Model: Random Forest (AUC 0.710), with SHAP-based explanations

    **Tech:** Python · scikit-learn · sentence-transformers · SHAP · FastAPI · Streamlit
    """)