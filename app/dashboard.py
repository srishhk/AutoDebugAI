from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.model.predict import score_pr
from src.data.fetch_pr import fetch_pr_features

st.set_page_config(page_title="AutoDebug AI", page_icon="🐛", layout="wide",
                   initial_sidebar_state="expanded")

# ---------- palette ----------
BG      = "#0f1420"
SURFACE = "#161d2e"
BORDER  = "#232c40"
TEXT    = "#e8ecf4"
MUTED   = "#8b93a7"
TEAL    = "#02261C"
C_HIGH, C_MED, C_LOW = "#e05a6b", "#e0a83a", "#3dd4a8"

st.markdown(f"""
<style>
    .stApp {{ background:{BG}; color:{TEXT}; }}
    section[data-testid="stSidebar"] {{ background:{SURFACE}; border-right:1px solid {BORDER}; }}
    section[data-testid="stSidebar"] * {{ color:{TEXT}; }}
    #MainMenu, header, footer {{ visibility:hidden; }}

    .card {{ background:{SURFACE}; border:0.5px solid {BORDER}; border-radius:12px; padding:20px; }}
    .card-accent {{ border-left:5px solid var(--a); border-radius:0; }}
    .lbl {{ font-size:11px; color:{MUTED}; text-transform:uppercase; letter-spacing:1px; }}
    .pill {{ padding:4px 16px; border-radius:999px; font-size:16px; font-weight:600; }}

    h1,h2,h3,h4,p,span,label,div {{ color:{TEXT}; }}
    .cap {{ color:{MUTED} !important; font-size:13px; }}

    .stButton>button {{ background:{TEAL}; color:#08110d; border:none; border-radius:8px;
                        font-weight:600; padding:8px 20px; }}
    .stButton>button:hover {{ background:#34c39a; color:#08110d; }}

    input {{ background:{BG} !important; color:{TEXT} !important; border:0.5px solid {BORDER} !important; }}
    .stNumberInput button {{ background:{SURFACE} !important; color:{TEXT} !important; }}
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {"high": C_HIGH, "medium": C_MED, "low": C_LOW}
RISK_BG = {"high": "#3a1720", "medium": "#3a2e14", "low": "#12352a"}
DOT = {"high": "🔴", "medium": "🟠", "low": "🟢"}


def risk_level(s):
    return "high" if s >= 0.7 else "medium" if s >= 0.4 else "low"


def gauge(score):
    lvl = risk_level(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score * 100,
        number={"suffix": "%", "font": {"size": 42, "color": RISK_COLORS[lvl]}},
        gauge={"axis": {"range": [0, 100], "tickcolor": MUTED, "tickwidth": 1},
               "bar": {"color": RISK_COLORS[lvl], "thickness": 0.72},
               "bgcolor": BORDER, "borderwidth": 0,
               "steps": [{"range": [0, 40], "color": "#12352a"},
                         {"range": [40, 70], "color": "#3a2e14"},
                         {"range": [70, 100], "color": "#3a1720"}]}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT})
    return fig


# ---------- sidebar ----------
with st.sidebar:
    st.markdown(f"### 🐛 AutoDebug")
    page = st.radio("Navigate", ["Score a PR", "Review Queue", "About"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f'<div class="card card-accent" style="--a:{TEAL}"><div class="lbl">Model AUC</div>'
                f'<div style="font-size:26px;font-weight:600">0.710</div></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(f'<div class="card card-accent" style="--a:{MUTED}"><div class="lbl">Training PRs</div>'
                f'<div style="font-size:26px;font-weight:600">500</div></div>', unsafe_allow_html=True)


DATA = Path("data/processed/labeled_prs_v2.csv")


@st.cache_data
def load_and_score(n=30):
    df = pd.read_csv(DATA).sort_values("merged_at", ascending=False).head(n)
    rows = []
    for _, r in df.iterrows():
        pr = {k: r[k] for k in ["additions", "deletions", "changed_files", "commits",
                                "comments", "review_comments", "num_files",
                                "author_past_prs", "author_past_bug_rate", "is_first_pr"]}
        pr["title"] = r["title"]
        s = score_pr(pr)["risk_score"]
        lvl = risk_level(s)
        rows.append({"Risk": f"{DOT[lvl]} {s:.0%}", "PR #": int(r["number"]),
                     "Title": r["title"], "Files": int(r["num_files"]),
                     "+Lines": int(r["additions"]), "_s": s})
    return pd.DataFrame(rows).sort_values("_s", ascending=False)


# ---------- SCORE A PR (default page) ----------
if page == "Score a PR":
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("## Score a pull request")
        st.markdown('<p class="cap">Fetches a live PR from psf/requests and predicts its bug risk.</p>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div style="text-align:right"><span style="background:#12352a;color:{TEAL};'
                    f'padding:6px 14px;border-radius:999px;font-size:13px;font-weight:600">Model AUC 0.710</span></div>',
                    unsafe_allow_html=True)

    pr_number = st.number_input("PR number", min_value=1, value=7555, step=1)

    if st.button("Analyze PR"):
        with st.spinner("Fetching from GitHub and scoring…"):
            try:
                features = fetch_pr_features(int(pr_number))
                result = score_pr(features)
                lvl = risk_level(result["risk_score"])

                g, d = st.columns([1, 1])
                with g:
                     st.markdown('<div class="lbl">Risk score</div>', unsafe_allow_html=True)
                     st.plotly_chart(gauge(result["risk_score"]), use_container_width=True)
                with d:
                    st.markdown(
                        f'<div class="card card-accent" style="--a:{RISK_COLORS[lvl]}">'
                        f'<div class="lbl">Risk level</div>'
                        f'<div style="margin:10px 0 18px"><span class="pill" '
                        f'style="background:{RISK_BG[lvl]};color:{RISK_COLORS[lvl]}">{DOT[lvl]} {lvl.upper()}</span></div>'
                        f'<div class="lbl">Why?</div>'
                        + "".join(f'<div style="color:{TEXT};padding:4px 0">• {r}</div>' for r in result["reasons"])
                        + '</div>', unsafe_allow_html=True)

                with st.expander("PR details"):
                    st.json(features)
            except Exception as e:
                st.error(f"Could not score PR #{pr_number}: {e}")


# ---------- REVIEW QUEUE ----------
elif page == "Review Queue":
    st.markdown("## Review queue")
    st.markdown('<p class="cap">30 most recent PRs, scored and sorted riskiest-first.</p>',
                unsafe_allow_html=True)
    with st.spinner("Scoring recent PRs…"):
        df = load_and_score()

    high = int((df["_s"] >= 0.7).sum())
    med = int(((df["_s"] >= 0.4) & (df["_s"] < 0.7)).sum())
    low = int((df["_s"] < 0.4).sum())

    cols = st.columns(3)
    for col, lbl, val, c in [(cols[0], "High risk", high, C_HIGH),
                             (cols[1], "Medium risk", med, C_MED),
                             (cols[2], "Low risk", low, C_LOW)]:
        col.markdown(f'<div class="card card-accent" style="--a:{c}"><div class="lbl">{lbl}</div>'
                     f'<div style="font-size:32px;font-weight:600;color:{c}">{val}</div></div>',
                     unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([2, 1])
    with left:
        st.dataframe(df.drop(columns="_s"), use_container_width=True, hide_index=True, height=440)
    with right:
        fig = go.Figure(go.Bar(x=[high, med, low], y=["High", "Medium", "Low"], orientation="h",
                               marker_color=[C_HIGH, C_MED, C_LOW], text=[high, med, low],
                               textposition="outside", textfont={"color": TEXT}))
        fig.update_layout(title="Risk distribution", height=440, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font={"color": TEXT},
                          margin=dict(l=10, r=10, t=40, b=10), xaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)


# ---------- ABOUT ----------
else:
    st.markdown("## About AutoDebug AI")
    st.markdown(f"""
    <div class="card card-accent" style="--a:{TEAL}">
    AutoDebug AI predicts which pull requests are likely to introduce bugs,
    so reviewers can prioritize the riskiest changes.
    </div>""", unsafe_allow_html=True)
    st.markdown("""
    **How it works**
    - Trained on 500 merged PRs and 2,000 commits from `psf/requests`
    - Features: PR size, review activity, and NLP embeddings of PR titles
    - Model: Random Forest (AUC 0.710), explained with SHAP

    **Tech:** Python · scikit-learn · sentence-transformers · SHAP · FastAPI · Streamlit
    """)                                                               





