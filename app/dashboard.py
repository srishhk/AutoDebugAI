import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.model.predict import score_pr
from src.data.fetch_pr import fetch_pr_features

# sidebar opens by default now
st.set_page_config(page_title="AutoDebug AI", page_icon="🐛", layout="wide",
                   initial_sidebar_state="expanded")

BG, SURFACE, BORDER = "#0f1420", "#161d2e", "#232c40"
TEXT, MUTED, TEAL = "#e8ecf4", "#8b93a7", "#3dd4a8"
C_HIGH, C_MED, C_LOW = "#e05a6b", "#e0a83a", "#3dd4a8"
BG_HIGH, BG_MED, BG_LOW = "#3a1720", "#3a2e14", "#12352a"

st.markdown(f"""
<style>
  .stApp {{ background:{BG}; }}
  section[data-testid="stSidebar"] {{ background:{SURFACE}; border-right:1px solid {BORDER}; }}
  section[data-testid="stSidebar"] * {{ color:{TEXT}; }}
  #MainMenu, footer {{ visibility:hidden; }}
  /* hide the header bar itself but keep the sidebar toggle arrow alive */
  header[data-testid="stHeader"] {{ background:transparent; }}
  header[data-testid="stHeader"] * {{ visibility:hidden; }}
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="stSidebarCollapsedControl"] *,
  [data-testid="collapsedControl"],
  [data-testid="collapsedControl"] *,
  [data-testid="stSidebarCollapseButton"],
  [data-testid="stSidebarCollapseButton"] * {{
      visibility:visible !important;
      color:{TEXT} !important;
  }}
  [data-testid="stSidebarCollapsedControl"] svg,
  [data-testid="collapsedControl"] svg,
  [data-testid="stSidebarCollapseButton"] svg {{ fill:{TEXT} !important; }}
  h1,h2,h3,h4,p,span,label,div,td,th {{ color:{TEXT}; }}
  .card {{ background:{SURFACE}; border-radius:10px; padding:16px; margin-bottom:10px; }}
  .card-a {{ border-left:4px solid var(--a); border-radius:0 10px 10px 0; }}
  .lbl {{ font-size:10px; color:{MUTED}; letter-spacing:1px; text-transform:uppercase; }}
  .big {{ font-size:26px; font-weight:700; }}
  .cap {{ color:{MUTED} !important; font-size:12px; }}
  .pill {{ padding:3px 12px; border-radius:99px; font-size:12px; font-weight:600; }}
  .stButton>button {{ background:{TEAL}; color:#08110d; border:none; border-radius:8px; font-weight:600; }}
  .stButton>button:hover {{ background:#34c39a; color:#08110d; }}
  input {{ background:{BG} !important; color:{TEXT} !important; border:1px solid {BORDER} !important; }}
  .stNumberInput button {{ background:{SURFACE} !important; color:{TEXT} !important; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  td, th {{ padding:7px 5px; border-top:1px solid {BORDER}; text-align:left; }}
  th {{ color:{MUTED} !important; font-weight:400; border-top:none; }}
</style>
""", unsafe_allow_html=True)


def lvl(s):
    if s >= 0.7: return "high", C_HIGH, BG_HIGH
    if s >= 0.4: return "medium", C_MED, BG_MED
    return "low", C_LOW, BG_LOW


def stat_card(label, value, color=TEXT, accent=None):
    a = f'class="card card-a" style="--a:{accent}"' if accent else 'class="card"'
    return (f'<div {a}><div class="lbl">{label}</div>'
            f'<div class="big" style="color:{color}">{value}</div></div>')


def bars_html(contributions):
    if not contributions:
        return ""
    mx = max(abs(c["value"]) for c in contributions) or 1
    rows = ""
    for c in contributions:
        col = C_HIGH if c["direction"] == "raises" else C_LOW
        w = abs(c["value"]) / mx * 100
        sign = "+" if c["direction"] == "raises" else "−"
        rows += (
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;font-size:12px">'
            f'<span style="width:130px;color:#c5cbd8">{c["feature"]}</span>'
            f'<div style="flex:1;height:11px;background:{BORDER};border-radius:2px">'
            f'<div style="width:{w}%;height:11px;background:{col};border-radius:2px"></div></div>'
            f'<span style="width:44px;text-align:right;color:{col};font-weight:600">{sign}{abs(c["value"])}%</span></div>')
    return rows


def bar_chart_html(labels, values, colors, height=150):
    mx = max(values) or 1
    bars = ""
    for label, val, col in zip(labels, values, colors):
        h = (val / mx) * 100
        bars += (
            f'<div style="flex:1;display:flex;flex-direction:column;align-items:center;'
            f'justify-content:flex-end;height:100%">'
            f'<div style="font-size:12px;color:{TEXT};margin-bottom:5px">{val}</div>'
            f'<div style="width:100%;background:{col};height:{h}%;border-radius:4px 4px 0 0"></div>'
            f'</div>')
    axis = "".join(
        f'<span style="flex:1;text-align:center;font-size:11px;color:{MUTED}">{l}</span>'
        for l in labels)
    return (f'<div class="card">'
            f'<div style="display:flex;align-items:flex-end;gap:12px;height:{height}px">{bars}</div>'
            f'<div style="display:flex;gap:12px;margin-top:8px">{axis}</div></div>')


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed/labeled_prs_v2.csv"


@st.cache_data(show_spinner=False)
def load_and_score(n=30):
    df = pd.read_csv(DATA).sort_values("merged_at", ascending=False).head(n)
    rows = []
    for _, r in df.iterrows():
        pr = {k: r[k] for k in ["additions", "deletions", "changed_files", "commits",
                                "comments", "review_comments", "num_files",
                                "author_past_prs", "author_past_bug_rate", "is_first_pr"]}
        pr["title"] = r["title"]
        s = score_pr(pr)["risk_score"]
        rows.append({"score": s, "number": int(r["number"]), "title": r["title"],
                     "files": int(r["num_files"]), "additions": int(r["additions"])})
    return pd.DataFrame(rows).sort_values("score", ascending=False)


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("### 🐛 AutoDebug")
    page = st.radio("Nav", ["Overview", "Score a PR", "Review Queue", "About"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown(stat_card("Model AUC", "0.710", TEXT, TEAL), unsafe_allow_html=True)
    st.markdown(stat_card("Training PRs", "500", TEXT, MUTED), unsafe_allow_html=True)


# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.markdown("## Overview")
    st.markdown('<p class="cap">Bug-risk analysis across recent pull requests</p>', unsafe_allow_html=True)

    with st.spinner("Scoring recent PRs…"):
        df = load_and_score()

    high = int((df["score"] >= 0.7).sum())
    med = int(((df["score"] >= 0.4) & (df["score"] < 0.7)).sum())
    low = int((df["score"] < 0.4).sum())
    avg = df["score"].mean()

    c = st.columns(4)
    c[0].markdown(stat_card("PRs analyzed", "500", TEXT, TEAL), unsafe_allow_html=True)
    c[1].markdown(stat_card("High risk", high, C_HIGH, C_HIGH), unsafe_allow_html=True)
    c[2].markdown(stat_card("Avg risk", f"{avg:.0%}", C_MED, C_MED), unsafe_allow_html=True)
    c[3].markdown(stat_card("Buggy rate", "40%", TEXT, MUTED), unsafe_allow_html=True)

    # two equal boxes side by side
    left, right = st.columns(2)
    with left:
        st.markdown("**How risk is distributed**")
        bins = pd.cut(df["score"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                      labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"])
        hist = bins.value_counts().sort_index()
        st.markdown(bar_chart_html(list(hist.index), list(hist.values), [TEAL] * 5),
                    unsafe_allow_html=True)
    with right:
        st.markdown("**By risk level**")
        st.markdown(bar_chart_html(["High", "Med", "Low"], [high, med, low],
                                   [C_HIGH, C_MED, C_LOW]), unsafe_allow_html=True)

    st.markdown("**⚠️ Needs review first**")
    top = df.head(5)
    rows = "".join(
        f'<tr><td style="width:60px"><span class="pill" style="background:{lvl(r["score"])[2]};'
        f'color:{lvl(r["score"])[1]}">{r["score"]:.0%}</span></td>'
        f'<td style="width:60px;color:{MUTED}">#{r["number"]}</td>'
        f'<td>{r["title"]}</td></tr>' for _, r in top.iterrows())
    st.markdown(f'<div class="card"><table>{rows}</table></div>', unsafe_allow_html=True)


# ---------------- SCORE A PR ----------------
elif page == "Score a PR":
    st.markdown("## Score a pull request")
    st.markdown('<p class="cap">Fetches a live PR from psf/requests and predicts its bug risk</p>',
                unsafe_allow_html=True)

    pr_number = st.number_input("PR number", min_value=1, value=7555, step=1)

    if st.button("Analyze PR"):
        with st.spinner("Fetching from GitHub and scoring…"):
            try:
                f = fetch_pr_features(int(pr_number))
                res = score_pr(f)
                name, col, bg = lvl(res["risk_score"])

                st.markdown(
                    f'<div class="card card-a" style="--a:{col}">'
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
                    f'<div><div style="font-size:14px;font-weight:600">{f["title"]}</div>'
                    f'<div class="cap">PR #{pr_number}</div></div>'
                    f'<span class="pill" style="background:{bg};color:{col}">{name.upper()}</span></div>'
                    f'<div style="display:flex;gap:22px;padding-top:10px;margin-top:10px;'
                    f'border-top:1px solid {BORDER};font-size:11px">'
                    f'<div><span style="color:{MUTED}">Files</span> <b>{f["changed_files"]}</b></div>'
                    f'<div><span style="color:{MUTED}">Added</span> <b style="color:{C_LOW}">+{f["additions"]}</b></div>'
                    f'<div><span style="color:{MUTED}">Deleted</span> <b style="color:{C_HIGH}">−{f["deletions"]}</b></div>'
                    f'<div><span style="color:{MUTED}">Commits</span> <b>{f["commits"]}</b></div>'
                    f'<div><span style="color:{MUTED}">Comments</span> <b>{f["comments"]}</b></div></div></div>',
                    unsafe_allow_html=True)

                g, b = st.columns([1, 1.6])
                with g:
                    st.markdown(
                        f'<div class="card" style="text-align:center">'
                        f'<div class="lbl">Risk score</div>'
                        f'<div style="font-size:52px;font-weight:700;color:{col};margin:10px 0">'
                        f'{res["risk_score"]:.0%}</div>'
                        f'<div style="height:8px;background:{BORDER};border-radius:99px">'
                        f'<div style="width:{res["risk_score"]*100}%;height:8px;background:{col};'
                        f'border-radius:99px"></div></div></div>', unsafe_allow_html=True)
                with b:
                    st.markdown(
                        f'<div class="card"><div class="lbl">Why? — feature contributions</div>'
                        f'<div style="margin-top:14px">{bars_html(res["contributions"])}</div>'
                        f'<div class="cap">Red raises risk · green lowers it (SHAP values)</div></div>',
                        unsafe_allow_html=True)

                with st.expander("Raw features"):
                    st.json(f)
            except Exception as e:
                st.error(f"Could not score PR #{pr_number}: {e}")


# ---------------- REVIEW QUEUE ----------------
elif page == "Review Queue":
    st.markdown("## Review queue")
    st.markdown('<p class="cap">30 most recent PRs, sorted riskiest-first</p>', unsafe_allow_html=True)

    with st.spinner("Scoring recent PRs…"):
        df = load_and_score()

    high = int((df["score"] >= 0.7).sum())
    med = int(((df["score"] >= 0.4) & (df["score"] < 0.7)).sum())
    low = int((df["score"] < 0.4).sum())

    c = st.columns(3)
    c[0].markdown(stat_card("High risk", high, C_HIGH, C_HIGH), unsafe_allow_html=True)
    c[1].markdown(stat_card("Medium risk", med, C_MED, C_MED), unsafe_allow_html=True)
    c[2].markdown(stat_card("Low risk", low, C_LOW, C_LOW), unsafe_allow_html=True)

    rows = "".join(
        f'<tr><td style="width:60px"><span class="pill" style="background:{lvl(r["score"])[2]};'
        f'color:{lvl(r["score"])[1]}">{r["score"]:.0%}</span></td>'
        f'<td style="width:60px;color:{MUTED}">#{r["number"]}</td><td>{r["title"]}</td>'
        f'<td style="width:50px;color:{MUTED}">{r["files"]}</td>'
        f'<td style="width:60px;color:{MUTED}">+{r["additions"]}</td></tr>'
        for _, r in df.iterrows())
    st.markdown(
        f'<div class="card"><table><tr><th>Risk</th><th>PR</th><th>Title</th>'
        f'<th>Files</th><th>+Lines</th></tr>{rows}</table></div>', unsafe_allow_html=True)


# ---------------- ABOUT ----------------
else:
    st.markdown("## About AutoDebug AI")
    st.markdown(
        f'<div class="card card-a" style="--a:{TEAL}">Predicts which pull requests are likely to '
        f'introduce bugs, so reviewers can prioritize the riskiest changes.</div>',
        unsafe_allow_html=True)

    c = st.columns(3)
    c[0].markdown('<div class="card"><div class="lbl">Model</div>'
                  '<div style="font-size:14px;margin-top:5px">Random Forest</div>'
                  '<p class="cap">beat XGBoost, LightGBM, LogReg</p></div>', unsafe_allow_html=True)
    c[1].markdown('<div class="card"><div class="lbl">Features</div>'
                  '<div style="font-size:14px;margin-top:5px">40 total</div>'
                  '<p class="cap">10 numeric + 30 PCA title embeddings</p></div>', unsafe_allow_html=True)
    c[2].markdown('<div class="card"><div class="lbl">Explainability</div>'
                  '<div style="font-size:14px;margin-top:5px">SHAP</div>'
                  '<p class="cap">per-PR feature contributions</p></div>', unsafe_allow_html=True)

    st.markdown('<p class="cap">Python · scikit-learn · sentence-transformers · SHAP · '
                'FastAPI · Streamlit · Docker</p>', unsafe_allow_html=True)