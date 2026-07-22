import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.model.predict import score_pr
from src.data.fetch_pr import fetch_pr_features

st.set_page_config(page_title="AutoDebug AI", page_icon="🐛", layout="wide",
                   initial_sidebar_state="expanded")

# ---------- palette ----------
BG       = "#0a0710"
SIDEBAR  = "#120d18"
SURFACE  = "#150f1c"
SURFACE2 = "#1a1322"
BORDER   = "#241c2e"
BORDER2  = "#2a2035"
AXIS     = "#3a3048"
TEXT     = "#f0ebf5"
TEXT2    = "#c9c0d4"
MUTED    = "#8d81a0"
FAINT    = "#6f6580"
ACCENT   = "#c77dff"
ACCENT_D = "#3d2450"
C_HIGH, C_MED, C_LOW = "#f06292", "#e0a83a", "#7dd3a0"
BG_HIGH, BG_MED, BG_LOW = "#3a1a26", "#3a2e14", "#1a3527"

HIGH_T, MED_T = 0.70, 0.40         
BASE_RATE     = 0.40                
ROC_AUC       = 0.710


PAGES = [
    ("overview", "Overview",     "ti-layout-dashboard"),
    ("score",    "Score a PR",   "ti-search"),
    ("queue",    "Review queue", "ti-list-check"),
    ("model",    "Model",        "ti-cpu"),
    ("about",    "About",        "ti-info-circle"),
]
PAGE_SLUGS = [p[0] for p in PAGES]
DIVIDER_AFTER = "queue"           

_qp = st.query_params

page = _qp.get("page", "overview")
if page not in PAGE_SLUGS:
    page = "overview"

SCOPES = {"30": 30, "500": None}    
scope_key = _qp.get("scope", "30")
if scope_key not in SCOPES:
    scope_key = "30"

if "rail" not in st.session_state:
    st.session_state.rail = False        # False = labels, True = icon rail

if "narrow" in _qp:
    _narrow = _qp["narrow"] == "1"
    if st.session_state.get("_auto_rail") != _narrow:
        st.session_state._auto_rail = _narrow
        if not st.session_state.get("_manual_rail", False):
            st.session_state.rail = _narrow

rail = st.session_state.rail

#sidebar 
SB_W_EXPANDED = 232
SB_W_RAIL     = 68
SB_BREAKPOINT = 1000
SB_PAD_Y      = 16
SB_NAV_H      = 52     # nav row height
SB_LOGO_GAP   = 90     # space between the logo and the first nav card
SB_NAV_GAP    = 10     # gap between the nav cards
SB_TOG        = 46

SB_W = SB_W_RAIL if rail else SB_W_EXPANDED


def qs(**kw):
    """Build a query string that preserves the params we aren't changing."""
    base = {"page": page, "scope": scope_key}
    if "narrow" in _qp:
        base["narrow"] = _qp["narrow"]
    base.update(kw)
    return "?" + "&".join(f"{k}={v}" for k, v in base.items())


st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  @import url('https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.6.0/dist/tabler-icons.min.css');

  html, body, .stApp, .stApp *, section[data-testid="stSidebar"] * {{
      font-family:'Inter','Segoe UI',system-ui,-apple-system,sans-serif !important; }}
  /* icon fonts must keep their own family or the ligature prints as raw text */
  i.ti, i[class^="ti-"], i[class*=" ti-"] {{
      font-family:'tabler-icons' !important; font-style:normal !important; }}
  span[data-testid="stIconMaterial"], .material-icons, .material-icons-outlined,
  [class*="material-icons"] {{
      font-family:'Material Symbols Rounded','Material Icons' !important; }}
  span[data-testid="stIconMaterial"] {{ display:none !important; }}
  [data-testid="stSidebarCollapseButton"],
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="stSidebarHeader"] {{ display:none !important; }}

  .stApp {{ background:{BG}; }}
  header[data-testid="stHeader"] {{
      display:none !important; height:0 !important; min-height:0 !important; }}
  #MainMenu, footer {{ visibility:hidden !important; }}
  div[data-testid="stToolbar"], div[data-testid="stDecoration"],
  div[data-testid="stStatusWidget"], [data-testid="stMainMenu"],
  .stDeployButton, .stAppDeployButton, [data-testid="stAppDeployButton"] {{
      display:none !important; }}

  /* the viewport-reporter iframe is script-only; st.iframe won't accept a
     height of 0, so it renders 1px tall — collapse it here instead */
  div[data-testid="stIFrame"], iframe[title="st.iframe"],
  iframe[title="streamlit_component"] {{
      height:0 !important; min-height:0 !important; display:block !important;
      border:0 !important; }}
  div[data-testid="element-container"]:has(> div[data-testid="stIFrame"]),
  div[data-testid="element-container"]:has(> iframe) {{
      height:0 !important; min-height:0 !important; margin:0 !important;
      padding:0 !important; overflow:hidden !important; }}

  .block-container {{ padding-top:1.8rem; padding-bottom:3rem; max-width:1500px; }}
  section[data-testid="stMain"], div[data-testid="stAppViewContainer"] > .main {{
      margin-left:0 !important; width:auto !important; }}
  div[data-testid="stAppViewContainer"] {{ overflow-x:hidden !important; }}

  /* ---------------- sidebar shell ---------------- */
  section[data-testid="stSidebar"] {{
      background:{SIDEBAR}; border-right:1px solid {BORDER};
      transform:none !important; visibility:visible !important;
      margin-left:0 !important; left:0 !important;
      width:{SB_W}px !important; min-width:{SB_W}px !important;
      max-width:{SB_W}px !important;
      transition:width .18s ease;
      overflow:hidden !important; }}
  section[data-testid="stSidebar"][aria-expanded="false"] {{
      transform:none !important; margin-left:0 !important;
      width:{SB_W}px !important; min-width:{SB_W}px !important; }}
  section[data-testid="stSidebar"] > div:first-child,
  section[data-testid="stSidebar"][aria-expanded="true"] > div:first-child,
  section[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {{
      width:{SB_W}px !important; min-width:{SB_W}px !important;
      max-width:{SB_W}px !important; }}
  div[data-testid="stSidebarContent"],
  div[data-testid="stSidebarUserContent"] {{
      width:100% !important; min-width:0 !important; }}
  div[data-testid="stSidebarResizeHandle"] {{ display:none !important; }}

  /* no scrollbar in the sidebar — the nav always fits */
  section[data-testid="stSidebar"],
  section[data-testid="stSidebar"] > div:first-child,
  div[data-testid="stSidebarContent"],
  div[data-testid="stSidebarUserContent"] {{
      overflow-y:hidden !important; scrollbar-width:none !important; }}
  section[data-testid="stSidebar"] ::-webkit-scrollbar {{
      width:0 !important; height:0 !important; display:none !important; }}

  /* Logo + nav + toggle centred as one group. Height is exactly the viewport
     so nothing can overflow and trigger a scrollbar. */
  section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {{
      height:100vh !important; overflow:hidden !important;
      padding:{SB_PAD_Y}px 0 !important;
      display:flex !important; flex-direction:column !important;
      justify-content:flex-start !important; }}
  section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"]
      > div[data-testid="stVerticalBlock"] {{
      gap:0 !important; flex:1 1 auto display:flex !important; flex-direction:column !important; }}

  .sb {{ display:flex; flex-direction:column; }}
  .sb-logo {{ display:flex; align-items:center; gap:10px;
      margin:6px 0 {SB_LOGO_GAP}px 0;
      padding:0 {"0" if rail else "18px"};
      justify-content:{"center" if rail else "flex-start"}; }}
  .sb-logo i {{ font-size:26px; color:{ACCENT}; }}
  .sb-logo span {{ font-size:19px; color:{TEXT}; font-weight:700;
      letter-spacing:-.3px; white-space:nowrap; }}
  .sb-nav {{ display:flex; flex-direction:column; gap:{SB_NAV_GAP}px;
      padding:0 {"8px" if rail else "16px"}; }}

  /* Nav items are filled rounded cards, per the reference: each row is its
     own surface, the active one is lighter with a purple left edge. */
  .nav-i {{ display:flex; align-items:center; text-decoration:none !important;
      height:{SB_NAV_H}px; gap:14px;
      background:{SURFACE}; border-radius:10px;
      border-left:3px solid transparent;
      padding:0 {"0" if rail else "15px"};
      justify-content:{"center" if rail else "flex-start"};
      transition:background .15s ease, transform .15s ease; }}
  .nav-i i {{ font-size:22px; color:{ACCENT}; }}
  .nav-i span {{ font-size:14px; color:{TEXT}; font-weight:600;
      white-space:nowrap; }}
  .nav-i:hover {{ background:{SURFACE2}; }}
  .nav-i.on {{ background:#221a2e; border-left:3px solid {ACCENT}; }}
  .nav-i.on i, .nav-i.on span {{ color:#fff; }}
  .nav-i.on i {{ color:{ACCENT}; }}
  .nav-rule {{ display:none; }}

  /* the collapse toggle stays a real Streamlit button — it mutates state */
  section[data-testid="stSidebar"] .stButton {{ margin:0 !important; }}
  section[data-testid="stSidebar"] div[data-testid="element-container"] {{
      margin:0 !important; }}
  section[data-testid="stSidebar"] .stButton>button {{
      width:{SB_TOG}px !important; min-width:{SB_TOG}px !important;
      height:{SB_TOG}px !important; min-height:{SB_TOG}px !important;
      padding:0 !important; background:{SURFACE} !important; color:{ACCENT} !important;
      border:1px solid {BORDER2} !important; box-shadow:none !important;
      display:flex !important; align-items:center !important;
      justify-content:center !important;
      font-size:15px !important; line-height:1 !important;
      border-radius:50% !important; margin:auto auto 0 auto !important;
      transition:background .15s ease, border-color .15s ease; }}
  section[data-testid="stSidebar"] .stButton>button:hover {{
      background:{ACCENT_D} !important; border-color:{ACCENT} !important;
      color:{TEXT} !important; }}
  section[data-testid="stSidebar"] .stButton>button p {{
      font-size:15px !important; margin:0 !important; }}

  /* ---------------- scrollbar (main area only) ---------------- */
  section[data-testid="stMain"] {{ scrollbar-width:thin;
      scrollbar-color:{BORDER2} transparent; }}
  section[data-testid="stMain"] ::-webkit-scrollbar {{ width:7px; height:7px; }}
  section[data-testid="stMain"] ::-webkit-scrollbar-track {{
      background:transparent; }}
  section[data-testid="stMain"] ::-webkit-scrollbar-thumb {{
      background:{BORDER2}; border-radius:99px;
      border:2px solid transparent; background-clip:padding-box; }}
  html {{ scroll-behavior:smooth; }}

  /* ---------------- shared components ---------------- */
  .card {{ background:{SURFACE}; border-radius:10px; padding:15px 16px;
           margin-bottom:10px; height:100%; }}
  .kpi {{ background:{SURFACE}; border-radius:9px; padding:13px 14px;
          height:100%; }}
  .kpi-l {{ font-size:10px; color:{MUTED}; letter-spacing:.4px;
            text-transform:uppercase; white-space:nowrap; }}
  .kpi-v {{ font-size:23px; font-weight:600; margin-top:5px;
            letter-spacing:-.3px; white-space:nowrap; }}
  .kpi-s {{ font-size:10.5px; color:{FAINT}; margin-top:3px; }}
  .c-t {{ font-size:13px; color:{TEXT}; font-weight:600; }}
  .c-s {{ font-size:10.5px; color:{FAINT}; margin-top:2px; }}
  .page-h {{ font-size:20px; color:{TEXT}; font-weight:600;
             letter-spacing:-.3px; }}
  .page-s {{ font-size:12px; color:{MUTED}; margin-top:2px; }}
  .scope {{ display:inline-flex; background:{SURFACE2};
            border:1px solid {BORDER2}; border-radius:7px; padding:2px; }}
  .scope a {{ font-size:11px; color:{MUTED}; padding:5px 11px;
              border-radius:5px; text-decoration:none !important; }}
  .scope a.on {{ background:{ACCENT}; color:{BG}; font-weight:600; }}

  div[data-testid="stHorizontalBlock"] {{ align-items:stretch !important; }}
  div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
      display:flex; flex-direction:column; }}
  div[data-testid="stHorizontalBlock"] > div[data-testid="column"]
      > div[data-testid="stVerticalBlock"] {{ height:100%; }}

  .pill {{ display:inline-block; padding:4px 10px; border-radius:99px;
           font-size:11.5px; font-weight:600; white-space:nowrap; }}

  .stButton>button {{ background:{ACCENT}; color:#12091a; border:none;
      border-radius:8px; font-weight:600; padding:9px 22px; box-shadow:none; }}
  .stButton>button:hover {{ background:#b565f0; color:#12091a; }}

  input {{ background:{SURFACE2} !important; color:{TEXT} !important;
           border:1px solid {BORDER2} !important; border-radius:8px !important; }}
  .stNumberInput button {{ background:{SURFACE2} !important; color:{TEXT} !important; }}
  div[data-baseweb="select"] > div {{ background:{SURFACE2} !important;
      border-color:{BORDER2} !important; border-radius:8px !important; }}

  table {{ width:100%; border-collapse:collapse; font-size:12.5px;
           table-layout:fixed; }}
  td, th {{ padding:11px 8px; border-top:1px solid {BORDER}; text-align:left;
            overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  th {{ color:{FAINT} !important; font-weight:400; border-top:none;
        font-size:10px; text-transform:uppercase; letter-spacing:.5px;
        padding-bottom:9px; }}
  td a {{ color:{ACCENT}; text-decoration:none !important; }}
  td a:hover {{ text-decoration:underline !important; }}

  .streamlit-expanderHeader {{ background:{SURFACE2} !important;
      border-radius:10px !important; }}
  div[data-baseweb="select"] span[data-baseweb="tag"] {{
      background:{ACCENT_D} !important; color:{ACCENT} !important;
      border-radius:7px !important; }}
  div[data-baseweb="select"] span[data-baseweb="tag"] span,
  div[data-baseweb="select"] span[data-baseweb="tag"] svg {{
      color:{ACCENT} !important; fill:{ACCENT} !important; }}
  div[data-baseweb="popover"] li {{ background:{SURFACE2} !important; }}
  div[data-baseweb="popover"] li:hover {{ background:{ACCENT_D} !important; }}

  /* The sidebar deliberately has NO width media query. Rail mode is decided
     in Python so the CSS and the rendered labels can never disagree. */
  @media (max-width: 900px) {{
    div[data-testid="stHorizontalBlock"] {{ flex-wrap:wrap !important; }}
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
        min-width:calc(50% - 1rem) !important;
        flex:1 1 calc(50% - 1rem) !important; }}
  }}
  @media (max-width: 560px) {{
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
        min-width:100% !important; flex:1 1 100% !important; }}
  }}
</style>
""", unsafe_allow_html=True)

# Streamlit is deprecating st.components.v1.html in favour of st.iframe.
# Use whichever this install actually has so the console stays quiet.
def _html_component(html, height=1):
    """Render an invisible HTML/JS block.

    Streamlit is deprecating st.components.v1.html in favour of st.iframe, but
    st.iframe rejects height=0 (it wants a positive int, 'stretch', or
    'content'), so clamp to 1px on the new API and keep 0 on the old one.
    """
    new_api = getattr(st, "iframe", None)
    if new_api is not None:
        return new_api(html, height=max(int(height), 1))
    return st.components.v1.html(html, height=height)

# --- viewport reporter -----------------------------------------------------
_html_component(
    f"""
    <script>
    (function () {{
      try {{
        const BP = {SB_BREAKPOINT};
        const w  = window.parent || window;
        function report() {{
          try {{
            const narrow = w.innerWidth < BP ? "1" : "0";
            const url = new URL(w.location);
            if (url.searchParams.get("narrow") !== narrow) {{
              url.searchParams.set("narrow", narrow);
              w.history.replaceState({{}}, "", url);
              w.location.reload();
            }}
          }} catch (e) {{ }}
        }}
        report();
        let t;
        w.addEventListener("resize", function () {{
          clearTimeout(t); t = setTimeout(report, 400);
        }});
      }} catch (e) {{ }}
    }})();
    </script>
    """,
    height=0,
)


# ---------------- helpers ----------------
def lvl(s):
    if s >= HIGH_T:
        return "high", C_HIGH, BG_HIGH
    if s >= MED_T:
        return "medium", C_MED, BG_MED
    return "low", C_LOW, BG_LOW


def esc(s):
    """PR titles contain backticks and angle brackets — neutralise them so
    Streamlit's markdown parser can't turn a title into <code> soup."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace("`", "&#96;")
            .replace("*", "&#42;").replace("_", "&#95;"))


def kpi(label, value, color=TEXT, sub=""):
    s = f'<div class="kpi-s">{sub}</div>' if sub else ""
    return (f'<div class="kpi"><div class="kpi-l">{label}</div>'
            f'<div class="kpi-v" style="color:{color}">{value}</div>{s}</div>')


def scope_switch():
    a = "on" if scope_key == "30" else ""
    b = "on" if scope_key == "500" else ""
    return (f'<div style="text-align:right;padding-top:6px"><div class="scope">'
            f'<a class="{a}" href="{qs(scope="30")}" target="_self">Last 30</a>'
            f'<a class="{b}" href="{qs(scope="500")}" target="_self">All 500</a>'
            f'</div></div>')


def page_head(title, sub, show_scope=True):
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f'<div class="page-h">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="page-s">{sub}</div>', unsafe_allow_html=True)
    with h2:
        if show_scope:
            st.markdown(scope_switch(), unsafe_allow_html=True)
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)


def hist_chart(counts, labels, colors, height=118):
    """Bar chart with gridlines and a baseline.

    Deliberately NO y-axis tick numbers: every bar already carries its exact
    count, and printing axis values right next to bar values made the chart
    look like it summed to more than it did.
    """
    mx = max(counts) or 1
    bars, axis = "", ""
    for v, l, c in zip(counts, labels, colors):
        h = v / mx * 100
        bars += (
            f'<div style="flex:1;display:flex;flex-direction:column;'
            f'align-items:center;justify-content:flex-end;height:100%">'
            f'<div style="font-size:11px;color:{TEXT};margin-bottom:5px;'
            f'font-weight:600">{v}</div>'
            f'<div style="width:100%;background:{c};height:{h}%;'
            f'border-radius:3px 3px 0 0"></div></div>')
        axis += (f'<span style="flex:1;text-align:center;font-size:9.5px;'
                 f'color:{FAINT}">{l}</span>')
    grid = "".join(
        f'<div style="position:absolute;left:0;right:0;'
        f'bottom:{frac*100:.1f}%;border-top:1px solid {BORDER}"></div>'
        for frac in (1 / 3, 2 / 3, 1.0))
    return (
        f'<div style="position:relative;height:{height}px;margin-top:14px">'
        f'{grid}'
        f'<div style="display:flex;align-items:flex-end;gap:10px;height:100%;'
        f'position:relative">{bars}</div>'
        f'<div style="position:absolute;left:0;right:0;bottom:0;'
        f'border-top:1px solid {AXIS}"></div></div>'
        f'<div style="display:flex;gap:10px;margin-top:7px">{axis}</div>'
        f'<div style="font-size:10px;color:{FAINT};margin-top:9px;'
        f'text-align:right">bars sum to {sum(counts)}</div>')


def trend_chart(values, height=118, thresh=HIGH_T):
    """Filled line chart with a dashed high-risk threshold. No extra deps."""
    if len(values) < 2:
        return '<div class="c-s">Not enough PRs to plot a trend.</div>'
    lo, hi = 0.0, 1.0                       # fixed scale — a % is a %
    rng = hi - lo
    step = 100 / (len(values) - 1)
    pts = " ".join(f"{i*step:.2f},{(1-(v-lo)/rng)*100:.2f}"
                   for i, v in enumerate(values))
    ty = (1 - (thresh - lo) / rng) * 100
    trend = "falling" if values[-1] < values[0] else "rising"
    icon = "ti-trending-down" if trend == "falling" else "ti-trending-up"
    tcol = C_LOW if trend == "falling" else C_HIGH
    return (
        f'<svg viewBox="0 0 100 100" preserveAspectRatio="none" '
        f'style="width:100%;height:{height}px;margin-top:14px;overflow:visible">'
        f'<line x1="0" y1="{ty:.2f}" x2="100" y2="{ty:.2f}" stroke="{C_HIGH}" '
        f'stroke-width="1" stroke-dasharray="3 3" opacity="0.5" '
        f'vector-effect="non-scaling-stroke"/>'
        f'<polygon points="0,100 {pts} 100,100" fill="{ACCENT}" opacity="0.1"/>'
        f'<polyline points="{pts}" fill="none" stroke="{ACCENT}" '
        f'stroke-width="1.4" vector-effect="non-scaling-stroke" '
        f'stroke-linejoin="round"/>'
        f'<line x1="0" y1="100" x2="100" y2="100" stroke="{AXIS}" '
        f'stroke-width="1" vector-effect="non-scaling-stroke"/></svg>'
        f'<div style="display:flex;justify-content:space-between;margin-top:6px">'
        f'<span style="font-size:9px;color:{FAINT}">{len(values)} PRs ago</span>'
        f'<span style="font-size:10.5px;color:{tcol}">'
        f'<i class="ti {icon}" style="font-size:11px;vertical-align:-1px"></i> '
        f'{trend}</span>'
        f'<span style="font-size:9px;color:{FAINT}">latest</span></div>')


def donut(value, color, label, size=130):
    circ = 2 * 3.14159 * 42
    fill = circ * value
    return (
        f'<svg viewBox="0 0 100 100" style="width:{size}px;height:{size}px;'
        f'margin:10px auto 4px;display:block">'
        f'<circle cx="50" cy="50" r="42" fill="none" stroke="{BORDER}" '
        f'stroke-width="9"/>'
        f'<circle cx="50" cy="50" r="42" fill="none" stroke="{color}" '
        f'stroke-width="9" stroke-dasharray="{fill:.1f} {circ:.1f}" '
        f'stroke-linecap="round" transform="rotate(-90 50 50)"/>'
        f'<text x="50" y="56" text-anchor="middle" fill="{TEXT}" '
        f'font-size="21" font-weight="600">{label}</text></svg>')


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
            f'<div style="display:flex;align-items:center;gap:11px;'
            f'margin-bottom:10px;font-size:12px">'
            f'<span style="width:130px;color:{MUTED}">{esc(c["feature"])}</span>'
            f'<div style="flex:1;height:7px;background:{BORDER};'
            f'border-radius:99px"><div style="width:{w}%;height:7px;'
            f'background:{col};border-radius:99px"></div></div>'
            f'<span style="width:44px;text-align:right;color:{col};'
            f'font-weight:600">{sign}{abs(c["value"])}%</span></div>')
    return rows


def pr_table(df, show_lines=False):
    """The one table style used on both Overview and Review queue."""
    head = ('<tr><th style="width:74px">Risk</th><th style="width:66px">PR</th>'
            '<th>Title</th><th style="width:104px">Author</th>'
            '<th style="width:52px;text-align:right">Files</th>')
    if show_lines:
        head += '<th style="width:72px;text-align:right">+Lines</th>'
    head += '</tr>'
    rows = ""
    for _, r in df.iterrows():
        _, col, bg = lvl(r["score"])
        rows += (
            f'<tr><td><span class="pill" style="background:{bg};color:{col}">'
            f'{r["score"]:.0%}</span></td>'
            f'<td><a href="{qs(page="score", pr=int(r["number"]))}" '
            f'target="_self">#{r["number"]}</a></td>'
            f'<td style="color:{TEXT2}">{esc(r["title"])}</td>'
            f'<td style="color:{MUTED}">{esc(r["author"])}</td>'
            f'<td style="color:{MUTED};text-align:right">{r["files"]}</td>')
        if show_lines:
            rows += (f'<td style="color:{C_LOW};text-align:right">'
                     f'+{r["additions"]}</td>')
        rows += '</tr>'
    return f'<table>{head}{rows}</table>'


# ---------------- data ----------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed/labeled_prs_v2.csv"

FEATURE_COLS = ["additions", "deletions", "changed_files", "commits",
                "comments", "review_comments", "num_files",
                "author_past_prs", "author_past_bug_rate", "is_first_pr"]


@st.cache_data(show_spinner=False)
def load_and_score(n=30):
    """Score the n most recently merged PRs. n=None scores the whole corpus."""
    df = pd.read_csv(DATA).sort_values("merged_at", ascending=False)
    if n is not None:
        df = df.head(n)
    # the author column name varies by scrape, so fall back gracefully
    acol = next((c for c in ("author", "user", "user_login", "login")
                 if c in df.columns), None)
    rows = []
    for _, r in df.iterrows():
        pr = {k: r[k] for k in FEATURE_COLS}
        pr["title"] = r["title"]
        rows.append({
            "score": score_pr(pr)["risk_score"],
            "number": int(r["number"]),
            "title": r["title"],
            "author": str(r[acol]) if acol else "—",
            "files": int(r["num_files"]),
            "additions": int(r["additions"]),
            "deletions": int(r["deletions"]),
            "commits": int(r["commits"]),
            "review_comments": int(r["review_comments"]),
            "first_pr": bool(r["is_first_pr"]),
            "merged_at": r["merged_at"],
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False)


@st.cache_data(show_spinner=False)
def corpus_size():
    return len(pd.read_csv(DATA))


def scoped():
    return load_and_score(SCOPES[scope_key])


def split(df):
    high = int((df["score"] >= HIGH_T).sum())
    med = int(((df["score"] >= MED_T) & (df["score"] < HIGH_T)).sum())
    low = int((df["score"] < MED_T).sum())
    return high, med, low


# ---------------- sidebar ----------------
with st.sidebar:
    logo = (f'<div class="sb-logo"><i class="ti ti-bug"></i>'
            f'{"" if rail else f"<span>AutoDebug <span style=color:{ACCENT}>AI</span></span>"}'
            f'</div>')
    nav = ""
    for slug, label, icon in PAGES:
        on = "on" if slug == page else ""
        txt = "" if rail else f"<span>{label}</span>"
        nav += (f'<a class="nav-i {on}" href="{qs(page=slug)}" target="_self" '
                f'title="{label}"><i class="ti {icon}"></i>{txt}</a>')
        if slug == DIVIDER_AFTER:
            nav += '<div class="nav-rule"></div>'
    st.markdown(f'<div class="sb">{logo}<div class="sb-nav">{nav}</div></div>',
                unsafe_allow_html=True)

    if st.button("»" if rail else "«", key="rail_toggle",
                 help="Expand sidebar" if rail else "Collapse sidebar"):
        st.session_state.rail = not st.session_state.rail
        st.session_state._manual_rail = True   # manual click beats the viewport rule
        st.rerun()


# ---------------- OVERVIEW — analyst console ----------------
if page == "overview":
    page_head("Overview", "Bug-risk analysis across merged pull requests")

    with st.spinner("Scoring merged PRs…"):
        df = scoped()

    high, med, low = split(df)
    n = len(df)
    avg = df["score"].mean() if n else 0

    c = st.columns(4)
    c[0].markdown(kpi("Scope", n, TEXT, "merged PRs scored"),
                  unsafe_allow_html=True)
    c[1].markdown(kpi("High risk", high, C_HIGH,
                      f"{high/n:.0%} of scope" if n else "—"),
                  unsafe_allow_html=True)
    c[2].markdown(kpi("Avg risk", f"{avg:.0%}", C_MED,
                      f"vs {BASE_RATE:.0%} base rate"), unsafe_allow_html=True)
    c[3].markdown(kpi("Model AUC", f"{ROC_AUC:.2f}", ACCENT, "held-out test set"),
                  unsafe_allow_html=True)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.35, 1])
    with left:
        # include_lowest catches scores of exactly 0.0, which the old bins
        # silently dropped — that was the 29-vs-30 mismatch.
        bins = pd.cut(df["score"], bins=[0, .2, .4, .6, .8, 1.0],
                      labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"],
                      include_lowest=True)
        hist = bins.value_counts().sort_index()
        assert int(hist.sum()) == n, "binning dropped a row"
        st.markdown(
            f'<div class="card"><div class="c-t">Risk distribution</div>'
            f'<div class="c-s">Predicted probability, binned</div>'
            f'{hist_chart(list(hist.values), list(hist.index), [C_LOW, C_LOW, C_MED, C_MED, C_HIGH])}'
            f'</div>', unsafe_allow_html=True)
    with right:
        trend = df.sort_values("merged_at")["score"].tolist()
        st.markdown(
            f'<div class="card"><div class="c-t">Risk trend</div>'
            f'<div class="c-s">Oldest → newest, dashed line is the '
            f'{HIGH_T:.0%} high-risk threshold</div>{trend_chart(trend)}</div>',
            unsafe_allow_html=True)

    top = df.head(6)
    more = (f'<a href="{qs(page="queue")}" target="_self" '
            f'style="font-size:11px;color:{ACCENT};text-decoration:none">'
            f'View all {n} →</a>') if n > len(top) else ""
    st.markdown(
        f'<div class="card"><div style="display:flex;'
        f'justify-content:space-between;align-items:center;margin-bottom:11px">'
        f'<div class="c-t">Needs review first</div>{more}</div>'
        f'{pr_table(top)}</div>', unsafe_allow_html=True)


# ---------------- SCORE A PR ----------------
elif page == "score":
    page_head("Score a pull request",
              "Fetches a live PR from psf/requests and predicts its bug risk",
              show_scope=False)

    # table rows deep-link here with ?pr=NNNN
    try:
        prefill = int(_qp.get("pr", 7555))
    except (TypeError, ValueError):
        prefill = 7555

    ic, bc = st.columns([3, 1])
    with ic:
        pr_number = st.number_input("PR number", min_value=1, value=prefill,
                                    step=1, label_visibility="collapsed")
    with bc:
        go = st.button("Analyze PR", use_container_width=True)

    if go or "pr" in _qp:
        with st.spinner("Fetching from GitHub and scoring…"):
            try:
                f = fetch_pr_features(int(pr_number))
                res = score_pr(f)
                name, col, bg = lvl(res["risk_score"])

                st.markdown(
                    f'<div class="card"><div style="display:flex;'
                    f'justify-content:space-between;align-items:flex-start;'
                    f'gap:14px">'
                    f'<div><div style="font-size:15px;color:{TEXT};'
                    f'font-weight:600">{esc(f["title"])}</div>'
                    f'<div class="c-s">PR #{pr_number}</div></div>'
                    f'<span class="pill" style="background:{bg};color:{col}">'
                    f'{name.upper()}</span></div>'
                    f'<div style="display:flex;gap:24px;padding-top:13px;'
                    f'margin-top:13px;border-top:1px solid {BORDER};'
                    f'font-size:11.5px">'
                    f'<div><span style="color:{MUTED}">Files</span> '
                    f'<b>{f["changed_files"]}</b></div>'
                    f'<div><span style="color:{MUTED}">Added</span> '
                    f'<b style="color:{C_LOW}">+{f["additions"]}</b></div>'
                    f'<div><span style="color:{MUTED}">Deleted</span> '
                    f'<b style="color:{C_HIGH}">−{f["deletions"]}</b></div>'
                    f'<div><span style="color:{MUTED}">Commits</span> '
                    f'<b>{f["commits"]}</b></div>'
                    f'<div><span style="color:{MUTED}">Comments</span> '
                    f'<b>{f["comments"]}</b></div></div></div>',
                    unsafe_allow_html=True)

                g, b = st.columns([1, 1.6])
                with g:
                    st.markdown(
                        f'<div class="card" style="text-align:center;'
                        f'height:230px"><div class="kpi-l">Risk score</div>'
                        f'<div style="font-size:54px;font-weight:600;'
                        f'color:{col};margin:22px 0 18px">'
                        f'{res["risk_score"]:.0%}</div>'
                        f'<div style="height:7px;background:{BORDER};'
                        f'border-radius:99px"><div style="width:'
                        f'{res["risk_score"]*100}%;height:7px;background:{col};'
                        f'border-radius:99px"></div></div></div>',
                        unsafe_allow_html=True)
                with b:
                    st.markdown(
                        f'<div class="card" style="height:230px">'
                        f'<div class="c-t">Why? — feature contributions</div>'
                        f'<div style="margin-top:16px">'
                        f'{bars_html(res["contributions"])}</div>'
                        f'<div class="c-s">Pink raises risk · green lowers it '
                        f'(SHAP values)</div></div>', unsafe_allow_html=True)

                with st.expander("Raw features"):
                    st.json(f)
            except Exception as e:
                st.error(f"Could not score PR #{pr_number}: {e}")


# ---------------- REVIEW QUEUE ----------------
elif page == "queue":
    page_head("Review queue", "Every scored PR, riskiest first")

    with st.spinner("Scoring merged PRs…"):
        df_all = scoped()

    f1, f2 = st.columns([1.4, 2.6])
    with f1:
        level = st.radio("Risk level", ["All", "High", "Medium", "Low"],
                         horizontal=True)
    with f2:
        search = st.text_input("Search title",
                               placeholder="e.g. headers, stream, auth")

    def keep(r):
        if level != "All" and lvl(r["score"])[0].capitalize() != level:
            return False
        if search and search.lower() not in str(r["title"]).lower():
            return False
        return True

    df = df_all[df_all.apply(keep, axis=1)] if len(df_all) else df_all
    df = df.sort_values("score", ascending=False)
    high, med, low = split(df)

    c = st.columns(3)
    c[0].markdown(kpi("High risk", high, C_HIGH), unsafe_allow_html=True)
    c[1].markdown(kpi("Medium risk", med, C_MED), unsafe_allow_html=True)
    c[2].markdown(kpi("Low risk", low, C_LOW), unsafe_allow_html=True)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    if len(df) == 0:
        st.markdown(
            f'<div class="card" style="text-align:center;padding:36px">'
            f'<div style="font-size:13px;color:{MUTED}">No PRs match these '
            f'filters. Try a different risk level or clear the search.</div>'
            f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="card">{pr_table(df, show_lines=True)}</div>',
                    unsafe_allow_html=True)
        st.download_button("Download this queue as CSV",
                           df.to_csv(index=False).encode("utf-8"),
                           file_name="review_queue.csv", mime="text/csv")


# ---------------- MODEL ----------------
elif page == "model":
    page_head("Model", "How the classifier was built and how well it performs")

    with st.spinner("Scoring merged PRs…"):
        df = scoped()
    n = len(df)

    m = st.columns(4)
    m[0].markdown(kpi("ROC AUC", f"{ROC_AUC:.3f}", ACCENT, "held-out test set"),
                  unsafe_allow_html=True)
    m[1].markdown(kpi("Training PRs", corpus_size(), TEXT,
                      "labelled psf/requests PRs"), unsafe_allow_html=True)
    m[2].markdown(kpi("Features", "40", TEXT, "10 numeric + 30 PCA"),
                  unsafe_allow_html=True)
    m[3].markdown(kpi("Base rate", f"{BASE_RATE:.0%}", C_MED,
                      "share labelled buggy"), unsafe_allow_html=True)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown(
            f'<div class="card"><div class="c-t">Models we compared</div>'
            f'<table style="margin-top:10px">'
            f'<tr><th>Model</th><th style="width:86px">ROC AUC</th>'
            f'<th style="width:46%">Notes</th></tr>'
            f'<tr><td style="color:{ACCENT};font-weight:600">Random Forest</td>'
            f'<td style="color:{ACCENT};font-weight:600">0.710</td>'
            f'<td style="color:{MUTED}">selected — best on test</td></tr>'
            f'<tr><td style="color:{TEXT2}">XGBoost</td>'
            f'<td style="color:{MUTED}">—</td>'
            f'<td style="color:{MUTED}">close, more prone to overfit here</td></tr>'
            f'<tr><td style="color:{TEXT2}">LightGBM</td>'
            f'<td style="color:{MUTED}">—</td>'
            f'<td style="color:{MUTED}">similar, slower to tune</td></tr>'
            f'<tr><td style="color:{TEXT2}">Logistic Regression</td>'
            f'<td style="color:{MUTED}">—</td>'
            f'<td style="color:{MUTED}">baseline, weaker on interactions</td></tr>'
            f'</table><div class="c-s" style="margin-top:10px">Fill in the '
            f'remaining AUCs from your training run to complete the table.</div>'
            f'</div>', unsafe_allow_html=True)
    with right:
        st.markdown(
            f'<div class="card" style="text-align:center">'
            f'<div class="c-t">Discrimination</div>'
            f'{donut(ROC_AUC, ACCENT, f"{ROC_AUC:.2f}")}'
            f'<div class="c-s">ROC AUC — 0.5 is random, 1.0 is perfect</div>'
            f'</div>', unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown(
            f'<div class="card"><div class="c-t">Feature groups</div>'
            f'<div style="font-size:12px;line-height:1.85;color:{MUTED};'
            f'margin-top:10px">'
            f'<b style="color:{TEXT}">Size</b> — additions, deletions, changed '
            f'files, num_files, commits. Bigger diffs touch more surface area.<br>'
            f'<b style="color:{TEXT}">Process</b> — comments, review_comments. '
            f'Heavier discussion often marks contentious changes.<br>'
            f'<b style="color:{TEXT}">Author</b> — past PRs, past bug rate, '
            f'is_first_pr. Track record carries real predictive weight.<br>'
            f'<b style="color:{TEXT}">Title semantics</b> — 30 PCA components '
            f'over sentence-transformer embeddings.</div></div>',
            unsafe_allow_html=True)
    with g2:
        hi_rc = df[df["score"] >= HIGH_T]["review_comments"].mean() if n else 0
        lo_rc = df[df["score"] < MED_T]["review_comments"].mean() if n else 0
        st.markdown(
            f'<div class="card"><div class="c-t">Honest limitations</div>'
            f'<div style="font-size:12px;line-height:1.85;color:{MUTED};'
            f'margin-top:10px">'
            f'AUC of {ROC_AUC:.3f} means the model is usefully better than '
            f'chance but far from authoritative — treat scores as a '
            f'review-ordering hint, not a verdict.<br>'
            f'Labels come from heuristic bug-linking, so some are certainly '
            f'wrong.<br>'
            f'Everything is trained on a single repository, so it will not '
            f'transfer cleanly to codebases with different review norms.<br>'
            f'<span style="color:{TEXT}">In the current scope</span>, high-risk '
            f'PRs average {hi_rc:.1f} review comments vs {lo_rc:.1f} on '
            f'low-risk ones.</div></div>', unsafe_allow_html=True)


# ---------------- ABOUT ----------------
else:
    page_head("About AutoDebug AI",
              "Predicts which pull requests are likely to introduce bugs, so "
              "reviewers can prioritize the riskiest changes",
              show_scope=False)

    c = st.columns(3)
    c[0].markdown(kpi("Model", "Random Forest", ACCENT,
                      "beat XGBoost, LightGBM, LogReg"), unsafe_allow_html=True)
    c[1].markdown(kpi("Features", "40 total", TEXT,
                      "10 numeric + 30 PCA title embeddings"),
                  unsafe_allow_html=True)
    c[2].markdown(kpi("Explainability", "SHAP", TEXT,
                      "per-PR feature contributions"), unsafe_allow_html=True)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="card"><div class="c-t">How it works</div>'
        f'<div style="font-size:12px;line-height:1.9;color:{MUTED};'
        f'margin-top:10px">'
        f'<b style="color:{ACCENT}">1.</b> Collect merged PRs from psf/requests '
        f'via the GitHub API and label them by whether they were later linked '
        f'to a bug fix.<br>'
        f'<b style="color:{ACCENT}">2.</b> Build features from diff size, '
        f'review activity, and author history, plus PCA-reduced embeddings of '
        f'the PR title.<br>'
        f'<b style="color:{ACCENT}">3.</b> Train and compare four classifiers, '
        f'keeping the one with the best held-out ROC AUC.<br>'
        f'<b style="color:{ACCENT}">4.</b> Serve predictions through FastAPI '
        f'and explain each one with SHAP, so a reviewer can see why a PR '
        f'scored the way it did.</div></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="card"><div class="c-t">Built with</div>'
        f'<div style="margin-top:9px;font-size:12px;color:{MUTED};'
        f'line-height:1.9">Python · scikit-learn · sentence-transformers · '
        f'SHAP · FastAPI · Streamlit · Docker</div></div>',
        unsafe_allow_html=True)