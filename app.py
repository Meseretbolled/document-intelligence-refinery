from __future__ import annotations

# Load .env before anything else — Streamlit doesn't use src/main.py entry point
from dotenv import load_dotenv
from pathlib import Path as _Path
load_dotenv(dotenv_path=_Path(__file__).resolve().parent / ".env", override=True)

import json
import time
from pathlib import Path
from typing import Optional

import streamlit as st

from src.service.refinery_service import run_refinery_on_pdf, get_query_agent
from src.settings import settings

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Document Intelligence Refinery",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — dark industrial / terminal aesthetic
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

:root {
  --bg:        #0d0f11;
  --surface:   #13161a;
  --surface2:  #1c2028;
  --border:    #2a2f38;
  --border2:   #353d4a;
  --accent:    #e8ff47;
  --accent2:   #47ffc8;
  --accent3:   #ff6b35;
  --text:      #e2e8f0;
  --text-muted:#7a8699;
  --text-dim:  #4a5568;
  --danger:    #ff4757;
  --warning:   #ffa502;
  --success:   #2ed573;
  --mono:      'Space Mono', monospace;
  --display:   'Syne', sans-serif;
  --body:      'DM Sans', sans-serif;
  --radius:    6px;
  --radius-lg: 12px;
}

.stApp { background: var(--bg); color: var(--text); font-family: var(--body); }
section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

h1, h2, h3, h4 { font-family: var(--display) !important; }
code, pre { font-family: var(--mono) !important; }

/* Sidebar */
.stSidebar .stMarkdown p { color: var(--text-muted); font-size: 0.78rem; font-family: var(--mono); text-transform: uppercase; letter-spacing: 0.08em; }
.stSidebar label { color: var(--text) !important; font-family: var(--body) !important; }
.stSidebar .stSelectbox label, .stSidebar .stTextInput label { color: var(--text-muted) !important; font-family: var(--mono) !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.06em; }

/* Inputs */
.stTextInput input, .stTextArea textarea {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text) !important;
  font-family: var(--body) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(232,255,71,0.12) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
  background: var(--surface2) !important;
  border: 1px dashed var(--border2) !important;
  border-radius: var(--radius-lg) !important;
  padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploader"] label { color: var(--text-muted) !important; font-family: var(--mono) !important; font-size: 0.8rem !important; }

/* Primary button */
.stButton > button {
  background: var(--accent) !important;
  color: #0d0f11 !important;
  border: none !important;
  border-radius: var(--radius) !important;
  font-family: var(--display) !important;
  font-weight: 700 !important;
  font-size: 0.9rem !important;
  letter-spacing: 0.04em !important;
  padding: 0.6rem 1.5rem !important;
  transition: all 0.15s !important;
}
.stButton > button:hover {
  background: #f5ff70 !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 20px rgba(232,255,71,0.25) !important;
}

/* Download button */
.stDownloadButton > button {
  background: var(--surface2) !important;
  color: var(--accent2) !important;
  border: 1px solid var(--border) !important;
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
  padding: 0.4rem 0.9rem !important;
  border-radius: var(--radius) !important;
}
.stDownloadButton > button:hover {
  border-color: var(--accent2) !important;
  box-shadow: 0 0 12px rgba(71,255,200,0.15) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
  padding: 0 0.5rem !important;
  border-radius: var(--radius-lg) var(--radius-lg) 0 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-muted) !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  padding: 0.75rem 1.1rem !important;
  border-bottom: 2px solid transparent !important;
  transition: all 0.15s !important;
}
.stTabs [aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 var(--radius-lg) var(--radius-lg) !important;
  padding: 1.5rem !important;
}

/* Metrics */
[data-testid="stMetric"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  padding: 1rem 1.25rem !important;
}
[data-testid="stMetric"] label {
  font-family: var(--mono) !important;
  font-size: 0.65rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--display) !important;
  font-size: 1.5rem !important;
  font-weight: 800 !important;
  color: var(--accent) !important;
}

/* Expanders */
.streamlit-expanderHeader {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
  color: var(--text-muted) !important;
}
.streamlit-expanderContent {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 var(--radius) var(--radius) !important;
}

/* JSON viewer */
.stJson { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }
.stProgress > div > div { background: var(--accent) !important; }

/* Selectbox */
[data-baseweb="select"] > div {
  background: var(--surface2) !important;
  border-color: var(--border) !important;
  border-radius: var(--radius) !important;
}
[data-baseweb="popover"] { background: var(--surface2) !important; border: 1px solid var(--border2) !important; }
[data-baseweb="option"] { background: var(--surface2) !important; color: var(--text) !important; }
[data-baseweb="option"]:hover { background: var(--surface) !important; }

/* ── Custom utility classes ── */
.tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 3px;
  font-family: var(--mono);
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.tag-a    { background: rgba(46,213,115,0.12); color: #2ed573; border: 1px solid rgba(46,213,115,0.3); }
.tag-b    { background: rgba(71,255,200,0.10); color: #47ffc8; border: 1px solid rgba(71,255,200,0.3); }
.tag-c    { background: rgba(255,107,53,0.12); color: #ff6b35; border: 1px solid rgba(255,107,53,0.3); }
.tag-ok   { background: rgba(46,213,115,0.10); color: #2ed573; border: 1px solid rgba(46,213,115,0.3); }
.tag-warn { background: rgba(255,165,2,0.10);  color: #ffa502; border: 1px solid rgba(255,165,2,0.3); }
.tag-err  { background: rgba(255,71,87,0.10);  color: #ff4757; border: 1px solid rgba(255,71,87,0.3); }
.tag-info { background: rgba(232,255,71,0.08); color: #e8ff47; border: 1px solid rgba(232,255,71,0.2); }

.pill-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 0.4rem 0; }

.stat-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.75rem 1rem;
  margin-bottom: 0.5rem;
}
.stat-card .stat-label {
  font-family: var(--mono);
  font-size: 0.63rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 3px;
}
.stat-card .stat-value {
  font-family: var(--display);
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text);
}

.prov-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent2);
  border-radius: var(--radius);
  padding: 0.7rem 1rem;
  margin-bottom: 0.5rem;
  font-family: var(--mono);
  font-size: 0.75rem;
}
.prov-card .prov-doc  { color: var(--accent2); font-weight: 700; margin-bottom: 4px; }
.prov-card .prov-meta { color: var(--text-muted); display: flex; gap: 16px; flex-wrap: wrap; }
.prov-card .prov-hash { color: var(--text-dim); font-size: 0.62rem; margin-top: 4px; word-break: break-all; }

.trace-step {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.55rem 0.85rem;
  margin-bottom: 4px;
  font-family: var(--mono);
  font-size: 0.72rem;
  display: flex;
  align-items: flex-start;
  gap: 10px;
}
.trace-step .step-key   { color: var(--accent); min-width: 90px; font-weight: 700; flex-shrink: 0; }
.trace-step .step-value { color: var(--text-muted); flex: 1; word-break: break-all; line-height: 1.5; }

.ldu-row {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.6rem 0.9rem;
  margin-bottom: 4px;
  display: flex;
  align-items: flex-start;
  gap: 10px;
  transition: border-color 0.15s;
}
.ldu-row:hover { border-color: var(--border2); }
.ldu-row .ldu-type  { min-width: 56px; flex-shrink: 0; }
.ldu-row .ldu-meta  { font-family: var(--mono); font-size: 0.63rem; color: var(--text-muted); min-width: 170px; flex-shrink: 0; line-height: 1.7; }
.ldu-row .ldu-text  { font-size: 0.82rem; color: var(--text); flex: 1; line-height: 1.55; }

.block-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.7rem 1rem;
  margin-bottom: 0.4rem;
  transition: border-color 0.15s;
}
.block-card:hover { border-color: var(--border2); }
.block-card .bc-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 5px;
  font-family: var(--mono);
  font-size: 0.67rem;
  color: var(--text-muted);
}
.block-card .bc-text { font-size: 0.83rem; line-height: 1.55; color: var(--text); }

.snippet-box {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius);
  padding: 0.75rem 1rem;
  font-size: 0.85rem;
  line-height: 1.6;
  margin-bottom: 0.5rem;
  color: var(--text);
}
.snippet-box .snip-idx {
  font-family: var(--mono);
  font-size: 0.62rem;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 5px;
}

.section-nav-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.7rem 1rem;
  margin-bottom: 0.4rem;
}
.section-nav-card .snc-title { font-family: var(--display); font-weight: 600; color: var(--text); font-size: 0.88rem; }
.section-nav-card .snc-meta  { font-family: var(--mono); font-size: 0.62rem; color: var(--text-muted); margin-top: 4px; display: flex; gap: 12px; flex-wrap: wrap; }
.section-nav-card .snc-score { color: var(--accent); font-weight: 700; }
.section-nav-card .snc-summary { font-size: 0.8rem; color: var(--text-muted); margin-top: 5px; line-height: 1.5; }

.empty-state {
  text-align: center;
  padding: 3rem 2rem;
  color: var(--text-dim);
  font-family: var(--mono);
  font-size: 0.78rem;
}
.empty-state .es-icon { font-size: 2.2rem; margin-bottom: 0.75rem; opacity: 0.35; display:block; }
.empty-state p { margin: 0; text-transform: uppercase; letter-spacing: 0.08em; }

.conf-bar-wrap { display: flex; align-items: center; gap: 8px; font-family: var(--mono); font-size: 0.7rem; }
.conf-bar-outer { flex: 1; height: 5px; background: var(--surface); border-radius: 3px; overflow: hidden; }
.conf-bar-inner { height: 100%; border-radius: 3px; }

.verdict-verified     { background: rgba(46,213,115,0.07); border: 1px solid rgba(46,213,115,0.35); border-radius: var(--radius); padding: 1rem 1.25rem; }
.verdict-unverifiable { background: rgba(255,165,2,0.07);  border: 1px solid rgba(255,165,2,0.30);  border-radius: var(--radius); padding: 1rem 1.25rem; }
.verdict-not_found    { background: rgba(255,71,87,0.07);  border: 1px solid rgba(255,71,87,0.30);  border-radius: var(--radius); padding: 1rem 1.25rem; }

.divider { border: none; border-top: 1px solid var(--border); margin: 1.25rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_tag(ct: str) -> str:
    colors = {
        "text":   "#7a8699",
        "header": "#e8ff47",
        "table":  "#47ffc8",
        "figure": "#ff6b35",
        "list":   "#c084fc",
    }
    c = colors.get(ct, "#7a8699")
    return (
        f'<span style="background:rgba(255,255,255,0.04);border:1px solid {c}55;'
        f'color:{c};border-radius:3px;padding:1px 7px;font-family:var(--mono);'
        f'font-size:0.63rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em">'
        f'{ct}</span>'
    )


def _strategy_tag(s: str) -> str:
    cls   = {"A": "tag-a", "B": "tag-b", "C": "tag-c"}.get(s, "tag-info")
    label = {"A": "Strategy A · pdfplumber", "B": "Strategy B · Docling", "C": "Strategy C · VLM"}.get(s, s)
    return f'<span class="tag {cls}">{label}</span>'


def _conf_bar(val: float) -> str:
    pct = round(val * 100)
    c = "#2ed573" if pct >= 80 else "#ffa502" if pct >= 60 else "#ff4757"
    return (
        f'<div class="conf-bar-wrap">'
        f'<div class="conf-bar-outer"><div class="conf-bar-inner" style="width:{pct}%;background:{c}"></div></div>'
        f'<span style="color:{c};font-weight:700">{pct}%</span>'
        f'</div>'
    )


def _verdict_icon(v: str) -> str:
    return {"verified": "🟢", "unverifiable": "🟡", "not_found": "🔴"}.get(v, "⚪")


def _list_pdfs(d: Path) -> list[Path]:
    return sorted(d.rglob("*.pdf")) if d.exists() else []


def _get_attr(obj, key: str, default=""):
    """Works for both dicts and Pydantic/dataclass objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem">
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:#e2e8f0;letter-spacing:-0.01em">⚗️ Refinery</div>
      <div style="font-family:'Space Mono',monospace;font-size:0.58rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;margin-top:2px">Document Intelligence v1.0</div>
    </div>
    <hr style="border:none;border-top:1px solid #2a2f38;margin:0.5rem 0 1rem">
    <p style="margin-bottom:6px">PDF Source</p>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    raw_dir_default = settings.project_root / "data" / "raw" / "data"
    local_dir = st.text_input(
        "Local PDF directory",
        value=str(raw_dir_default),
        label_visibility="collapsed",
        placeholder="Path to PDF folder…",
    )

    st.markdown('<p style="margin:8px 0 4px">Or pick from folder</p>', unsafe_allow_html=True)
    local_pdfs = _list_pdfs(Path(local_dir))
    selected_local = st.selectbox(
        "Local PDF",
        options=[""] + [str(p) for p in local_pdfs[:200]],
        index=0,
        label_visibility="collapsed",
        format_func=lambda x: Path(x).name if x else "— select —",
    )

    st.markdown('<hr style="border:none;border-top:1px solid #2a2f38;margin:1rem 0">', unsafe_allow_html=True)
    run_btn = st.button("🚀  Run Refinery", use_container_width=True)

    st.markdown('<hr style="border:none;border-top:1px solid #2a2f38;margin:1rem 0">', unsafe_allow_html=True)
    st.markdown('<p>Pipeline Stages</p>', unsafe_allow_html=True)
    for num, name in [("01","Triage Agent"),("02","Structure Extraction"),("03","Semantic Chunking"),("04","PageIndex Builder"),("05","Query Interface")]:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0">'
            f'<span style="font-family:var(--mono);font-size:0.63rem;color:#4a5568">{num}</span>'
            f'<span style="font-size:0.8rem;color:#7a8699">{name}</span></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:2px">
  <span style="font-size:1.5rem;filter:drop-shadow(0 0 8px rgba(232,255,71,0.4))">⚗️</span>
  <h1 style="font-family:'Syne',sans-serif!important;font-size:1.55rem!important;font-weight:800!important;
             color:#e2e8f0!important;letter-spacing:-0.02em!important;margin:0!important;padding:0!important;line-height:1.1!important">
    Document Intelligence <span style="color:#e8ff47">Refinery</span>
  </h1>
</div>
<div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#4a5568;text-transform:uppercase;
            letter-spacing:0.1em;margin-bottom:1.5rem;padding-left:44px">
  Agentic · Multi-Strategy · Provenance-Preserving Extraction
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RESOLVE PDF PATH
# ─────────────────────────────────────────────────────────────────────────────

pdf_to_process: Optional[Path] = None

if uploaded is not None:
    temp_dir = settings.project_root / ".refinery" / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    pdf_to_process = temp_dir / uploaded.name
    pdf_to_process.write_bytes(uploaded.getbuffer())
elif selected_local:
    pdf_to_process = Path(selected_local)


# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────

if not run_btn and "refinery_out" not in st.session_state:
    st.markdown("""
    <div class="empty-state">
      <span class="es-icon">⚗️</span>
      <p>Upload or select a PDF · then click Run Refinery</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, label, desc in [
        (c1, "Stage 1–2 · Extraction", "Classifies document and routes to the correct strategy: A → B → C with confidence gates"),
        (c2, "Stage 3–4 · Structure",  "Builds semantic LDUs with 5 chunking rules + hierarchical PageIndex section tree"),
        (c3, "Stage 5 · Query",        "Natural language query with provenance chain · PageIndex navigation · Claim audit"),
    ]:
        with col:
            st.markdown(
                f'<div class="stat-card"><div class="stat-label">{label}</div>'
                f'<div style="font-size:0.82rem;color:#7a8699;margin-top:3px;line-height:1.5">{desc}</div></div>',
                unsafe_allow_html=True,
            )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    if pdf_to_process is None:
        st.error("No PDF selected. Upload a file or choose one from the local folder.")
        st.stop()

    progress_bar = st.progress(0, text="Initialising pipeline…")
    t0 = time.time()

    _stages = [
        (0.18, "Stage 1 · Triage — classifying document…"),
        (0.38, "Stage 2 · Extraction — routing strategy…"),
        (0.58, "Stage 3 · Semantic Chunking — building LDUs…"),
        (0.78, "Stage 4 · PageIndex — constructing section tree…"),
        (0.93, "Stage 5 · Query Agent — ingesting vectors + facts…"),
    ]

    import threading

    _result, _error = {}, {}

    def _run_pipeline():
        try:
            _result["out"] = run_refinery_on_pdf(str(pdf_to_process))
        except Exception as e:
            _error["err"] = e

    t = threading.Thread(target=_run_pipeline, daemon=True)
    t.start()

    _si = 0
    while t.is_alive():
        if _si < len(_stages):
            progress_bar.progress(_stages[_si][0], text=_stages[_si][1])
            _si += 1
        time.sleep(1.1)
        t.join(timeout=0)

    t.join()
    progress_bar.progress(1.0, text="Pipeline complete ✓")
    time.sleep(0.25)
    progress_bar.empty()

    if "err" in _error:
        st.error(f"Pipeline error: {_error['err']}")
        st.stop()

    st.session_state["refinery_out"]     = _result["out"]
    st.session_state["refinery_pdf"]     = str(pdf_to_process)
    st.session_state["refinery_elapsed"] = time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────────────────────────────────────

if "refinery_out" not in st.session_state:
    st.stop()

out          = st.session_state["refinery_out"]
pdf_path_str = st.session_state.get("refinery_pdf", "")
elapsed      = st.session_state.get("refinery_elapsed", 0.0)

profile    = out.profile    or {}
extracted  = out.extracted  or {}
ldus       = out.ldus       or []
page_idx   = out.page_index or {}
meta       = (extracted.get("meta") or {}) if isinstance(extracted, dict) else {}
violations = out.chunk_violations or []

strategy   = (extracted.get("strategy_used", "?") if isinstance(extracted, dict) else "?")
conf       = float(extracted.get("confidence", 0.0) if isinstance(extracted, dict) else 0.0)
n_blocks   = len(extracted.get("blocks") or [] if isinstance(extracted, dict) else [])
n_ldus     = len(ldus)
n_pages    = (page_idx.get("page_count") if isinstance(page_idx, dict) else 0) or profile.get("page_count", "?")
n_sections = len(page_idx.get("sections", []) if isinstance(page_idx, dict) else [])


# ── Summary banner ────────────────────────────────────────────────────────────

doc_name = Path(pdf_path_str).name if pdf_path_str else ""
st.markdown(
    f'<div style="display:flex;align-items:center;justify-content:space-between;'
    f'background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
    f'padding:0.85rem 1.25rem;margin-bottom:1.25rem;flex-wrap:wrap;gap:8px">'
    f'<div style="display:flex;align-items:center;gap:10px">'
    f'  <span style="font-family:var(--mono);font-size:0.68rem;color:var(--text-muted)">Processed</span>'
    f'  <span style="font-family:var(--mono);font-size:0.68rem;color:var(--accent)">{doc_name}</span>'
    f'</div>'
    f'<div style="font-family:var(--mono);font-size:0.65rem;color:var(--text-dim)">⏱ {elapsed:.1f}s</div>'
    f'</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Strategy",   strategy)
m2.metric("Confidence", f"{conf:.0%}")
m3.metric("Pages",      str(n_pages))
m4.metric("Blocks",     str(n_blocks))
m5.metric("LDUs",       str(n_ldus))
m6.metric("Sections",   str(n_sections))

viol_tag = (
    f'<span class="tag tag-warn">⚠ {len(violations)} violation(s)</span>'
    if violations else
    '<span class="tag tag-ok">✓ 0 violations · all chunking rules passed</span>'
)
st.markdown(f'<div style="margin:0.6rem 0 0.75rem">{viol_tag}</div>', unsafe_allow_html=True)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⬡  Profile",
    "⬡  Extracted",
    "⬡  LDUs",
    "⬡  Page Index",
    "⬡  Query",
    "⬡  Audit",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DOCUMENT PROFILE
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("##### Document Classification")

        fields = [
            ("doc_id",               profile.get("doc_id", "—") if isinstance(profile, dict) else "—"),
            ("origin_type",          profile.get("origin_type", "—") if isinstance(profile, dict) else "—"),
            ("layout_complexity",    profile.get("layout_complexity", "—") if isinstance(profile, dict) else "—"),
            ("language",             profile.get("language", "—") if isinstance(profile, dict) else "—"),
            ("language_confidence",  f"{profile.get('language_confidence', 0):.0%}" if isinstance(profile, dict) else "—"),
            ("domain_hint",          profile.get("domain_hint", "—") if isinstance(profile, dict) else "—"),
            ("cost_tier",            profile.get("cost_tier", "—") if isinstance(profile, dict) else "—"),
            ("page_count",           str(profile.get("page_count", "—") if isinstance(profile, dict) else "—")),
            ("avg_chars/page",       f"{profile.get('avg_text_chars_per_page', 0):.0f}" if isinstance(profile, dict) else "—"),
            ("avg_image_ratio",      f"{profile.get('avg_image_area_ratio', 0):.3f}" if isinstance(profile, dict) else "—"),
        ]
        for lbl, val in fields:
            st.markdown(
                f'<div class="trace-step"><span class="step-key">{lbl}</span>'
                f'<span class="step-value">{val}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "↓ profile.json",
            data=json.dumps(profile, indent=2),
            file_name=f"{profile.get('doc_id','doc') if isinstance(profile,dict) else 'doc'}_profile.json",
            mime="application/json",
        )

    with right:
        st.markdown("##### Decision Trace")
        trace = meta.get("decision_trace") or [] if isinstance(meta, dict) else []
        if trace:
            for step in trace:
                if not isinstance(step, dict):
                    continue
                name  = step.get("step", "?")
                rest  = {k: v for k, v in step.items() if k != "step" and v is not None}
                details = "  ·  ".join(f"{k}={v}" for k, v in list(rest.items())[:4])
                st.markdown(
                    f'<div class="trace-step"><span class="step-key">{name}</span>'
                    f'<span class="step-value">{details}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="empty-state" style="padding:1.5rem">'
                '<span class="es-icon">🔍</span><p>No trace data</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("##### Pipeline Notes")
        notes = out.notes or ""
        if notes:
            st.markdown(
                f'<div class="snippet-box"><div class="snip-idx">notes</div>{notes}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span style="font-family:var(--mono);font-size:0.72rem;color:var(--text-dim)">No notes.</span>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXTRACTED BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    blocks_raw = (extracted.get("blocks") or []) if isinstance(extracted, dict) else []

    if not blocks_raw:
        st.markdown(
            '<div class="empty-state"><span class="es-icon">📄</span><p>No blocks extracted</p></div>',
            unsafe_allow_html=True,
        )
    else:
        btypes: dict[str, int] = {}
        for b in blocks_raw:
            bt = (b.get("block_type", "text") if isinstance(b, dict) else getattr(b, "block_type", "text"))
            btypes[bt] = btypes.get(bt, 0) + 1

        strat_html = _strategy_tag(strategy)
        conf_html  = _conf_bar(conf)
        pills_html = " ".join(f'<span class="tag tag-info">{k} · {v}</span>' for k, v in sorted(btypes.items()))

        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:1rem">'
            f'{strat_html}'
            f'<div style="flex:1;max-width:180px">{conf_html}</div>'
            f'<div class="pill-row">{pills_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        max_show = st.slider("Blocks to display", 10, min(500, len(blocks_raw)), min(60, len(blocks_raw)), step=10, key="block_slider")

        for i, b in enumerate(blocks_raw[:max_show]):
            if isinstance(b, dict):
                btype, text, prov = b.get("block_type","text"), b.get("text",""), b.get("provenance") or {}
            else:
                btype, text, prov = getattr(b,"block_type","text"), getattr(b,"text","") or "", {}
            spans = prov.get("spans", []) if isinstance(prov, dict) else []
            page  = spans[0].get("page","?") if spans else "?"
            h     = (prov.get("content_hash","") if isinstance(prov,dict) else "")[:14]
            tag   = _chunk_tag(btype)
            text  = text or ""
            st.markdown(
                f'<div class="block-card">'
                f'<div class="bc-header">{tag} &nbsp;pg {page}'
                f'  <span style="color:var(--text-dim);font-size:0.6rem">{h}…</span></div>'
                f'<div class="bc-text">{text[:360]}{"…" if len(text)>360 else ""}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if len(blocks_raw) > max_show:
            st.markdown(
                f'<div style="text-align:center;font-family:var(--mono);font-size:0.7rem;color:var(--text-dim);padding:0.5rem">'
                f'+ {len(blocks_raw)-max_show} more blocks not shown</div>',
                unsafe_allow_html=True,
            )

    st.download_button(
        "↓ extracted.json",
        data=json.dumps(extracted, indent=2),
        file_name=f"{profile.get('doc_id','doc') if isinstance(profile,dict) else 'doc'}_extracted.json",
        mime="application/json",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LDUs
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    if not ldus:
        st.markdown(
            '<div class="empty-state"><span class="es-icon">🧩</span><p>No LDUs produced</p></div>',
            unsafe_allow_html=True,
        )
    else:
        ltype_counts: dict[str, int] = {}
        for l in ldus:
            ct = (l.get("chunk_type","text") if isinstance(l,dict) else getattr(l,"chunk_type","text"))
            ltype_counts[ct] = ltype_counts.get(ct, 0) + 1

        pills = " ".join(f'<span class="tag tag-info">{k} · {v}</span>' for k, v in sorted(ltype_counts.items()))
        st.markdown(f'<div class="pill-row" style="margin-bottom:1rem">{pills}</div>', unsafe_allow_html=True)

        if violations:
            with st.expander(f"⚠  {len(violations)} chunk violation(s)"):
                for v in violations:
                    st.markdown(
                        f'<span style="font-family:var(--mono);font-size:0.72rem;color:var(--danger)">{v}</span><br>',
                        unsafe_allow_html=True,
                    )

        max_ldu = st.slider("LDUs to display", 10, min(300, len(ldus)), min(80, len(ldus)), step=10, key="ldu_slider")

        for ldu in ldus[:max_ldu]:
            if isinstance(ldu, dict):
                ct      = ldu.get("chunk_type","text")
                content = ldu.get("content","")
                pages   = ldu.get("page_refs",[])
                section = ldu.get("parent_section") or ""
                h       = ldu.get("content_hash","")[:12]
                tokens  = ldu.get("token_count", 0)
            else:
                ct      = getattr(ldu,"chunk_type","text")
                content = getattr(ldu,"content","") or ""
                pages   = getattr(ldu,"page_refs",[])
                section = getattr(ldu,"parent_section","") or ""
                h       = getattr(ldu,"content_hash","")[:12]
                tokens  = getattr(ldu,"token_count",0)

            page = pages[0] if pages else "?"
            sec_short = (section[:42]+"…") if len(section)>42 else section

            st.markdown(
                f'<div class="ldu-row">'
                f'<div class="ldu-type">{_chunk_tag(ct)}</div>'
                f'<div class="ldu-meta">pg {page} · {tokens} tok<br>'
                f'<span style="color:var(--text-dim)">{h}…</span><br>'
                f'<span title="{section}" style="color:var(--text-dim)">{sec_short}</span>'
                f'</div>'
                f'<div class="ldu-text">{content[:300]}{"…" if len(content)>300 else ""}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if len(ldus) > max_ldu:
            st.markdown(
                f'<div style="text-align:center;font-family:var(--mono);font-size:0.7rem;color:var(--text-dim);padding:0.5rem">'
                f'+ {len(ldus)-max_ldu} more LDUs not shown</div>',
                unsafe_allow_html=True,
            )

    st.download_button(
        "↓ ldu.json",
        data=json.dumps(ldus, indent=2),
        file_name=f"{profile.get('doc_id','doc') if isinstance(profile,dict) else 'doc'}_ldu.json",
        mime="application/json",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PAGE INDEX
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    sections   = (page_idx.get("sections", []) if isinstance(page_idx, dict) else [])
    flat_pages = (page_idx.get("root", [])     if isinstance(page_idx, dict) else [])

    tree_col, pages_col = st.columns([3, 2], gap="large")

    with tree_col:
        st.markdown("##### Section Tree")
        if not sections:
            st.markdown(
                '<div class="empty-state" style="padding:1.5rem">'
                '<span class="es-icon">🌲</span><p>No sections detected</p></div>',
                unsafe_allow_html=True,
            )
        else:
            def _render_section(node: dict, depth: int = 0) -> None:
                if not isinstance(node, dict):
                    return
                indent   = depth * 16
                title    = str(node.get("title","?"))[:60]
                ps, pe   = node.get("page_start","?"), node.get("page_end","?")
                level    = node.get("level", 1)
                summary  = str(node.get("summary","") or "")
                entities = node.get("key_entities") or []
                children = node.get("child_sections") or []
                n_l      = len(node.get("ldu_ids") or [])
                ent_str  = " · ".join(str(e) for e in entities[:3])

                st.markdown(
                    f'<div class="section-nav-card" style="margin-left:{indent}px">'
                    f'<div class="snc-title">'
                    f'<span style="color:var(--accent);font-family:var(--mono);font-size:0.6rem">L{level} </span>{title}'
                    f'</div>'
                    f'<div class="snc-meta">'
                    f'<span>pp {ps}–{pe}</span><span>{n_l} LDUs</span>'
                    f'{"<span>"+ent_str+"</span>" if ent_str else ""}'
                    f'</div>'
                    f'{"<div class=snc-summary>"+summary[:160]+"</div>" if summary else ""}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                for child in children:
                    _render_section(child, depth + 1)

            for sec in sections:
                _render_section(sec)

    with pages_col:
        st.markdown("##### Flat Page Index")
        if not flat_pages:
            st.markdown(
                '<div class="empty-state" style="padding:1.5rem">'
                '<span class="es-icon">📋</span><p>No pages indexed</p></div>',
                unsafe_allow_html=True,
            )
        else:
            for pg in flat_pages:
                if not isinstance(pg, dict):
                    continue
                pnum   = pg.get("page","?")
                items  = pg.get("items") or []
                dtypes = pg.get("data_types_present") or []
                chars  = pg.get("char_count", 0)
                dtype_pills = " ".join(_chunk_tag(d) for d in dtypes)
                st.markdown(
                    f'<div class="stat-card" style="padding:0.5rem 0.8rem">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<span style="font-family:var(--mono);font-size:0.7rem;color:var(--accent)">pg {pnum}</span>'
                    f'<span style="font-family:var(--mono);font-size:0.6rem;color:var(--text-dim)">{len(items)} items · {chars} ch</span>'
                    f'</div>'
                    f'<div class="pill-row" style="margin-top:4px">{dtype_pills}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.download_button(
        "↓ page_index.json",
        data=json.dumps(page_idx, indent=2),
        file_name=f"{profile.get('doc_id','doc') if isinstance(profile,dict) else 'doc'}_page_index.json",
        mime="application/json",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — QUERY INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("##### Natural Language Query")
    st.markdown(
        '<div style="font-family:var(--mono);font-size:0.7rem;color:var(--text-muted);margin-bottom:0.75rem">'
        'pageindex_navigate &nbsp;→&nbsp; semantic_search &nbsp;→&nbsp; structured_query'
        '</div>',
        unsafe_allow_html=True,
    )

    q_col, btn_col = st.columns([5, 1], gap="small")
    with q_col:
        question = st.text_input(
            "Question",
            placeholder="What is the total revenue reported?",
            label_visibility="collapsed",
            key="query_input",
        )
    with btn_col:
        ask_btn = st.button("Ask ↗", key="ask_btn", use_container_width=True)

    if ask_btn and question:
        with st.spinner("Querying…"):
            try:
                agent = get_query_agent(pdf_path_str)
                if agent is None:
                    st.error("Document not fully processed — re-run the pipeline.")
                    st.stop()
                result = agent.ask(question, top_k=5)
            except Exception as e:
                st.error(f"Query failed: {e}")
                st.stop()

        tools = result.get("tools_used") or []
        tool_pills = " ".join(f'<span class="tag tag-b">{t}</span>' for t in tools)
        st.markdown(f'<div style="margin-bottom:1rem">Tools used: &nbsp;{tool_pills}</div>', unsafe_allow_html=True)

        ans_col, prov_col = st.columns([3, 2], gap="large")

        with ans_col:
            st.markdown("**Answer Snippets**")
            snippets = result.get("answer_snippets") or []
            if snippets:
                for i, snip in enumerate(snippets, 1):
                    st.markdown(
                        f'<div class="snippet-box">'
                        f'<div class="snip-idx">Result {i}</div>'
                        f'{str(snip)[:420]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown('<div class="empty-state" style="padding:1rem"><p>No results</p></div>', unsafe_allow_html=True)

            facts = result.get("structured_facts") or []
            if facts:
                st.markdown("**Structured Facts · SQL**")
                for f in facts[:6]:
                    fn = str(f.get("field_name",""))
                    fv = str(f.get("value",""))
                    fu = str(f.get("unit") or "")
                    fp = str(f.get("page") or "?")
                    st.markdown(
                        f'<div class="trace-step">'
                        f'<span class="step-key">{fn[:28]}</span>'
                        f'<span class="step-value">{fv} {fu} &nbsp;<span style="color:var(--text-dim)">pg{fp}</span></span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        with prov_col:
            st.markdown("**ProvenanceChain**")
            provs = result.get("provenance_chain") or []
            for prov in provs:
                if not isinstance(prov, dict):
                    continue
                spans = prov.get("spans") or []
                page  = spans[0].get("page","?") if spans else "?"
                bbox  = spans[0].get("bbox") if spans else None
                h     = (prov.get("content_hash") or "")[:18]
                dname = str(prov.get("document_name","?"))
                st.markdown(
                    f'<div class="prov-card">'
                    f'<div class="prov-doc">📄 {dname}</div>'
                    f'<div class="prov-meta">'
                    f'<span>Page <b>{page}</b></span>'
                    f'{"<span>BBox "+str(bbox)+"</span>" if bbox else ""}'
                    f'</div>'
                    f'<div class="prov-hash">{h}…</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            nav_hits = result.get("navigation_hits") or []
            if nav_hits:
                st.markdown("**PageIndex Sections**")
                for sec in nav_hits:
                    if not isinstance(sec, dict):
                        continue
                    score = sec.get("score", 0)
                    st.markdown(
                        f'<div class="section-nav-card">'
                        f'<div class="snc-title">'
                        f'{str(sec.get("title","?"))[:55]}'
                        f'  <span class="snc-score">↑{score:.2f}</span>'
                        f'</div>'
                        f'<div class="snc-meta">'
                        f'<span>pp {sec.get("page_start","?")}–{sec.get("page_end","?")}</span>'
                        f'<span>L{sec.get("level","?")}</span>'
                        f'</div>'
                        f'{"<div class=snc-summary>"+str(sec.get("summary",""))[:110]+"</div>" if sec.get("summary") else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.markdown(
            '<div class="empty-state" style="padding:2rem">'
            '<span class="es-icon">💬</span>'
            '<p>Type a question and press Ask ↗</p></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — AUDIT MODE
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.markdown("##### Audit Mode — Claim Verification")
    st.markdown(
        '<div style="font-family:var(--mono);font-size:0.7rem;color:var(--text-muted);margin-bottom:0.75rem">'
        'Verify any factual claim against the document. &nbsp;'
        '<span style="color:#2ed573">verified</span> · '
        '<span style="color:#ffa502">unverifiable</span> · '
        '<span style="color:#ff4757">not_found</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    c_col, b_col = st.columns([5, 1], gap="small")
    with c_col:
        claim_input = st.text_input(
            "Claim",
            placeholder='e.g. "The report states revenue was $4.2B in Q3"',
            label_visibility="collapsed",
            key="audit_input",
        )
    with b_col:
        verify_btn = st.button("Verify ↗", key="verify_btn", use_container_width=True)

    if verify_btn and claim_input:
        with st.spinner("Auditing claim…"):
            try:
                agent = get_query_agent(pdf_path_str)
                if agent is None:
                    st.error("Document not fully processed — re-run the pipeline.")
                    st.stop()
                audit_result = agent.verify_claim(claim_input)
            except Exception as e:
                st.error(f"Audit failed: {e}")
                st.stop()

        v      = audit_result.verdict
        icon   = _verdict_icon(v)
        color  = {"verified": "#2ed573", "unverifiable": "#ffa502", "not_found": "#ff4757"}.get(v, "#7a8699")
        conf_v = audit_result.confidence

        st.markdown(
            f'<div class="verdict-{v}" style="margin-bottom:1.25rem">'
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">'
            f'  <span style="font-size:1.3rem">{icon}</span>'
            f'  <span style="font-family:var(--display);font-size:1.1rem;font-weight:800;color:{color}">'
            f'    {v.upper().replace("_"," ")}'
            f'  </span>'
            f'  <div style="flex:1;max-width:130px">{_conf_bar(conf_v)}</div>'
            f'</div>'
            f'<div style="font-size:0.83rem;color:var(--text-muted);line-height:1.55">'
            f'{audit_result.explanation}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if audit_result.evidence:
            st.markdown("**Supporting Evidence**")
            for ev in audit_result.evidence:
                if not isinstance(ev, dict):
                    continue
                spans = ev.get("spans") or []
                page  = spans[0].get("page","?") if spans else ev.get("page","?")
                bbox  = spans[0].get("bbox") if spans else None
                h     = (ev.get("content_hash") or "")[:18]
                dname = str(ev.get("document_name","?"))
                st.markdown(
                    f'<div class="prov-card">'
                    f'<div class="prov-doc">📄 {dname}</div>'
                    f'<div class="prov-meta">'
                    f'<span>Page <b>{page}</b></span>'
                    f'{"<span>BBox "+str(bbox)+"</span>" if bbox else ""}'
                    f'</div>'
                    f'<div class="prov-hash">{h}…</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            '<div class="empty-state" style="padding:2rem">'
            '<span class="es-icon">🔍</span>'
            '<p>Enter a claim to verify against the document</p></div>',
            unsafe_allow_html=True,
        )