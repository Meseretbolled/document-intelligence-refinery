from __future__ import annotations

import json
from pathlib import Path
import streamlit as st

from src.service.refinery_service import run_refinery_on_pdf
from src.settings import settings


st.set_page_config(page_title="Document Intelligence Refinery", layout="wide")


def list_local_pdfs(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        return []
    return sorted(root_dir.rglob("*.pdf"))


st.title("📄 Document Intelligence Refinery")
st.caption("Upload or choose a PDF → run triage + extraction + LDU + page index with cost-aware routing.")


col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Input")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    raw_dir_default = settings.project_root / "data" / "raw" / "data"
    st.write("Or pick from local data folder:")
    local_dir = st.text_input("Local PDF directory", value=str(raw_dir_default))

    local_pdfs = list_local_pdfs(Path(local_dir))
    selected_local = st.selectbox(
        "Select a local PDF",
        options=[""] + [str(p) for p in local_pdfs[:200]],
        index=0,
    )

    run_btn = st.button("🚀 Run Refinery", type="primary")

with col2:
    st.subheader("Run Info / Decision Trace")
    st.info(
        "This UI surfaces the 'engineering story': strategy selection, confidence, "
        "budget/cost estimation, and decision trace."
    )

pdf_to_process: Path | None = None

# If upload: write to temp location inside project
if uploaded is not None:
    temp_dir = settings.project_root / ".refinery" / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    pdf_to_process = temp_dir / uploaded.name
    pdf_to_process.write_bytes(uploaded.getbuffer())

elif selected_local:
    pdf_to_process = Path(selected_local)

if run_btn:
    if pdf_to_process is None:
        st.error("Please upload or select a PDF.")
        st.stop()

    with st.spinner("Running refinery pipeline..."):
        out = run_refinery_on_pdf(str(pdf_to_process))

    # --- Summary panel
    meta = out.extracted.get("meta") or {}
    decision_trace = meta.get("decision_trace") or []

    st.success("Done ✅")

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Strategy", out.extracted.get("strategy_used", ""))
    top2.metric("Confidence", f"{out.extracted.get('confidence', 0.0):.2f}")
    top3.metric("Blocks", str(len(out.extracted.get("blocks") or [])))
    top4.metric("LDUs", str(len(out.ldus)))

    st.write("**Notes:**", out.notes if out.notes else "(none)")

    st.write("**Decision trace:**")
    st.json(decision_trace)

    # --- Tabs: profile/extracted/ldu/page_index
    tab1, tab2, tab3, tab4 = st.tabs(["Profile", "Extracted", "LDUs", "Page Index"])

    with tab1:
        st.json(out.profile)
        st.caption(f"Saved: {out.profile_path}")

    with tab2:
        # Show extracted blocks nicely
        blocks = out.extracted.get("blocks") or []
        st.caption(f"Saved: {out.extracted_path}")

        for i, b in enumerate(blocks):
            with st.expander(f"Block {i} • {b.get('block_type')}"):
                txt = b.get("text") or ""
                if txt:
                    st.text(txt)
                else:
                    st.write("(no text)")
                prov = b.get("provenance")
                if prov:
                    st.write("Provenance:")
                    st.json(prov)

        st.write("Full extracted JSON:")
        st.json(out.extracted)

    with tab3:
        st.caption(f"Saved: {out.ldu_path}")
        st.json(out.ldus[:50])
        if len(out.ldus) > 50:
            st.info(f"Showing first 50 LDUs (total {len(out.ldus)}).")

    with tab4:
        st.caption(f"Saved: {out.page_index_path}")
        st.json(out.page_index)

    # --- Download buttons
    dl1, dl2, dl3, dl4 = st.columns(4)
    dl1.download_button("Download Profile JSON", data=json.dumps(out.profile, indent=2), file_name="profile.json")
    dl2.download_button("Download Extracted JSON", data=json.dumps(out.extracted, indent=2), file_name="extracted.json")
    dl3.download_button("Download LDUs JSON", data=json.dumps(out.ldus, indent=2), file_name="ldu.json")
    dl4.download_button("Download PageIndex JSON", data=json.dumps(out.page_index, indent=2), file_name="page_index.json")