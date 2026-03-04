from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from src.models.document_profile import DocumentProfile
from src.utils.pdf_signals import compute_pdf_signals


def triage_pdf(pdf_path: str, rules: dict):
    path = Path(pdf_path)
    doc_id = path.stem

    signals = compute_pdf_signals(pdf_path)

    scanned_rules = rules["triage"]["scanned_image_threshold"]

    is_scanned = (
        signals.avg_image_area_ratio >= float(scanned_rules["image_area_ratio_gte"])
        and signals.avg_text_chars_per_page <= float(scanned_rules["text_chars_per_page_lte"])
    )

    origin_type = "scanned_image" if is_scanned else "native_digital"

    # very simple interim heuristics
    if is_scanned:
        layout_complexity = "mixed"
        cost_tier = "needs_layout_model"
    else:
        # if lots of text, usually okay; if very low text but not scanned, can be mixed
        if signals.avg_text_chars_per_page > 1200:
            layout_complexity = "single_column"
            cost_tier = "fast_text_sufficient"
        else:
            layout_complexity = "mixed"
            cost_tier = "needs_layout_model"

    return DocumentProfile(
        doc_id=doc_id,
        source_path=str(path),
        origin_type=origin_type,
        layout_complexity=layout_complexity,
        language=None,
        language_confidence=0.0,
        domain_hint=None,
        page_count=signals.page_count,
        avg_text_chars_per_page=signals.avg_text_chars_per_page,
        avg_image_area_ratio=signals.avg_image_area_ratio,
        cost_tier=cost_tier,
    )