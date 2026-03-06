from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.models.profile import DocumentProfile
from src.utils.pdf_signals import compute_pdf_signals
from src.utils.pdf_layout import compute_layout_signals

_DOMAIN_KEYWORDS = {
    "financial": ["balance sheet", "income statement", "profit", "loss", "revenue", "fiscal", "tax"],
    "legal": ["hereby", "pursuant", "audit report", "independent auditor", "court", "proclamation"],
    "technical": ["methodology", "assessment", "implementation", "framework", "system", "analysis"],
    "medical": ["patient", "diagnosis", "clinical", "treatment", "hospital", "laboratory"],
}


def _count_ethiopic_chars(text: str) -> int:
    """
    Counts characters in the Ethiopic Unicode block.
    Covers the main Ethiopic range used by Amharic text.
    """
    if not text:
        return 0
    return sum(1 for c in text if "\u1200" <= c <= "\u137F")


def _count_latin_letters(text: str) -> int:
    """
    Counts basic Latin alphabet letters.
    """
    if not text:
        return 0
    return sum(1 for c in text if ("A" <= c <= "Z") or ("a" <= c <= "z"))


def _looks_ethiopic(text: str, min_chars: int = 5) -> bool:
    """
    True if the text contains enough Ethiopic characters to strongly suggest
    Amharic/Ethiopic script presence.
    """
    return _count_ethiopic_chars(text) >= min_chars


def _detect_language(text_sample: str) -> tuple[Optional[str], float]:
    """
    Language detection with Ethiopic/Amharic-aware overrides.

    Returns:
      - "am" for clearly Ethiopic-script text
      - "mixed" for strong English + Ethiopic coexistence
      - detected language from langdetect if available
      - safe fallback otherwise
    """
    sample = (text_sample or "").strip()
    if not sample:
        return None, 0.0

    ethiopic_count = _count_ethiopic_chars(sample)
    latin_count = _count_latin_letters(sample)
    total_len = max(1, len(sample))

    ethiopic_ratio = ethiopic_count / total_len
    latin_ratio = latin_count / total_len

    # Strong bilingual / mixed-language signal
    if ethiopic_count >= 5 and latin_count >= 20:
        return "mixed", 0.85

    # Strong Ethiopic script signal
    if ethiopic_count >= 5 and ethiopic_ratio > 0.05:
        return "am", 0.90

    try:
        from langdetect import detect_langs  # type: ignore

        langs = detect_langs(sample[:2000])
        if langs:
            top = langs[0]
            detected_lang = str(top.lang)
            detected_prob = float(top.prob)

            # Safety override: if langdetect misses Ethiopic but text clearly has it
            if ethiopic_count >= 5:
                if latin_count >= 20:
                    return "mixed", max(detected_prob, 0.80)
                return "am", max(detected_prob, 0.80)

            return detected_lang, detected_prob

    except Exception:
        pass

    # Fallback heuristics
    if ethiopic_count >= 5:
        if latin_count >= 20:
            return "mixed", 0.80
        return "am", 0.80

    ascii_ratio = sum(1 for c in sample if ord(c) < 128) / max(1, len(sample))
    if ascii_ratio > 0.95 or latin_ratio > 0.30:
        return "en", 0.40

    return None, 0.10


def _detect_domain(text_sample: str) -> Optional[str]:
    s = (text_sample or "").lower()
    if not s:
        return None

    best_domain = None
    best_hits = 0
    for dom, kws in _DOMAIN_KEYWORDS.items():
        hits = sum(1 for k in kws if k in s)
        if hits > best_hits:
            best_domain = dom
            best_hits = hits

    return best_domain or "general"


def triage_pdf(pdf_path: str, rules: dict) -> DocumentProfile:
    """
    Stage 1: Triage Agent
    Produces DocumentProfile which governs strategy routing.

    Notes:
    - For digital PDFs, language detection can work directly from the text layer.
    - For scanned PDFs, text_sample may be empty before OCR; in that case language
      may still be None at triage time and can be updated later after extraction.
    """
    path = Path(pdf_path)
    doc_id = path.stem

    signals = compute_pdf_signals(pdf_path)
    layout = compute_layout_signals(pdf_path)

    scanned_rules = rules["triage"]["scanned_image_threshold"]

    is_scanned = (
        signals.avg_image_area_ratio >= float(scanned_rules["image_area_ratio_gte"])
        and signals.avg_text_chars_per_page <= float(scanned_rules["text_chars_per_page_lte"])
    )

    origin_type = "scanned_image" if is_scanned else "native_digital"

    # Layout complexity classification (heuristic)
    if layout.tableish_score >= 0.50:
        layout_complexity = "table_heavy"
    elif layout.figureish_score >= 0.45:
        layout_complexity = "figure_heavy"
    elif layout.approx_column_count >= 2:
        layout_complexity = "multi_column"
    else:
        layout_complexity = "single_column"

    # Collect a tiny text sample for language/domain hints
    text_sample = ""
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            for p in pdf.pages[:2]:
                t = (p.extract_text() or "").strip()
                if t:
                    text_sample += "\n" + t
    except Exception:
        text_sample = ""

    language, language_conf = _detect_language(text_sample)
    domain_hint = _detect_domain(text_sample)

    # Estimated extraction cost tier
    if origin_type == "scanned_image":
        cost_tier = "needs_vision_model"
    else:
        if layout_complexity == "single_column":
            cost_tier = "fast_text_sufficient"
        else:
            cost_tier = "needs_layout_model"

    return DocumentProfile(
        doc_id=doc_id,
        source_path=str(path),
        origin_type=origin_type,
        layout_complexity=layout_complexity,
        language=language,
        language_confidence=float(language_conf),
        domain_hint=domain_hint,
        page_count=signals.page_count,
        avg_text_chars_per_page=signals.avg_text_chars_per_page,
        avg_image_area_ratio=signals.avg_image_area_ratio,
        cost_tier=cost_tier,
    )