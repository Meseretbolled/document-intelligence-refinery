from __future__ import annotations

import base64
import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

from src.models.profile import DocumentProfile
from src.utils.pdf_signals import compute_pdf_signals
from src.utils.pdf_layout import compute_layout_signals

_DOMAIN_KEYWORDS = {
    "financial": ["balance sheet", "income statement", "profit", "loss", "revenue", "fiscal", "tax"],
    "legal":     ["hereby", "pursuant", "audit report", "independent auditor", "court", "proclamation"],
    "technical": ["methodology", "assessment", "implementation", "framework", "system", "analysis"],
    "medical":   ["patient", "diagnosis", "clinical", "treatment", "hospital", "laboratory"],
}


# ── Text-based language detection (for native-digital PDFs) ──────────────────

def _count_ethiopic_chars(text: str) -> int:
    if not text:
        return 0
    return sum(1 for c in text if "\u1200" <= c <= "\u137F")


def _count_latin_letters(text: str) -> int:
    if not text:
        return 0
    return sum(1 for c in text if ("A" <= c <= "Z") or ("a" <= c <= "z"))


def _detect_language(text_sample: str) -> tuple[Optional[str], float]:
    """
    Language detection from text. Used for native-digital PDFs.
    Returns (language_code, confidence).
    """
    sample = (text_sample or "").strip()
    if not sample:
        return None, 0.0

    ethiopic_count = _count_ethiopic_chars(sample)
    latin_count    = _count_latin_letters(sample)
    total_len      = max(1, len(sample))
    ethiopic_ratio = ethiopic_count / total_len
    latin_ratio    = latin_count / total_len

    if ethiopic_count >= 5 and latin_count >= 20:
        return "mixed", 0.85

    if ethiopic_count >= 5 and ethiopic_ratio > 0.05:
        return "am", 0.90

    try:
        from langdetect import detect_langs  # type: ignore
        langs = detect_langs(sample[:2000])
        if langs:
            top = langs[0]
            detected_lang = str(top.lang)
            detected_prob = float(top.prob)
            if ethiopic_count >= 5:
                if latin_count >= 20:
                    return "mixed", max(detected_prob, 0.80)
                return "am", max(detected_prob, 0.80)
            return detected_lang, detected_prob
    except Exception:
        pass

    if ethiopic_count >= 5:
        if latin_count >= 20:
            return "mixed", 0.80
        return "am", 0.80

    ascii_ratio = sum(1 for c in sample if ord(c) < 128) / max(1, len(sample))
    if ascii_ratio > 0.95 or latin_ratio > 0.30:
        return "en", 0.40

    return None, 0.10


# ── Image-based language detection (for scanned PDFs) ────────────────────────

def _render_page_png_b64(pdf_path: str, page_index: int = 0, zoom: float = 1.5) -> Optional[str]:
    """
    Render a PDF page as a PNG and return base64-encoded bytes.
    Uses pymupdf (already a project dependency via vision_vlm.py).
    Returns None if rendering fails.
    """
    try:
        import pymupdf  # type: ignore
        with pymupdf.open(pdf_path) as doc:
            if page_index >= len(doc):
                page_index = 0
            page = doc.load_page(page_index)
            mat  = pymupdf.Matrix(zoom, zoom)
            pix  = page.get_pixmap(matrix=mat, alpha=False)
            return base64.b64encode(pix.tobytes("png")).decode("ascii")
    except Exception:
        return None


# Free-tier VLM models that handle multilingual detection well.
# Ordered: most capable first. Same chain as vision_vlm.py.
_LANG_PROBE_MODELS = [
    "qwen/qwen3-vl-235b-thinking:free",
    "google/gemma-3-27b-it:free",
    "qwen/qwen3-vl-30b-a3b-thinking:free",
    "meta-llama/llama-4-maverick:free",
    "openrouter/auto:free",
]

# Prompt is intentionally tiny — we only need the language code, not full extraction.
_LANG_PROBE_PROMPT = (
    "Look at this document image. "
    "Reply with ONLY a JSON object like: "
    '{\"language\": \"am\"} '
    "Use these codes: "
    "\"am\" for Amharic/Ethiopic script, "
    "\"en\" for English, "
    "\"mixed\" if both scripts are present, "
    "\"other\" for any other language. "
    "Nothing else — just the JSON."
)


def _parse_lang_response(raw: str) -> Optional[str]:
    """
    Robustly extract a language code from any VLM response format.

    Handles: clean JSON, fenced JSON, prose ("The language is Amharic"),
    key-value ("Language: Amharic"), bare code ("am"), Ethiopic chars in text.

    This is the reason language detection was returning None even with
    OPENROUTER_API_KEY set: json.loads() was throwing on prose responses,
    the exception was silently caught, and all models were skipped.
    """
    import re as _re
    if not raw:
        return None

    # 1. Try JSON parse — strip fences first
    clean = _re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=_re.IGNORECASE)
    clean = _re.sub(r"\s*```$", "", clean).strip()
    try:
        obj = json.loads(clean)
        if isinstance(obj, dict):
            lang = str(obj.get("language") or "").strip().lower()
            if lang:
                return lang
    except Exception:
        pass

    # 2. JSON fragment anywhere in the text
    m = _re.search(r'\{[^}]*"language"\s*:\s*"([^"]+)"[^}]*\}', raw, _re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()

    # 3. Prose / keyword detection
    lower = raw.lower()
    if any(w in lower for w in ["amharic", "ethiopic", "ge'ez", "geez", "\u12a0\u121d\u1203\u122d\u129b"]):
        return "am"
    if any(w in lower for w in ["tigrinya", "tigri"]):
        return "am"
    if "mixed" in lower and any(w in lower for w in ["amharic", "english", "ethiopic"]):
        return "mixed"
    if any(w in lower for w in ["english", "latin script"]):
        return "en"

    # 4. Check for Ethiopic Unicode characters in the raw response itself
    if any("\u1200" <= c <= "\u137f" for c in raw):
        return "am"

    # 5. Bare language code on its own line
    for line in raw.strip().splitlines():
        stripped = line.strip().lower().strip(".,;:")
        if stripped in ("am", "amharic", "ethiopic"):
            return "am"
        if stripped in ("en", "english"):
            return "en"
        if stripped == "mixed":
            return "mixed"

    return None


def _detect_language_from_image(pdf_path: str) -> tuple[Optional[str], float]:
    """
    Detect language of a scanned PDF by sending page 1 image to the VLM.

    - Only called when: origin_type == "scanned_image" AND language is still None
    - Requires OPENROUTER_API_KEY to be set in environment
    - If API key missing or all models fail: returns (None, 0.0) safely
    - Uses max_tokens=20 — extremely fast and cheap (still $0.00 on free tier)

    Returns (language_code, confidence).
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        # No API key — return None gracefully, pipeline continues without language
        return None, 0.0

    img_b64 = _render_page_png_b64(pdf_path, page_index=0, zoom=1.5)
    if not img_b64:
        return None, 0.0

    base_url = os.getenv("OPENROUTER_URL", os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    headers  = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    for model in _LANG_PROBE_MODELS:
        payload = {
            "model": model,
            "max_tokens": 20,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text",      "text": _LANG_PROBE_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }],
        }
        try:
            req = urllib.request.Request(
                url=f"{base_url.rstrip('/')}/chat/completions",
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            raw = (data["choices"][0]["message"]["content"] or "").strip()
            print(f"  [triage-lang] model={model} raw response: {raw[:120]!r}")

            # Parse response — VLMs return many formats: JSON, prose, bare code.
            # Use a multi-strategy parser so any format succeeds.
            lang = _parse_lang_response(raw)
            print(f"  [triage-lang] parsed lang={lang!r}")

            if lang in ("am", "amharic", "ethiopic", "tigrinya", "ti"):
                return "am", 0.90
            if lang == "mixed":
                return "mixed", 0.85
            if lang in ("en", "english"):
                return "en", 0.85
            if lang and lang != "other":
                return lang, 0.75

        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:300]
            except Exception:
                pass
            print(f"  [triage-lang] model={model} HTTP {e.code}: {body}")
            if e.code in (429, 503, 529, 400, 404, 422):
                continue
            break
        except Exception as _ex:
            print(f"  [triage-lang] model={model} error: {type(_ex).__name__}: {_ex}")
            continue

    # All models failed — return None safely
    print(f"  [triage-lang] all models failed — language probe returning None")
    return None, 0.0


# ── Domain detection ──────────────────────────────────────────────────────────

def _detect_domain(text_sample: str) -> Optional[str]:
    s = (text_sample or "").lower()
    if not s:
        return None
    best_domain, best_hits = None, 0
    for dom, kws in _DOMAIN_KEYWORDS.items():
        hits = sum(1 for k in kws if k in s)
        if hits > best_hits:
            best_domain, best_hits = dom, hits
    return best_domain or "general"


# ── Main triage function ──────────────────────────────────────────────────────

def triage_pdf(pdf_path: str, rules: dict) -> DocumentProfile:
    """
    Stage 1: Triage Agent.
    Produces DocumentProfile which governs strategy routing.

    For native-digital PDFs: language detected from text layer.
    For scanned PDFs: language detected by sending page 1 image to VLM
                      (requires OPENROUTER_API_KEY; graceful fallback if not set).
    """
    path   = Path(pdf_path)
    doc_id = path.stem

    signals = compute_pdf_signals(pdf_path)
    layout  = compute_layout_signals(pdf_path)

    scanned_rules = rules["triage"]["scanned_image_threshold"]
    is_scanned = (
        signals.avg_image_area_ratio >= float(scanned_rules["image_area_ratio_gte"])
        and signals.avg_text_chars_per_page <= float(scanned_rules["text_chars_per_page_lte"])
    )
    origin_type = "scanned_image" if is_scanned else "native_digital"

    # Layout complexity
    if layout.tableish_score >= 0.50:
        layout_complexity = "table_heavy"
    elif layout.figureish_score >= 0.45:
        layout_complexity = "figure_heavy"
    elif layout.approx_column_count >= 2:
        layout_complexity = "multi_column"
    else:
        layout_complexity = "single_column"

    # Try to read text layer (works for native-digital; empty for scanned)
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

    # Language detection
    language, language_conf = _detect_language(text_sample)

    if language is None and is_scanned:
        # ── Scanned doc with no text layer: probe language via VLM image ─────
        # Sends page 1 as an image to the VLM with a tiny prompt (max_tokens=20).
        # Requires OPENROUTER_API_KEY in .env — returns (None, 0.0) if not set.
        print(f"  [triage] scanned PDF, no text layer — probing language via VLM for {doc_id}")
        language, language_conf = _detect_language_from_image(pdf_path)
        if language:
            print(f"  [triage] detected language={language} (conf={language_conf:.2f}) from image")
        else:
            print(f"  [triage] language probe inconclusive — ensure OPENROUTER_API_KEY is set in .env")

    domain_hint = _detect_domain(text_sample)

    # Fallback: infer domain from filename when text layer is empty (scanned docs)
    if not domain_hint or domain_hint == "general":
        fname_lower = doc_id.lower()
        if any(k in fname_lower for k in ["budget", "expense", "expenditure", "revenue", "tax", "fiscal", "finance", "financial"]):
            domain_hint = "financial"
        elif any(k in fname_lower for k in ["audit", "auditor", "legal", "court", "proclamation"]):
            domain_hint = "legal"
        elif any(k in fname_lower for k in ["procurement", "tender", "contract"]):
            domain_hint = "legal"
        elif any(k in fname_lower for k in ["assessment", "survey", "report", "analysis"]):
            domain_hint = "technical"

    # Cost tier
    if origin_type == "scanned_image":
        cost_tier = "needs_vision_model"
    else:
        cost_tier = "fast_text_sufficient" if layout_complexity == "single_column" else "needs_layout_model"

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