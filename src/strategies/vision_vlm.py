from __future__ import annotations

"""
Strategy C3 — Vision LLM extractor via OpenRouter (FREE tier only).

Default model chain — ALL FREE, $0 cost:
  1. qwen/qwen3-vl-235b-thinking:free   — best OCR, 32-lang, document AI specialist
  2. qwen/qwen3-vl-30b-a3b-thinking:free — free, 131K ctx, strong multilingual
  3. nvidia/nemotron-nano-12b-vl:free    — trained on OCR/DocVQA tasks
  4. meta-llama/llama-4-maverick:free    — free, multimodal fallback
  5. google/gemma-3-27b-it:free          — 140+ languages, good for Amharic
  6. openrouter/auto:free                — final safety net: auto-routes to any free model

For Amharic/Ethiopic docs, chain is re-ordered to prefer multilingual-strong models first.

Rate limits (free tier): ~20 req/min, ~200 req/day per model.
When a model hits its limit (HTTP 429/503), automatically tries next model in chain
with exponential backoff (2s → 4s → 8s) before escalating.

All costs are $0.00 — logged to ledger as 0.0 for budget transparency.
"""

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf

from src.models.extracted_document import ExtractedDocument, ExtractedBlock
from src.models.provenance import ProvenanceChain, ProvenanceSpan
from src.utils.hashing import sha256_text
from .base import BaseExtractor


# ── Free model chains ─────────────────────────────────────────────────────────
# Ordered: best document OCR first, lightest/most-available last.
# All models end in :free — zero cost.

FREE_MODEL_CHAIN = [
    "qwen/qwen3-vl-235b-thinking:free",        # Best: OCR specialist, 32 languages, 262K ctx
    "qwen/qwen3-vl-30b-a3b-thinking:free",     # Good: free, 131K ctx, strong vision
    "nvidia/nemotron-nano-12b-vl:free",         # Good: trained on OCRBench + DocVQA
    "meta-llama/llama-4-maverick:free",         # Fallback: large multimodal
    "google/gemma-3-27b-it:free",              # Fallback: 140+ languages
    "openrouter/auto:free",                     # Safety net: OpenRouter picks best available free
]

# For Amharic/Ethiopic: prefer models with stronger multilingual training
FREE_AMHARIC_MODEL_CHAIN = [
    "qwen/qwen3-vl-235b-thinking:free",        # Best multilingual OCR
    "google/gemma-3-27b-it:free",              # 140+ languages explicitly
    "qwen/qwen3-vl-30b-a3b-thinking:free",     # Strong fallback
    "meta-llama/llama-4-maverick:free",         # Broad multilingual
    "openrouter/auto:free",                     # Safety net
]

# HTTP codes meaning "rate limited" → retry with backoff, then try next model
RATE_LIMIT_CODES = {429, 503, 529}
# HTTP codes meaning "model can't handle this request" → skip immediately
SKIP_MODEL_CODES = {400, 404, 422}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_page_png(pdf_path: str, page_index: int, zoom: float = 2.0) -> bytes:
    with pymupdf.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")


def _strip_fences(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    m = re.match(r"^\s*```(?:json)?\s*\n(.*)\n\s*```\s*$", t, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else t


def _safe_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    payload = _strip_fences(text)
    for attempt in [payload, payload[payload.find("{"):payload.rfind("}") + 1]]:
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _confidence(blocks: List[ExtractedBlock]) -> float:
    total = sum(len((b.text or "").strip()) for b in blocks)
    if total >= 3000: return 0.88
    if total >= 1200: return 0.80
    if total >= 400:  return 0.70
    if total >= 150:  return 0.55
    if total >= 60:   return 0.35
    return 0.20


# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(language: Optional[str]) -> str:
    is_amharic = (language or "") in ("am", "ti", "mixed")
    lang_note = (
        "CRITICAL: This document is in Amharic (Ethiopic/Ge'ez script ፊደል). "
        "You MUST preserve every Ethiopic character exactly as written. "
        "Do NOT transliterate, romanise, skip, or replace with '?' — "
        "output every word in its original Ethiopic script. "
    ) if is_amharic else ""

    return (
        "You are a high-accuracy document OCR and structured extraction engine.\n"
        f"{lang_note}"
        "Extract ALL text visible in the provided page image(s).\n"
        "Return ONLY valid JSON — no markdown fences, no explanation.\n\n"
        "{\n"
        '  "doc_type": "<document type, e.g. financial report, government form>",\n'
        '  "language": "<primary language, e.g. am, en, mixed>",\n'
        '  "full_text": "<complete verbatim text from the entire page — do not omit anything>",\n'
        '  "sections": [{"heading": "<section heading>", "body": "<section text>"}],\n'
        '  "tables": [{"caption": "<caption or empty>", "rows": [["<cell>", "..."]]}],\n'
        '  "key_fields": [{"name": "<field name>", "value": "<field value>"}]\n'
        "}\n\n"
        "Rules:\n"
        "- full_text must contain EVERY line of text on the page — do not summarise.\n"
        "- tables: every row, every cell, preserve numeric precision exactly.\n"
        "- key_fields: extract name/value pairs like dates, IDs, totals, titles.\n"
        "- Empty fields → empty string or []. No extra JSON keys.\n"
    )


# ── HTTP call ─────────────────────────────────────────────────────────────────

class _RateLimited(Exception):
    def __init__(self, code: int, msg: str):
        self.code = code
        super().__init__(msg)


class _SkipModel(Exception):
    def __init__(self, code: int, msg: str):
        self.code = code
        super().__init__(msg)


def _call(model: str, images_b64: List[str], prompt: str,
          api_key: str, base_url: str, site_url: str, app_name: str,
          timeout: int = 120) -> Tuple[str, str]:
    """Returns (raw_text, resolved_model). Raises _RateLimited or _SkipModel on HTTP errors."""
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": (
                [{"type": "text", "text": prompt}]
                + [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}}
                   for b in images_b64]
            ),
        }],
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        text = (data["choices"][0]["message"]["content"] or "").strip()
        return text, data.get("model", model)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:300]
        except Exception:
            pass
        if e.code in RATE_LIMIT_CODES:
            raise _RateLimited(e.code, f"HTTP {e.code} — {model}: {body}")
        if e.code in SKIP_MODEL_CODES:
            raise _SkipModel(e.code, f"HTTP {e.code} — {model}: {body}")
        raise


# ── Block parser ──────────────────────────────────────────────────────────────

def _parse_blocks(parsed: Optional[Dict], raw: str,
                  spans: List[ProvenanceSpan], source_path: str, doc_name: str
                  ) -> List[ExtractedBlock]:

    def prov(text: str) -> ProvenanceChain:
        return ProvenanceChain(
            source_path=source_path, document_name=doc_name,
            content_hash=sha256_text(text), spans=spans,
        )

    blocks: List[ExtractedBlock] = []
    if parsed:
        doc_type  = str(parsed.get("doc_type") or "").strip()
        full_text = str(parsed.get("full_text") or "").strip()
        sections  = parsed.get("sections") or []
        tables    = parsed.get("tables") or []
        kfields   = parsed.get("key_fields") or []

        if doc_type:
            t = f"Document Type: {doc_type}"
            blocks.append(ExtractedBlock(block_type="header", text=t, provenance=prov(t)))
        if full_text:
            blocks.append(ExtractedBlock(block_type="text", text=full_text, provenance=prov(full_text)))
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            h = str(sec.get("heading") or "").strip()
            b = str(sec.get("body") or "").strip()
            if h:
                blocks.append(ExtractedBlock(block_type="header", text=h, provenance=prov(h)))
            if b and b != full_text:
                blocks.append(ExtractedBlock(block_type="text", text=b, provenance=prov(b)))
        for tbl in tables:
            if not isinstance(tbl, dict):
                continue
            rows = tbl.get("rows") or []
            if not rows:
                continue
            cap = str(tbl.get("caption") or "").strip()
            lines = ([f"[Table: {cap}]"] if cap else []) + [
                " | ".join(str(c) for c in r) if isinstance(r, list) else str(r)
                for r in rows
            ]
            t = "\n".join(lines)
            blocks.append(ExtractedBlock(block_type="table", text=t, provenance=prov(t)))
        if kfields:
            kvlines = [
                f"{kv.get('name','').strip()}: {kv.get('value','').strip()}".strip(": ")
                for kv in kfields if isinstance(kv, dict) and (kv.get("name") or kv.get("value"))
            ]
            if kvlines:
                t = "\n".join(kvlines)
                blocks.append(ExtractedBlock(block_type="table", text=t, provenance=prov(t)))

    if not blocks:
        cleaned = _strip_fences(raw)
        if cleaned:
            blocks.append(ExtractedBlock(block_type="text", text=cleaned, provenance=prov(cleaned)))
    return blocks


# ── Main extractor ────────────────────────────────────────────────────────────

class VisionVLMExtractor(BaseExtractor):
    """
    Strategy C3 — Vision LLM via OpenRouter using ONLY free models.

    Model fallback chain (all free, $0):
      qwen/qwen3-vl-235b-thinking:free  (best OCR + multilingual)
      qwen/qwen3-vl-30b-a3b-thinking:free
      nvidia/nemotron-nano-12b-vl:free  (DocVQA specialist)
      meta-llama/llama-4-maverick:free
      google/gemma-3-27b-it:free        (140+ languages, good for Amharic)
      openrouter/auto:free              (safety net)

    Amharic fast-path reorders to multilingual-strongest first.
    Exponential backoff (2s→4s→8s) on 429, then tries next model.
    """

    def extract(
        self,
        doc_id: str,
        source_path: str,
        language: Optional[str] = None,
    ) -> Tuple[ExtractedDocument, Optional[str]]:

        api_key  = os.getenv("OPENROUTER_API_KEY", "").strip()
        base_url = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1").strip()
        site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
        app_name = os.getenv("OPENROUTER_APP_NAME", "").strip()
        max_pages = int(os.getenv("MAX_VLM_PAGES", "6"))

        is_amharic = (language or "") in ("am", "ti", "mixed")

        # Build model chain: env override → language-specific free chain → default free chain
        env_chain = os.getenv("OPENROUTER_MODEL_CHAIN", "").strip()
        if env_chain:
            model_chain = [m.strip() for m in env_chain.split(",") if m.strip()]
        elif is_amharic:
            model_chain = FREE_AMHARIC_MODEL_CHAIN.copy()
        else:
            model_chain = FREE_MODEL_CHAIN.copy()

        # Legacy single-model env var: insert at front but keep chain as fallback
        single = (os.getenv("MODEL_NAME", "") or os.getenv("OPENROUTER_MODEL", "")).strip()
        if single and single not in model_chain:
            model_chain = [single] + model_chain

        p = Path(source_path)
        if not p.exists():
            return _fail(doc_id, source_path, "file not found")
        if not api_key:
            return _fail(doc_id, source_path, "OPENROUTER_API_KEY not set")

        # Render pages
        try:
            with pymupdf.open(source_path) as doc:
                page_count = doc.page_count
            pages_to_send = min(max_pages, page_count)
            images_b64: List[str] = []
            spans: List[ProvenanceSpan] = []
            for i in range(pages_to_send):
                png = _render_page_png(source_path, i, zoom=2.0)
                images_b64.append(base64.b64encode(png).decode())
                spans.append(ProvenanceSpan(page=i + 1, bbox=None))
        except Exception as e:
            return _fail(doc_id, source_path, f"page render failed: {e}")

        prompt = _build_prompt(language)
        doc_name = p.name
        tried: List[str] = []
        last_error = ""
        t0 = time.time()

        # ── Free model fallback loop ──────────────────────────────────────────
        for model in model_chain:
            tried.append(model)
            backoffs = [2.0, 4.0, 8.0]   # exponential backoff delays
            attempt = 0

            while True:
                try:
                    raw_text, resolved = _call(
                        model=model, images_b64=images_b64, prompt=prompt,
                        api_key=api_key, base_url=base_url,
                        site_url=site_url, app_name=app_name,
                    )
                    # ── Success ───────────────────────────────────────────────
                    dt = time.time() - t0
                    parsed = _safe_json(raw_text)
                    blocks = _parse_blocks(parsed, raw_text, spans, source_path, doc_name)
                    conf = _confidence(blocks)

                    return (
                        ExtractedDocument(
                            doc_id=doc_id,
                            source_path=source_path,
                            strategy_used="C",
                            confidence=conf,
                            blocks=blocks,
                            meta={
                                "strategy_c_level": "C3",
                                "model_used": resolved,
                                "model_chain_tried": tried,
                                "all_free_tier": True,
                                "estimated_cost_usd": 0.0,   # free!
                                "pages_sent_to_vlm": pages_to_send,
                                "page_count": page_count,
                                "latency_s": round(dt, 3),
                                "language_hint": language,
                                "is_amharic": is_amharic,
                                "needs_review": conf < 0.70,
                            },
                        ),
                        (
                            f"Strategy C3 (free) — model={resolved}, "
                            f"tried={tried}, pages={pages_to_send}, "
                            f"lang={language}, cost=$0.00, latency={dt:.1f}s"
                        ),
                    )

                except _RateLimited as e:
                    # 429 — wait and retry same model
                    last_error = str(e)
                    if attempt < len(backoffs):
                        time.sleep(backoffs[attempt])
                        attempt += 1
                        continue
                    # Exhausted retries → next model
                    last_error = f"{model} rate-limited after {attempt} retries (429)"
                    break

                except _SkipModel as e:
                    # 400/404/422 — model can't handle it, skip immediately
                    last_error = str(e)
                    break

                except Exception as e:
                    last_error = f"{model}: {type(e).__name__}: {e}"
                    break

        # ── All free models exhausted ─────────────────────────────────────────
        dt = time.time() - t0
        return (
            ExtractedDocument(
                doc_id=doc_id,
                source_path=source_path,
                strategy_used="C",
                confidence=0.0,
                blocks=[],
                meta={
                    "strategy_c_level": "C3",
                    "model_chain_tried": tried,
                    "all_free_tier": True,
                    "estimated_cost_usd": 0.0,
                    "last_error": last_error,
                    "needs_review": True,
                    "latency_s": round(dt, 3),
                },
            ),
            f"Strategy C3 FAILED — all free models exhausted: {tried}. Last: {last_error}",
        )


def _fail(doc_id: str, source_path: str, reason: str) -> Tuple[ExtractedDocument, str]:
    return (
        ExtractedDocument(
            doc_id=doc_id, source_path=source_path,
            strategy_used="C", confidence=0.0, blocks=[],
            meta={"strategy_c_level": "C3", "all_free_tier": True,
                  "estimated_cost_usd": 0.0, "error": reason, "needs_review": True},
        ),
        f"Strategy C3 not run: {reason}",
    )