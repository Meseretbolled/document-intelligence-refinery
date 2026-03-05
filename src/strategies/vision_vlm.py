from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import base64
import json
import os
import re
import time
import urllib.request

import pymupdf  # PyMuPDF

from src.models.extracted_document import ExtractedDocument, ExtractedBlock
from src.models.provenance import ProvenanceChain, ProvenanceSpan
from src.utils.hashing import sha256_text
from .base import BaseExtractor


def _render_page_png(pdf_path: str, page_index: int, zoom: float = 2.0) -> bytes:
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(page_index)
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def _strip_code_fences(text: str) -> str:
    """
    Removes ```json ... ``` or ``` ... ``` fences if present.
    """
    if not text:
        return text
    t = text.strip()
    m = re.match(r"^\s*```(?:json)?\s*\n(.*)\n\s*```\s*$", t, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else t


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort parse JSON returned by the model.
    """
    if not text:
        return None
    payload = _strip_code_fences(text)
    try:
        obj = json.loads(payload)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _confidence_from_blocks(blocks: List[ExtractedBlock]) -> float:
    """
    Explainable confidence: based on total extracted text length across blocks.
    """
    total_chars = sum(len((b.text or "").strip()) for b in blocks)
    if total_chars >= 3000:
        return 0.88
    if total_chars >= 1200:
        return 0.80
    if total_chars >= 400:
        return 0.70
    if total_chars >= 150:
        return 0.55
    if total_chars >= 60:
        return 0.35
    return 0.20


class VisionVLMExtractor(BaseExtractor):
    """
    Strategy C (Vision) via OpenRouter.

    Env vars:
      - OPENROUTER_API_KEY
      - OPENROUTER_MODEL (default: openai/gpt-4o-mini)
      - MAX_VLM_PAGES (default: 2)  # cost-aware guardrail

    Output:
      - multiple ExtractedBlocks with VALID block_type values only:
        'text', 'table', 'figure', 'header', 'footer'
    """

    def extract(self, doc_id: str, source_path: str) -> Tuple[ExtractedDocument, Optional[str]]:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
        max_pages = int(os.getenv("MAX_VLM_PAGES", "2"))

        p = Path(source_path)
        if not p.exists():
            return (
                ExtractedDocument(doc_id=doc_id, source_path=source_path, strategy_used="C", confidence=0.0, blocks=[]),
                "Strategy C: source file not found",
            )

        if not api_key:
            return (
                ExtractedDocument(doc_id=doc_id, source_path=source_path, strategy_used="C", confidence=0.0, blocks=[]),
                "Strategy C not run: OPENROUTER_API_KEY not set (skipping VLM).",
            )

        try:
            # Determine page count
            doc = pymupdf.open(source_path)
            page_count = doc.page_count
            pages_to_send = min(max_pages, page_count)

            images_b64: List[str] = []
            spans: List[ProvenanceSpan] = []
            for i in range(pages_to_send):
                png = _render_page_png(source_path, i, zoom=2.0)
                b64 = base64.b64encode(png).decode("utf-8")
                images_b64.append(b64)
                spans.append(ProvenanceSpan(page=i + 1, bbox=None))

            prompt = (
                "You are a document extraction engine.\n"
                "Return ONLY valid JSON (no markdown fences).\n"
                "Schema:\n"
                "{\n"
                '  \"doc_type\": string,\n'
                '  \"short_text\": string,\n'
                '  \"key_fields\": [{\"name\": string, \"value\": string}]\n'
                "}\n"
                "If unsure, use empty strings.\n"
            )

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            [{"type": "text", "text": prompt}]
                            + [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                                for b64 in images_b64
                            ]
                        ),
                    }
                ],
                "temperature": 0.0,
            }

            req = urllib.request.Request(
                url="https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            t0 = time.time()
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read().decode("utf-8")
            dt = time.time() - t0

            data = json.loads(raw)
            model_text = (data["choices"][0]["message"]["content"] or "").strip()

            parsed = _safe_json_loads(model_text)

            # provenance shared across blocks
            doc_name = p.name

            def prov_for(text: str) -> ProvenanceChain:
                return ProvenanceChain(
                    source_path=source_path,
                    document_name=doc_name,
                    content_hash=sha256_text(text),
                    spans=spans,
                )

            blocks: List[ExtractedBlock] = []

            if parsed:
                doc_type = (parsed.get("doc_type") or "").strip()
                short_text = (parsed.get("short_text") or "").strip()
                key_fields = parsed.get("key_fields") or []

                # doc_type as HEADER (valid literal)
                if doc_type:
                    line = f"Document Type: {doc_type}"
                    blocks.append(
                        ExtractedBlock(
                            block_type="header",
                            text=line,
                            html=None,
                            provenance=prov_for(line),
                        )
                    )

                # short summary as TEXT
                if short_text:
                    blocks.append(
                        ExtractedBlock(
                            block_type="text",
                            text=short_text,
                            html=None,
                            provenance=prov_for(short_text),
                        )
                    )

                # key_fields as TABLE-style text block(s)
                if isinstance(key_fields, list) and key_fields:
                    # make one table-like block with multiple lines (still block_type='table')
                    lines: List[str] = []
                    for kv in key_fields:
                        if not isinstance(kv, dict):
                            continue
                        name = (kv.get("name") or "").strip()
                        value = (kv.get("value") or "").strip()
                        if not name and not value:
                            continue
                        lines.append(f"{name}: {value}".strip(": ").strip())

                    if lines:
                        table_text = "\n".join(lines)
                        blocks.append(
                            ExtractedBlock(
                                block_type="table",
                                text=table_text,
                                html=None,
                                provenance=prov_for(table_text),
                            )
                        )

            # Fallback: if parse failed, store cleaned text as TEXT
            if not blocks:
                cleaned = _strip_code_fences(model_text)
                if cleaned:
                    blocks.append(
                        ExtractedBlock(
                            block_type="text",
                            text=cleaned,
                            html=None,
                            provenance=prov_for(cleaned),
                        )
                    )

            conf = float(_confidence_from_blocks(blocks))

            return (
                ExtractedDocument(
                    doc_id=doc_id,
                    source_path=source_path,
                    strategy_used="C",
                    confidence=conf,
                    blocks=blocks,
                    metadata={
                        "openrouter_model": model,
                        "pages_sent_to_vlm": pages_to_send,
                        "page_count": page_count,
                        "latency_s": round(dt, 3),
                    },
                ),
                f"Strategy C used OpenRouter model={model} (pages_sent={pages_to_send})",
            )

        except Exception as e:
            return (
                ExtractedDocument(doc_id=doc_id, source_path=source_path, strategy_used="C", confidence=0.0, blocks=[]),
                f"Strategy C failed: {type(e).__name__}: {e}",
            )