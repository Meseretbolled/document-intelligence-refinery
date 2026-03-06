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
    """
    Render one PDF page as PNG bytes.
    """
    with pymupdf.open(pdf_path) as doc:
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
    m = re.match(
        r"^\s*```(?:json)?\s*\n(.*)\n\s*```\s*$",
        t,
        flags=re.DOTALL | re.IGNORECASE,
    )
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
        pass

    # Extra fallback: try to extract the first {...} block
    try:
        start = payload.find("{")
        end = payload.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(payload[start : end + 1])
            return obj if isinstance(obj, dict) else None
    except Exception:
        return None

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
    Strategy C3 (Vision LLM) via OpenRouter.

    Env vars:
      - OPENROUTER_API_KEY
      - OPENROUTER_URL (default: https://openrouter.ai/api/v1)
      - OPENROUTER_MODEL or MODEL_NAME
      - MAX_VLM_PAGES (default: 2)
      - OPENROUTER_SITE_URL (optional)
      - OPENROUTER_APP_NAME (optional)

    Output:
      - multiple ExtractedBlocks with VALID block_type values only:
        'text', 'table', 'figure', 'header', 'footer'
    """

    def extract(self, doc_id: str, source_path: str) -> Tuple[ExtractedDocument, Optional[str]]:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        base_url = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1").strip()
        model = (
            os.getenv("OPENROUTER_MODEL", "").strip()
            or os.getenv("MODEL_NAME", "").strip()
            or "openai/gpt-4o-mini"
        )
        max_pages = int(os.getenv("MAX_VLM_PAGES", "2"))
        site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
        app_name = os.getenv("OPENROUTER_APP_NAME", "").strip()

        p = Path(source_path)
        if not p.exists():
            return (
                ExtractedDocument(
                    doc_id=doc_id,
                    source_path=source_path,
                    strategy_used="C",
                    confidence=0.0,
                    blocks=[],
                    meta={"strategy_c_level": "C3"},
                ),
                "Strategy C3: source file not found",
            )

        if not api_key:
            return (
                ExtractedDocument(
                    doc_id=doc_id,
                    source_path=source_path,
                    strategy_used="C",
                    confidence=0.0,
                    blocks=[],
                    meta={"strategy_c_level": "C3"},
                ),
                "Strategy C3 not run: OPENROUTER_API_KEY not set",
            )

        try:
            with pymupdf.open(source_path) as doc:
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
                "Return ONLY valid JSON with no markdown fences.\n"
                "Schema:\n"
                "{\n"
                '  "doc_type": string,\n'
                '  "short_text": string,\n'
                '  "key_fields": [{"name": string, "value": string}]\n'
                "}\n"
                "Rules:\n"
                "- Be faithful to the document image.\n"
                "- If unsure, use empty strings.\n"
                "- Do not add extra keys.\n"
            )

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            [{"type": "text", "text": prompt}]
                            + [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                                }
                                for b64 in images_b64
                            ]
                        ),
                    }
                ],
                "temperature": 0.0,
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            if site_url:
                headers["HTTP-Referer"] = site_url
            if app_name:
                headers["X-Title"] = app_name

            req = urllib.request.Request(
                url=f"{base_url.rstrip('/')}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )

            t0 = time.time()
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read().decode("utf-8")
            dt = time.time() - t0

            data = json.loads(raw)
            resolved_model = data.get("model", model)
            model_text = (data["choices"][0]["message"]["content"] or "").strip()

            parsed = _safe_json_loads(model_text)

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
                doc_type = str(parsed.get("doc_type") or "").strip()
                short_text = str(parsed.get("short_text") or "").strip()
                key_fields = parsed.get("key_fields") or []

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

                if short_text:
                    blocks.append(
                        ExtractedBlock(
                            block_type="text",
                            text=short_text,
                            html=None,
                            provenance=prov_for(short_text),
                        )
                    )

                if isinstance(key_fields, list) and key_fields:
                    lines: List[str] = []
                    for kv in key_fields:
                        if not isinstance(kv, dict):
                            continue
                        name = str(kv.get("name") or "").strip()
                        value = str(kv.get("value") or "").strip()
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
                    meta={
                        "strategy_c_level": "C3",
                        "requested_model": model,
                        "resolved_model": resolved_model,
                        "pages_sent_to_vlm": pages_to_send,
                        "page_count": page_count,
                        "latency_s": round(dt, 3),
                        "needs_review": conf < 0.70,
                    },
                ),
                f"Strategy C3 used OpenRouter model={resolved_model} (requested={model}, pages_sent={pages_to_send})",
            )

        except Exception as e:
            return (
                ExtractedDocument(
                    doc_id=doc_id,
                    source_path=source_path,
                    strategy_used="C",
                    confidence=0.0,
                    blocks=[],
                    meta={"strategy_c_level": "C3", "needs_review": True},
                ),
                f"Strategy C3 failed: {type(e).__name__}: {e}",
            )