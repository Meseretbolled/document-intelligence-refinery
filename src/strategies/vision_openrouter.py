from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import requests

from src.models.extracted_document import ExtractedBlock, ExtractedDocument
from src.models.provenance import ProvenanceChain, ProvenanceSpan
from src.utils.hashing import sha256_text


def _strip_code_fences(text: str) -> str:
    """
    Removes ```json ... ``` or ``` ... ``` fences if present.
    Keeps the inner payload.
    """
    if not text:
        return text
    t = text.strip()

    # Matches ```json\n ... \n```  OR ```\n...\n```
    fence = re.compile(r"^\s*```(?:json)?\s*\n(.*)\n\s*```\s*$", re.DOTALL | re.IGNORECASE)
    m = fence.match(t)
    if m:
        return m.group(1).strip()
    return t


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON parse:
    - strips code fences
    - tries json.loads
    - returns None if parsing fails
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
    Simple, explainable confidence estimate based on extracted text volume.
    """
    total_chars = sum(len((b.text or "").strip()) for b in blocks)
    if total_chars >= 4000:
        return 0.90
    if total_chars >= 1500:
        return 0.82
    if total_chars >= 500:
        return 0.72
    if total_chars >= 150:
        return 0.55
    if total_chars >= 50:
        return 0.35
    return 0.20


def _pixmap_to_png_bytes(pix: fitz.Pixmap) -> bytes:
    """
    Converts a PyMuPDF pixmap to PNG bytes.
    """
    if pix.alpha:  # remove alpha for compatibility
        pix = fitz.Pixmap(pix, 0)
    return pix.tobytes("png")


def _png_bytes_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@dataclass
class VisionExtractResult:
    text: str
    model: str
    pages_sent: int


class OpenRouterVisionExtractor:
    """
    Strategy C: Vision/VLM extraction using OpenRouter.
    Expects to return structured JSON, but safely handles raw text too.
    """

    def __init__(self, model: Optional[str] = None, timeout_s: int = 60):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.timeout_s = timeout_s

    def _call_openrouter(self, page_images: List[str]) -> VisionExtractResult:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        prompt = (
            "You are extracting information from a PDF page image.\n"
            "Return ONLY valid JSON (no markdown fences).\n"
            "Schema:\n"
            "{\n"
            '  "doc_type": string,\n'
            '  "short_text": string,\n'
            '  "key_fields": [{"name": string, "value": string}]\n'
            "}\n"
            "If a field is unknown, use an empty string.\n"
        )

        # OpenRouter expects OpenAI-compatible payload
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a careful document extraction engine."},
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                + [{"type": "image_url", "image_url": {"url": img}} for img in page_images],
            },
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # optional metadata
        site_url = os.getenv("OPENROUTER_SITE_URL")
        app_name = os.getenv("OPENROUTER_APP_NAME")
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()

        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return VisionExtractResult(text=text or "", model=self.model, pages_sent=len(page_images))

    def extract(self, doc_id: str, source_path: str, max_pages: int = 2) -> Tuple[ExtractedDocument, str]:
        """
        Extracts from up to max_pages (cost-aware).
        Returns (ExtractedDocument, notes).
        """
        # Render first N pages to images
        doc = fitz.open(source_path)
        page_count = doc.page_count
        pages_to_send = min(max_pages, page_count)

        images: List[str] = []
        spans: List[ProvenanceSpan] = []
        for i in range(pages_to_send):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=180)  # good compromise for cost vs OCR/vision quality
            png_bytes = _pixmap_to_png_bytes(pix)
            images.append(_png_bytes_to_data_url(png_bytes))
            spans.append(ProvenanceSpan(page=i + 1, bbox=None))

        # Call VLM
        result = self._call_openrouter(images)

        # Parse model output
        raw_text = (result.text or "").strip()
        parsed = _safe_json_loads(raw_text)

        blocks: List[ExtractedBlock] = []
        prov = ProvenanceChain(spans=spans, source="strategy_c_openrouter")

        if parsed:
            doc_type = (parsed.get("doc_type") or "").strip()
            short_text = (parsed.get("short_text") or "").strip()
            key_fields = parsed.get("key_fields") or []

            if doc_type:
                blocks.append(
                    ExtractedBlock(
                        block_id=f"{doc_id}-c-doctype",
                        block_type="meta",
                        text=f"Document Type: {doc_type}",
                        provenance=prov,
                        content_hash=sha256_text(f"Document Type: {doc_type}"),
                    )
                )

            if short_text:
                blocks.append(
                    ExtractedBlock(
                        block_id=f"{doc_id}-c-summary",
                        block_type="summary",
                        text=short_text,
                        provenance=prov,
                        content_hash=sha256_text(short_text),
                    )
                )

            if isinstance(key_fields, list):
                for idx, kv in enumerate(key_fields):
                    if not isinstance(kv, dict):
                        continue
                    name = (kv.get("name") or "").strip()
                    value = (kv.get("value") or "").strip()
                    if not name and not value:
                        continue
                    line = f"{name}: {value}".strip(": ").strip()
                    blocks.append(
                        ExtractedBlock(
                            block_id=f"{doc_id}-c-kv-{idx}",
                            block_type="key_value",
                            text=line,
                            provenance=prov,
                            content_hash=sha256_text(line),
                        )
                    )

        # Fallback: if parse failed, store cleaned raw text as a single block
        if not blocks:
            cleaned = _strip_code_fences(raw_text)
            if cleaned:
                blocks.append(
                    ExtractedBlock(
                        block_id=f"{doc_id}-c-raw",
                        block_type="raw",
                        text=cleaned,
                        provenance=prov,
                        content_hash=sha256_text(cleaned),
                    )
                )

        confidence = _confidence_from_blocks(blocks)

        extracted = ExtractedDocument(
            doc_id=doc_id,
            source_path=source_path,
            strategy_used="C",
            confidence=confidence,
            blocks=blocks,
            metadata={
                "openrouter_model": result.model,
                "pages_sent_to_vlm": result.pages_sent,
                "page_count": page_count,
            },
        )

        notes = f"Strategy C used OpenRouter model={result.model} (pages_sent={result.pages_sent})"
        return extracted, notes