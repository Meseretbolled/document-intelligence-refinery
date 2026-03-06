from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
import re

import pdfplumber

from src.models.extracted_document import ExtractedDocument, ExtractedBlock
from src.models.provenance import ProvenanceChain, ProvenanceSpan
from src.utils.hashing import sha256_text
from .base import BaseExtractor


def _count_ethiopic_chars(text: str) -> int:
    if not text:
        return 0
    return sum(1 for c in text if "\u1200" <= c <= "\u137F")


def _count_latin_letters(text: str) -> int:
    if not text:
        return 0
    return sum(1 for c in text if ("A" <= c <= "Z") or ("a" <= c <= "z"))


def _detect_script_language(text: str) -> tuple[Optional[str], Optional[str], float]:
    """
    Returns:
      (language, script, confidence)

    Possible language values:
      - "am"
      - "en"
      - "mixed"
      - None
    """
    t = (text or "").strip()
    if not t:
        return None, None, 0.0

    ethiopic_count = _count_ethiopic_chars(t)
    latin_count = _count_latin_letters(t)
    total_len = max(1, len(t))

    ethiopic_ratio = ethiopic_count / total_len
    latin_ratio = latin_count / total_len

    if ethiopic_count >= 5 and latin_count >= 20:
        return "mixed", "mixed", 0.85

    if ethiopic_count >= 5 and ethiopic_ratio > 0.03:
        return "am", "ethiopic", 0.90

    if latin_count >= 20 and latin_ratio > 0.15:
        return "en", "latin", 0.75

    return None, None, 0.10


def _chunk_markdown(md: str, max_chars: int = 1200) -> List[str]:
    """
    Split markdown into chunks that map better to LDUs:
      - split by headings (#, ##, ###)
      - then by blank lines
      - keep chunks under max_chars
    """
    md = (md or "").strip()
    if not md:
        return []

    lines = md.splitlines()
    sections: List[str] = []
    buf: List[str] = []

    heading_re = re.compile(r"^\s{0,3}#{1,6}\s+\S")

    def flush():
        nonlocal buf
        text = "\n".join(buf).strip()
        if text:
            sections.append(text)
        buf = []

    for line in lines:
        if heading_re.match(line) and buf:
            flush()
        buf.append(line)

    flush()

    chunks: List[str] = []
    for sec in sections:
        parts = re.split(r"\n\s*\n", sec.strip())
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= max_chars:
                chunks.append(part)
            else:
                start = 0
                while start < len(part):
                    piece = part[start : start + max_chars].strip()
                    if piece:
                        chunks.append(piece)
                    start += max_chars

    return [c for c in chunks if c]


def _confidence_from_md(md: str, chunks: List[str]) -> float:
    """
    Cheap, explainable heuristic:
    - more text + more chunks => higher confidence
    """
    n = len(chunks)
    total = len((md or "").strip())

    if total >= 5000 and n >= 15:
        return 0.90
    if total >= 2000 and n >= 8:
        return 0.85
    if total >= 800 and n >= 4:
        return 0.78
    if total >= 300:
        return 0.65
    if total >= 80:
        return 0.45
    return 0.25


def _ocr_quality_flags(text: str) -> List[str]:
    """
    Very light quality heuristics to detect noisy OCR/markdown extraction.
    """
    t = (text or "").strip()
    if not t:
        return ["empty extracted markdown"]

    reasons: List[str] = []

    weird_symbols = sum(1 for c in t if c in "∞①②③④⑤⑥⑦⑧⑨○●◉◌※")
    if weird_symbols >= 3:
        reasons.append("contains unusual OCR symbols")

    alnum = sum(1 for c in t if c.isalnum())
    if len(t) > 0 and (alnum / len(t)) < 0.35:
        reasons.append("low alphanumeric ratio")

    ethiopic_count = _count_ethiopic_chars(t)
    latin_count = _count_latin_letters(t)

    # If there are many non-ASCII chars but almost no Ethiopic or Latin,
    # it may indicate OCR corruption.
    non_ascii = sum(1 for c in t if ord(c) >= 128)
    if non_ascii >= 10 and ethiopic_count < 3 and latin_count < 10:
        reasons.append("non-ascii text without clear script signal")

    return reasons


class DoclingLayoutExtractor(BaseExtractor):
    """
    Strategy B (Layout-aware)
    Uses Docling if installed.

    Final behavior:
    - converts Docling markdown into multiple blocks
    - annotates extracted output with language/script hints
    - flags likely OCR noise in metadata
    """

    def extract(self, doc_id: str, source_path: str) -> Tuple[ExtractedDocument, Optional[str]]:
        try:
            from docling.document_converter import DocumentConverter  # type: ignore
        except Exception:
            return (
                ExtractedDocument(
                    doc_id=doc_id,
                    source_path=source_path,
                    strategy_used="B",
                    confidence=0.0,
                    blocks=[],
                    meta={"needs_review": True},
                ),
                "Docling not installed. Install with: pip install docling",
            )

        page_count = 1
        try:
            with pdfplumber.open(source_path) as pdf:
                page_count = len(pdf.pages) or 1
        except Exception:
            page_count = 1

        converter = DocumentConverter()

        try:
            result = converter.convert(source_path)
        except Exception as e:
            return (
                ExtractedDocument(
                    doc_id=doc_id,
                    source_path=source_path,
                    strategy_used="B",
                    confidence=0.0,
                    blocks=[],
                    meta={"needs_review": True},
                ),
                f"Docling conversion failed: {type(e).__name__}: {e}",
            )

        export_md = getattr(result.document, "export_to_markdown", None)
        md = export_md() if callable(export_md) else ""
        md = (md or "").strip()

        spans = [ProvenanceSpan(page=i, bbox=None) for i in range(1, page_count + 1)]
        doc_name = Path(source_path).name

        chunks = _chunk_markdown(md, max_chars=1200)

        blocks: List[ExtractedBlock] = []
        for chunk in chunks:
            blocks.append(
                ExtractedBlock(
                    block_type="text",
                    text=chunk,
                    html=None,
                    provenance=ProvenanceChain(
                        source_path=source_path,
                        document_name=doc_name,
                        content_hash=sha256_text(chunk),
                        spans=spans,
                    ),
                )
            )

        confidence = float(_confidence_from_md(md, chunks))
        detected_language, detected_script, lang_conf = _detect_script_language(md)
        quality_reasons = _ocr_quality_flags(md)

        notes: List[str] = []
        if md and len(blocks) <= 1:
            notes.append(
                "Docling returned markdown, but chunking produced <=1 block (document may be very short)."
            )
        if not md:
            notes.append("Docling returned empty markdown.")
        if quality_reasons:
            notes.append("Potential OCR/layout quality issue: " + "; ".join(quality_reasons))

        meta = {
            "detected_language_after_extraction": detected_language,
            "detected_script_after_extraction": detected_script,
            "language_detection_confidence": float(lang_conf),
            "ocr_quality_reasons": quality_reasons,
            "needs_review": confidence < 0.70 or len(quality_reasons) > 0,
            "page_count": page_count,
            "chunk_count": len(chunks),
        }

        return (
            ExtractedDocument(
                doc_id=doc_id,
                source_path=source_path,
                strategy_used="B",
                confidence=confidence,
                blocks=blocks,
                meta=meta,
            ),
            " | ".join(notes) if notes else None,
        )