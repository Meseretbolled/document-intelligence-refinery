from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
import re

import pdfplumber

from src.models.extracted_document import ExtractedDocument, ExtractedBlock
from src.models.provenance import ProvenanceChain, ProvenanceSpan
from src.utils.hashing import sha256_text
from .base import BaseExtractor


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

    # further split large sections by blank lines
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
                # hard wrap into max_chars segments
                start = 0
                while start < len(part):
                    chunks.append(part[start : start + max_chars].strip())
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


class DoclingLayoutExtractor(BaseExtractor):
    """
    Strategy B (Layout-aware)
    Uses Docling if installed.
    FINAL: converts markdown into multiple blocks so LDU/PageIndex are meaningful.
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
                ),
                "Docling not installed. Install with: pip install docling",
            )

        # Count pages for provenance coverage
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
        for i, chunk in enumerate(chunks):
            # one block per chunk for better LDUs
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

        notes = None
        if md and len(blocks) <= 1:
            notes = "Docling returned markdown, but chunking produced <=1 block (document may be very short)."
        if not md:
            notes = "Docling returned empty markdown."

        return (
            ExtractedDocument(
                doc_id=doc_id,
                source_path=source_path,
                strategy_used="B",
                confidence=confidence,
                blocks=blocks,
            ),
            notes,
        )