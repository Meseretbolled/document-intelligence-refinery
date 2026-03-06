from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import pymupdf  # PyMuPDF

from src.models.extracted_document import ExtractedDocument, ExtractedBlock
from src.models.provenance import ProvenanceChain, ProvenanceSpan
from src.utils.hashing import sha256_text
from .base import BaseExtractor


def _render_page_text_fallback(page) -> str:
    """
    Cheap local fallback if no OCR library is available:
    return any embedded text layer if present.
    """
    try:
        return (page.get_text("text") or "").strip()
    except Exception:
        return ""


def _confidence_from_blocks(blocks: List[ExtractedBlock]) -> float:
    total_chars = sum(len((b.text or "").strip()) for b in blocks)
    if total_chars >= 3000:
        return 0.82
    if total_chars >= 1200:
        return 0.72
    if total_chars >= 400:
        return 0.60
    if total_chars >= 150:
        return 0.45
    if total_chars >= 60:
        return 0.30
    return 0.10


class LocalOCRExtractor(BaseExtractor):
    """
    Strategy C1:
    local OCR / free fallback for scanned PDFs.

    Current implementation:
    - prefers embedded local text if available
    - designed as local recovery path
    - later you can plug RapidOCR directly here if needed
    """

    def extract(self, doc_id: str, source_path: str) -> Tuple[ExtractedDocument, Optional[str]]:
        p = Path(source_path)
        if not p.exists():
            return (
                ExtractedDocument(
                    doc_id=doc_id,
                    source_path=source_path,
                    strategy_used="C",
                    confidence=0.0,
                    blocks=[],
                    metadata={"strategy_c_level": "C1"},
                ),
                "C1 failed: source file not found",
            )

        try:
            doc = pymupdf.open(source_path)
            blocks: List[ExtractedBlock] = []

            for i in range(doc.page_count):
                page = doc.load_page(i)
                text = _render_page_text_fallback(page)

                if not text:
                    continue

                spans = [ProvenanceSpan(page=i + 1, bbox=None)]
                blocks.append(
                    ExtractedBlock(
                        block_type="text",
                        text=text,
                        html=None,
                        provenance=ProvenanceChain(
                            source_path=source_path,
                            document_name=p.name,
                            content_hash=sha256_text(text),
                            spans=spans,
                        ),
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
                        "strategy_c_level": "C1",
                        "page_count": doc.page_count,
                        "needs_review": conf < 0.60,
                    },
                ),
                f"C1 local OCR fallback used ({len(blocks)} blocks)",
            )

        except Exception as e:
            return (
                ExtractedDocument(
                    doc_id=doc_id,
                    source_path=source_path,
                    strategy_used="C",
                    confidence=0.0,
                    blocks=[],
                    metadata={"strategy_c_level": "C1"},
                ),
                f"C1 failed: {type(e).__name__}: {e}",
            )