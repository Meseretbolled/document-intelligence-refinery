from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
import re

import pymupdf  # PyMuPDF

from src.models.extracted_document import ExtractedDocument, ExtractedBlock
from src.models.provenance import ProvenanceChain, ProvenanceSpan
from src.utils.hashing import sha256_text
from .base import BaseExtractor


def _classify_block(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "text"

    # Short title-like / heading line
    if len(t) < 120 and (
        t.isupper()
        or t.endswith(":")
        or re.match(r"^[A-Z][A-Za-z0-9\s,&/\-()]{3,}$", t)
    ):
        return "header"

    # Table-ish heuristic
    if ":" in t:
        return "table"

    numeric_lines = sum(1 for line in t.splitlines() if re.search(r"\d", line))
    if numeric_lines >= 3:
        return "table"

    return "text"


def _confidence_from_blocks(blocks: List[ExtractedBlock]) -> float:
    total_chars = sum(len((b.text or "").strip()) for b in blocks)
    table_blocks = sum(1 for b in blocks if b.block_type == "table")
    header_blocks = sum(1 for b in blocks if b.block_type == "header")

    score = 0.15
    if total_chars >= 300:
        score += 0.20
    if total_chars >= 1000:
        score += 0.20
    if total_chars >= 2500:
        score += 0.15
    if table_blocks > 0:
        score += 0.10
    if header_blocks > 0:
        score += 0.10

    return max(0.0, min(0.85, round(score, 3)))


class OCRLayoutExtractor(BaseExtractor):
    """
    Strategy C2:
    local OCR + light structure reconstruction.
    Better than raw OCR because it tries to infer
    header/table/text blocks.
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
                    metadata={"strategy_c_level": "C2"},
                ),
                "C2 failed: source file not found",
            )

        try:
            doc = pymupdf.open(source_path)
            blocks: List[ExtractedBlock] = []

            for i in range(doc.page_count):
                page = doc.load_page(i)
                text = (page.get_text("text") or "").strip()
                if not text:
                    continue

                # Split into rough paragraph chunks
                raw_chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
                if not raw_chunks:
                    raw_chunks = [text]

                for chunk in raw_chunks:
                    btype = _classify_block(chunk)
                    spans = [ProvenanceSpan(page=i + 1, bbox=None)]

                    blocks.append(
                        ExtractedBlock(
                            block_type=btype,
                            text=chunk,
                            html=None,
                            provenance=ProvenanceChain(
                                source_path=source_path,
                                document_name=p.name,
                                content_hash=sha256_text(chunk),
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
                        "strategy_c_level": "C2",
                        "page_count": doc.page_count,
                        "needs_review": conf < 0.70,
                    },
                ),
                f"C2 OCR+layout reconstruction used ({len(blocks)} blocks)",
            )

        except Exception as e:
            return (
                ExtractedDocument(
                    doc_id=doc_id,
                    source_path=source_path,
                    strategy_used="C",
                    confidence=0.0,
                    blocks=[],
                    metadata={"strategy_c_level": "C2"},
                ),
                f"C2 failed: {type(e).__name__}: {e}",
            )