from __future__ import annotations

import re
from typing import Tuple, Optional
from pathlib import Path

import pdfplumber

from src.models.extracted_document import ExtractedDocument, ExtractedBlock
from src.models.provenance import ProvenanceChain, ProvenanceSpan
from src.utils.hashing import sha256_text

HEADING_RE = re.compile(
    r"^(\d+[\.\d]*\.?\s+[A-Z].{3,80}|[A-Z][A-Z\s,&/:\-()]{5,79})$"
)
CAPTION_RE = re.compile(r"^(Figure|Fig\.?|Table|Exhibit|Chart)\s*\d+", re.I)


def _page_bbox_union(page) -> Optional[list[float]]:
    try:
        words = page.extract_words() or []
        if not words:
            return None
        x0 = min(float(w["x0"]) for w in words if "x0" in w)
        y0 = min(float(w["top"]) for w in words if "top" in w)
        x1 = max(float(w["x1"]) for w in words if "x1" in w)
        y1 = max(float(w["bottom"]) for w in words if "bottom" in w)
        return [x0, y0, x1, y1]
    except Exception:
        return None


class FastTextExtractor:
    """
    Strategy A (Fast Text)
    Emits header / table / figure / text blocks so all 5 chunking rules fire.
    """

    def extract(self, doc_id: str, source_path: str) -> Tuple[ExtractedDocument, Optional[str]]:
        p = Path(source_path)
        if not p.exists():
            return (
                ExtractedDocument(
                    doc_id=doc_id,
                    source_path=source_path,
                    strategy_used="A",
                    confidence=0.0,
                    blocks=[],
                ),
                "Strategy A: source file not found",
            )

        blocks: list[ExtractedBlock] = []
        total_chars = 0
        pages_with_text = 0

        try:
            with pdfplumber.open(str(p)) as pdf:
                for i, page in enumerate(pdf.pages, start=1):

                    # R1: extract tables first as standalone structured blocks
                    table_texts: set[str] = set()
                    for tbl in (page.extract_tables() or []):
                        if not tbl or not tbl[0]:
                            continue
                        rows = [
                            " | ".join(str(c or "").strip() for c in row)
                            for row in tbl
                        ]
                        tbl_text = "\n".join(rows).strip()
                        if len(tbl_text) < 5:
                            continue
                        table_texts.add(tbl_text[:60])  # dedup key
                        h = sha256_text(tbl_text)
                        blocks.append(ExtractedBlock(
                            block_type="table",
                            text=tbl_text,
                            provenance=ProvenanceChain(
                                source_path=source_path,
                                document_name=p.name,
                                content_hash=h,
                                spans=[ProvenanceSpan(page=i, bbox=_page_bbox_union(page))],
                            ),
                        ))
                        total_chars += len(tbl_text)

                    # Text layer: classify each line
                    raw = (page.extract_text() or "").strip()
                    if not raw:
                        continue
                    pages_with_text += 1
                    total_chars += len(raw)
                    bbox = _page_bbox_union(page)

                    for line in raw.splitlines():
                        line = line.strip()
                        if not line or len(line) < 3:
                            continue
                        if HEADING_RE.match(line) and len(line) < 90:
                            btype = "header"
                        elif CAPTION_RE.match(line):
                            btype = "figure"
                        else:
                            btype = "text"
                        h = sha256_text(line)
                        blocks.append(ExtractedBlock(
                            block_type=btype,
                            text=line,
                            provenance=ProvenanceChain(
                                source_path=source_path,
                                document_name=p.name,
                                content_hash=h,
                                spans=[ProvenanceSpan(page=i, bbox=bbox)],
                            ),
                        ))

            if pages_with_text == 0 or total_chars < 30:
                conf = 0.25
                note = "Strategy A: very little/no text extracted (will escalate)"
            else:
                conf = min(0.95, 0.60 + (total_chars / 2000.0))
                note = None

            return ExtractedDocument(
                doc_id=doc_id,
                source_path=source_path,
                strategy_used="A",
                confidence=float(conf),
                blocks=blocks,
            ), note

        except Exception as e:
            return ExtractedDocument(
                doc_id=doc_id,
                source_path=source_path,
                strategy_used="A",
                confidence=0.0,
                blocks=[],
            ), f"Strategy A failed: {type(e).__name__}: {e}"