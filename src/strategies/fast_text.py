from __future__ import annotations

from typing import Tuple, Optional
from pathlib import Path

import pdfplumber

from src.models.extracted_document import ExtractedDocument, ExtractedBlock
from src.models.provenance import ProvenanceChain, ProvenanceSpan
from src.utils.hashing import sha256_text


def _page_bbox_union(page) -> Optional[list[float]]:
    """
    For interim: build a single bbox per page from word boxes.
    If anything is missing, return None (never crash).
    """
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
    - For native-digital PDFs with a text layer.
    - Uses pdfplumber to extract text page-by-page.
    - Confidence is high when we successfully extract enough characters.
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
                    text = (page.extract_text() or "").strip()
                    if not text:
                        continue

                    pages_with_text += 1
                    total_chars += len(text)

                    bbox = _page_bbox_union(page)
                    h = sha256_text(text)

                    blocks.append(
                        ExtractedBlock(
                            block_type="text",
                            text=text,
                            html=None,
                            provenance=ProvenanceChain(
                                source_path=source_path,
                                document_name=p.name,
                                content_hash=h,
                                spans=[ProvenanceSpan(page=i, bbox=bbox)],
                            ),
                        )
                    )

            if pages_with_text == 0 or total_chars < 30:
                conf = 0.25
                note = "Strategy A: very little/no text extracted (will escalate if allowed)"
            else:
                conf = min(0.95, 0.60 + (total_chars / 2000.0))
                note = None

            extracted = ExtractedDocument(
                doc_id=doc_id,
                source_path=source_path,
                strategy_used="A",
                confidence=float(conf),
                blocks=blocks,
            )
            return extracted, note

        except Exception as e:
            extracted = ExtractedDocument(
                doc_id=doc_id,
                source_path=source_path,
                strategy_used="A",
                confidence=0.0,
                blocks=[],
            )
            return extracted, f"Strategy A failed: {type(e).__name__}: {e}"