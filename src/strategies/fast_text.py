from __future__ import annotations

from typing import Tuple, Optional
from pathlib import Path

import pdfplumber

from src.models.extracted_document import ExtractedDocument, ExtractedBlock
from src.models.provenance import ProvenanceChain, ProvenanceSpan


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
                    text = page.extract_text() or ""
                    text = text.strip()

                    if text:
                        pages_with_text += 1
                        total_chars += len(text)

                        blocks.append(
                            ExtractedBlock(
                                block_type="text",
                                text=text,
                                html=None,
                                provenance=ProvenanceChain(
                                    source_path=source_path,
                                    content_hash="",  # ok for interim if you don't hash here
                                    spans=[ProvenanceSpan(page=i, bbox=None)],
                                ),
                            )
                        )

            # Confidence heuristic:
            # - If at least one page has text and we extracted meaningful chars -> high confidence.
            # - Otherwise -> low confidence and caller will escalate.
            if pages_with_text == 0 or total_chars < 30:
                conf = 0.25
                note = "Strategy A: very little/no text extracted (will escalate if allowed)"
            else:
                # scale up with total_chars but cap at 0.95
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
            return extracted, f"Strategy A failed: {e}"