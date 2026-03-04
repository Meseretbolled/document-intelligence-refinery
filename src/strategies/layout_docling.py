from __future__ import annotations

from typing import Optional, Tuple, List

from src.models.extracted_document import ExtractedDocument, ExtractedBlock
from src.models.provenance import ProvenanceChain, ProvenanceSpan
from src.utils.hashing import sha256_text
from .base import BaseExtractor


class DoclingLayoutExtractor(BaseExtractor):
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
                "Docling not installed. Install with: pip install -e '.[docling]'",
            )

        converter = DocumentConverter()

        try:
            result = converter.convert(source_path)
        except Exception as e:
            # ✅ Do not crash the whole pipeline
            return (
                ExtractedDocument(
                    doc_id=doc_id,
                    source_path=source_path,
                    strategy_used="B",
                    confidence=0.0,
                    blocks=[],
                ),
                f"Docling conversion failed (likely OCR/GPU memory). Falling back. Error: {type(e).__name__}",
            )

        # normalize to markdown/text
        export_md = getattr(result.document, "export_to_markdown", None)
        md = export_md() if callable(export_md) else ""

        blocks: List[ExtractedBlock] = []
        if md.strip():
            blocks.append(
                ExtractedBlock(
                    block_type="text",
                    text=md,
                    provenance=ProvenanceChain(
                        source_path=source_path,
                        content_hash=sha256_text(md),
                        spans=[ProvenanceSpan(page=1, bbox=None)],
                    ),
                )
            )

        confidence = 0.75 if len(md) > 1000 else (0.55 if len(md) > 200 else 0.25)

        return (
            ExtractedDocument(
                doc_id=doc_id,
                source_path=source_path,
                strategy_used="B",
                confidence=confidence,
                blocks=blocks,
            ),
            None,
        )