from __future__ import annotations

from typing import Optional, Tuple

from src.models.extracted_document import ExtractedDocument
from .base import BaseExtractor


class VisionVLMExtractor(BaseExtractor):
    def extract(self, doc_id: str, source_path: str) -> Tuple[ExtractedDocument, Optional[str]]:
        # Interim stub: return low confidence, no blocks
        return (
            ExtractedDocument(
                doc_id=doc_id,
                source_path=source_path,
                strategy_used="C",
                confidence=0.0,
                blocks=[],
            ),
            "Strategy C (VLM) is stubbed for interim. Implement for final submission.",
        )