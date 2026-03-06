from __future__ import annotations

from typing import Optional, Tuple, List

from src.models.extracted_document import ExtractedDocument
from .ocr_local import LocalOCRExtractor
from .ocr_layout import OCRLayoutExtractor
from .vision_vlm import VisionVLMExtractor


class StrategyCRouter:
    """
    Layered Strategy C:

    C1 -> local OCR
    C2 -> OCR + layout reconstruction
    C3 -> Vision LLM

    Stops as soon as confidence is good enough.
    """

    def __init__(
        self,
        c1_threshold: float = 0.60,
        c2_threshold: float = 0.72,
        allow_c3: bool = True,
    ) -> None:
        self.c1 = LocalOCRExtractor()
        self.c2 = OCRLayoutExtractor()
        self.c3 = VisionVLMExtractor()
        self.c1_threshold = c1_threshold
        self.c2_threshold = c2_threshold
        self.allow_c3 = allow_c3

    def extract(self, doc_id: str, source_path: str) -> Tuple[ExtractedDocument, Optional[str]]:
        notes: List[str] = []

        # C1
        out1, note1 = self.c1.extract(doc_id, source_path)
        if note1:
            notes.append(note1)
        if out1.confidence >= self.c1_threshold:
            out1.metadata = out1.metadata or {}
            out1.metadata["strategy_c_route"] = ["C1"]
            return out1, " | ".join(notes)

        # C2
        out2, note2 = self.c2.extract(doc_id, source_path)
        if note2:
            notes.append(note2)
        if out2.confidence >= self.c2_threshold:
            out2.metadata = out2.metadata or {}
            out2.metadata["strategy_c_route"] = ["C1", "C2"]
            return out2, " | ".join(notes)

        # C3
        if self.allow_c3:
            out3, note3 = self.c3.extract(doc_id, source_path)
            if note3:
                notes.append(note3)
            out3.metadata = out3.metadata or {}
            out3.metadata["strategy_c_route"] = ["C1", "C2", "C3"]
            out3.metadata["needs_review"] = out3.confidence < 0.70
            return out3, " | ".join(notes)

        # If C3 disabled, return best of C1/C2
        best = out2 if out2.confidence >= out1.confidence else out1
        best.metadata = best.metadata or {}
        best.metadata["strategy_c_route"] = ["C1", "C2"]
        best.metadata["fallback_reason"] = "C3 disabled or budget-blocked"
        best.metadata["needs_review"] = True
        return best, " | ".join(notes)