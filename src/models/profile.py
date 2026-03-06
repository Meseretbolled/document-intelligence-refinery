from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel


OriginType = Literal["native_digital", "scanned_image", "mixed"]
LayoutComplexity = Literal["single_column", "multi_column", "table_heavy", "figure_heavy"]
CostTier = Literal["fast_text_sufficient", "needs_layout_model", "needs_vision_model"]


class DocumentProfile(BaseModel):
    """
    Output of Stage 1 (Triage).
    Controls routing for strategies A / B / C.
    """

    doc_id: str
    source_path: str

    origin_type: OriginType
    layout_complexity: LayoutComplexity

    language: Optional[str] = None
    language_confidence: float = 0.0

    domain_hint: Optional[str] = None

    page_count: int = 1
    avg_text_chars_per_page: float = 0.0
    avg_image_area_ratio: float = 0.0

    cost_tier: CostTier