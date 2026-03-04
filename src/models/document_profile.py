from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


OriginType = Literal["native_digital", "scanned_image", "mixed", "form_fillable"]
LayoutComplexity = Literal["single_column", "multi_column", "table_heavy", "figure_heavy", "mixed"]
CostTier = Literal["fast_text_sufficient", "needs_layout_model", "needs_vision_model"]


class DocumentProfile(BaseModel):
    doc_id: str = Field(..., description="Stable ID for this document (filename hash or user ID).")
    source_path: str = Field(..., description="Local path of the input document.")
    origin_type: OriginType
    layout_complexity: LayoutComplexity
    language: Optional[str] = None
    language_confidence: float = 0.0
    domain_hint: Optional[str] = None

    page_count: int = 0
    avg_text_chars_per_page: float = 0.0
    avg_image_area_ratio: float = 0.0

    cost_tier: CostTier