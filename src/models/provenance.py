from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field


class ProvenanceSpan(BaseModel):
    page: int = Field(..., ge=1)
    bbox: Optional[List[float]] = Field(
        default=None, description="Bounding box [x0,y0,x1,y1] if available."
    )


class ProvenanceChain(BaseModel):
    source_path: str
    content_hash: str
    spans: List[ProvenanceSpan] = Field(default_factory=list)