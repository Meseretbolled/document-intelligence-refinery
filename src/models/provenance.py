from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ProvenanceSpan(BaseModel):
    model_config = ConfigDict(extra="allow")   # ← CRITICAL FIX for R5 cross-refs

    page: int = Field(..., ge=1)
    bbox: Optional[List[float]] = Field(
        default=None, description="Bounding box [x0,y0,x1,y1] in PDF coordinate space."
    )


class ProvenanceChain(BaseModel):
    model_config = ConfigDict(extra="allow")   # ← CRITICAL FIX for R5 cross-refs

    source_path: str
    document_name: Optional[str] = None
    content_hash: str
    spans: List[ProvenanceSpan] = Field(default_factory=list)
    cross_refs: List[str] = Field(default_factory=list)   # explicit R5 storage