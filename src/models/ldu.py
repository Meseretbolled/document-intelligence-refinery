from __future__ import annotations

from typing import Literal, Optional, List
from pydantic import BaseModel, Field

from .provenance import ProvenanceChain


LDUType = Literal["text", "table", "figure", "list", "header"]


class LDU(BaseModel):
    """
    Logical Document Unit (LDU) model.
    Interim requirement: schema must exist (final will enforce chunking rules).
    """
    ldu_id: str = Field(..., description="Stable chunk id (e.g., doc_id + block index hash).")
    doc_id: str
    chunk_type: LDUType

    content: str
    token_count: int = 0

    parent_section: Optional[str] = None
    page_refs: List[int] = Field(default_factory=list)

    # A single bbox is enough for interim; final can store per-span bboxes.
    bounding_box: Optional[List[float]] = None  # [x0,y0,x1,y1]

    content_hash: str = Field(..., description="Hash used to verify provenance later.")
    provenance: ProvenanceChain