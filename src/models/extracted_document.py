from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from .provenance import ProvenanceChain


BlockType = Literal["text", "table", "figure", "header", "footer"]


class ExtractedBlock(BaseModel):
    block_type: BlockType
    text: Optional[str] = None
    html: Optional[str] = None  # for tables if available
    provenance: ProvenanceChain


class ExtractedDocument(BaseModel):
    doc_id: str
    source_path: str
    strategy_used: Literal["A", "B", "C"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    blocks: List[ExtractedBlock] = Field(default_factory=list)