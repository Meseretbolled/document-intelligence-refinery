from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.models.provenance import ProvenanceChain


class ExtractedBlock(BaseModel):
    """
    A single unit of extracted content.
    """
    block_type: Literal["text", "table", "figure", "header", "footer", "list", "section_header", "caption"]
    text: Optional[str] = None
    html: Optional[str] = None
    provenance: Optional[ProvenanceChain] = None


class ExtractedDocument(BaseModel):
    """
    Final structured output for a processed document.
    """
    doc_id: str
    source_path: str

    strategy_used: str
    confidence: float

    blocks: List[ExtractedBlock] = Field(default_factory=list)

    # optional metadata for engineering decisions (budget, qa, cache, trace, etc.)
    meta: Optional[Dict[str, Any]] = None