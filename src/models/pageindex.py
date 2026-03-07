from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


BlockType = Literal["text", "table", "figure", "header", "footer", "list"]


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class PageIndexItem(BaseModel):
    ldu_id: str = ""
    chunk_type: str = "text"
    snippet: str = ""
    content_hash: str = ""
    bbox: Optional[BBox] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class PageIndexPage(BaseModel):
    page: int
    items: List[PageIndexItem] = Field(default_factory=list)
    data_types_present: List[str] = Field(default_factory=list)
    char_count: int = 0
    item_count: int = 0


class SectionNode(BaseModel):
    """
    Hierarchical section node for PageIndex tree.
    Mirrors VectifyAI PageIndex concept.
    """
    title: str
    page_start: int
    page_end: int
    level: int = 1                          # heading depth (H1=1, H2=2, …)
    summary: str = ""                       # LLM-generated 2-3 sentence summary
    key_entities: List[str] = Field(default_factory=list)
    data_types_present: List[str] = Field(default_factory=list)
    child_sections: List["SectionNode"] = Field(default_factory=list)
    ldu_ids: List[str] = Field(default_factory=list)   # LDUs belonging to this section


SectionNode.model_rebuild()


class PageIndex(BaseModel):
    doc_id: str
    source_path: str
    root: List[PageIndexPage] = Field(default_factory=list)       # flat page-level index
    sections: List[SectionNode] = Field(default_factory=list)     # hierarchical section tree
    page_count: int = 0
    data_types_present: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)