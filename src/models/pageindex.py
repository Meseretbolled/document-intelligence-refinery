from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


BlockType = Literal["text", "table", "figure", "header", "footer"]


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class PageIndexItem(BaseModel):
    ldu_id: str = ""
    chunk_type: BlockType = "text"
    snippet: str = ""
    content_hash: str = ""
    bbox: Optional[BBox] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class PageIndexPage(BaseModel):
    page: int
    items: List[PageIndexItem] = Field(default_factory=list)
    data_types_present: List[BlockType] = Field(default_factory=list)
    char_count: int = 0
    item_count: int = 0


class PageIndex(BaseModel):
    doc_id: str
    source_path: str
    root: List[PageIndexPage] = Field(default_factory=list)
    page_count: int = 0
    data_types_present: List[BlockType] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)