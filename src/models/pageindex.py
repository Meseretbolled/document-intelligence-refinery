from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class PageIndexSection(BaseModel):
    title: str
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)

    child_sections: List["PageIndexSection"] = Field(default_factory=list)

    key_entities: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    data_types_present: List[str] = Field(default_factory=list)  # ["tables","figures","equations"]


class PageIndex(BaseModel):
    doc_id: str
    source_path: str
    root: PageIndexSection