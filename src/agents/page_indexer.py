from __future__ import annotations

from typing import List

from src.models.pageindex import PageIndex, PageIndexSection
from src.models.ldu import LDU


def build_page_index(doc_id: str, source_path: str, ldus: List[LDU]) -> PageIndex:
    """
    Stage 5: Build a minimal PageIndex.

    Final-ready minimum:
    - root section with page range
    - light summary
    - later you can add child sections using headings detection
    """
    pages = sorted({p for ldu in ldus for p in (ldu.page_refs or [])})
    if pages:
        page_start, page_end = pages[0], pages[-1]
    else:
        page_start, page_end = 1, 1

    summary = None
    if ldus:
        summary = (ldus[0].content or "").strip()[:500] or None

    root = PageIndexSection(
        title="Document",
        page_start=page_start,
        page_end=page_end,
        child_sections=[],
        key_entities=[],
        summary=summary,
        data_types_present=[],
    )

    return PageIndex(doc_id=doc_id, source_path=source_path, root=root)