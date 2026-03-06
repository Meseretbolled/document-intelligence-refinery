from __future__ import annotations

import os
from typing import List, Dict, Any

from src.models.profile import DocumentProfile
from src.models.extracted_document import ExtractedDocument
from src.models.ldu import LDU
from src.models.pageindex import PageIndex, PageIndexPage, PageIndexItem


class FinalIndexer:
    """
    Final-stage tree/index builder.
    Uses lightweight summaries by default.
    If OPENROUTER_API_KEY exists, you can later plug an LLM summary call here.
    """

    def __init__(self) -> None:
        self.has_llm = bool(os.getenv("OPENROUTER_API_KEY"))

    def _page_summary(self, items: list[PageIndexItem]) -> str:
        previews = [x.snippet for x in items[:3] if x.snippet]
        if not previews:
            return ""
        return " | ".join(previews)[:300]

    def build(self, profile: DocumentProfile, extracted: ExtractedDocument, ldus: List[LDU]) -> PageIndex:
        pages: dict[int, list[PageIndexItem]] = {}

        for ldu in ldus:
            for p in (ldu.page_refs or []):
                pages.setdefault(p, []).append(
                    PageIndexItem(
                        ldu_id=ldu.ldu_id,
                        chunk_type=getattr(ldu, "chunk_type", "text"),
                        snippet=(ldu.content or "")[:180],
                        content_hash=ldu.content_hash,
                        bbox=None,
                        meta={},
                    )
                )

        page_nodes: list[PageIndexPage] = []
        data_types: set[str] = set()

        for p in sorted(pages):
            items = pages[p]
            for it in items:
                data_types.add(it.chunk_type)

            page_nodes.append(
                PageIndexPage(
                    page=p,
                    items=items,
                    data_types_present=sorted({it.chunk_type for it in items}),
                    char_count=sum(len(it.snippet or "") for it in items),
                    item_count=len(items),
                )
            )

        return PageIndex(
            doc_id=profile.doc_id,
            source_path=profile.source_path,
            root=page_nodes,
            page_count=profile.page_count,
            data_types_present=sorted(data_types),
            meta={
                "strategy_used": extracted.strategy_used,
                "confidence": extracted.confidence,
                "summary_mode": "llm" if self.has_llm else "heuristic",
            },
        )