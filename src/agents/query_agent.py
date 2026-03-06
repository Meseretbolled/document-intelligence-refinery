from __future__ import annotations

from typing import Any, Dict, List

from src.models.pageindex import PageIndex
from src.models.ldu import LDU


class QueryAgent:
    """
    Lightweight final deliverable query agent.
    Exposes 3 required tools:
      - pageindex_navigate
      - semantic_search
      - structured_query
    """

    def __init__(self, page_index: PageIndex, ldus: List[LDU]) -> None:
        self.page_index = page_index
        self.ldus = ldus

    def pageindex_navigate(self, page: int) -> Dict[str, Any]:
        for node in self.page_index.root:
            if node.page == page:
                return node.model_dump()
        return {"page": page, "items": [], "message": "page not found"}

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = (query or "").lower().strip()
        if not q:
            return []

        hits: list[tuple[int, LDU]] = []
        for ldu in self.ldus:
            text = (ldu.content or "").lower()
            score = text.count(q)
            if score > 0:
                hits.append((score, ldu))

        hits.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "ldu_id": ldu.ldu_id,
                "score": score,
                "page_refs": ldu.page_refs,
                "snippet": (ldu.content or "")[:220],
                "content_hash": ldu.content_hash,
            }
            for score, ldu in hits[:top_k]
        ]

    def structured_query(self, field_name: str) -> List[Dict[str, Any]]:
        key = (field_name or "").lower().strip()
        out: list[dict[str, Any]] = []

        for ldu in self.ldus:
            text = (ldu.content or "")
            if ":" in text:
                left, right = text.split(":", 1)
                if left.lower().strip() == key:
                    out.append(
                        {
                            "field": left.strip(),
                            "value": right.strip(),
                            "page_refs": ldu.page_refs,
                            "ldu_id": ldu.ldu_id,
                            "content_hash": ldu.content_hash,
                        }
                    )
        return out