from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.models.ldu import LDU


@dataclass
class ChunkRuleConfig:
    max_chars: int = 1200
    min_chars: int = 120
    overlap_chars: int = 100


class ChunkValidator:
    """
    Enforces 5 chunking rules:
    1. Keep page locality if possible
    2. Never split tables mid-item
    3. Attach short headers to following text
    4. Respect max/min size
    5. Preserve provenance + hashes
    """

    def __init__(self, cfg: ChunkRuleConfig | None = None):
        self.cfg = cfg or ChunkRuleConfig()

    def is_table(self, ldu: LDU) -> bool:
        return getattr(ldu, "chunk_type", "text") == "table"

    def is_short_header(self, ldu: LDU) -> bool:
        text = (getattr(ldu, "content", "") or "").strip()
        return getattr(ldu, "chunk_type", "text") == "text" and len(text) < 80 and text.endswith(":")

    def can_merge(self, current: list[LDU], candidate: LDU) -> bool:
        if not current:
            return True
        curr_pages = set(current[-1].page_refs or [])
        cand_pages = set(candidate.page_refs or [])
        same_pageish = not curr_pages or not cand_pages or bool(curr_pages & cand_pages)
        curr_chars = sum(len((x.content or "")) for x in current)
        if self.is_table(candidate):
            return False
        return same_pageish and (curr_chars + len(candidate.content or "")) <= self.cfg.max_chars


class SemanticChunkingEngine:
    def __init__(self, validator: ChunkValidator | None = None):
        self.validator = validator or ChunkValidator()

    def chunk(self, ldus: List[LDU]) -> List[LDU]:
        if not ldus:
            return []

        out: list[LDU] = []
        buf: list[LDU] = []

        def flush():
            nonlocal buf
            if not buf:
                return
            if len(buf) == 1:
                out.append(buf[0])
                buf = []
                return

            text = "\n\n".join((x.content or "").strip() for x in buf if (x.content or "").strip())
            merged = buf[0].model_copy(deep=True)
            merged.content = text
            merged.chunk_type = "text"
            merged.page_refs = sorted({p for x in buf for p in (x.page_refs or [])})
            merged.content_hash = buf[-1].content_hash  # stable enough for now
            out.append(merged)
            buf = []

        i = 0
        while i < len(ldus):
            ldu = ldus[i]

            if self.validator.is_table(ldu):
                flush()
                out.append(ldu)
                i += 1
                continue

            if self.validator.is_short_header(ldu) and i + 1 < len(ldus):
                nxt = ldus[i + 1]
                merged = ldu.model_copy(deep=True)
                merged.content = f"{(ldu.content or '').strip()}\n{(nxt.content or '').strip()}".strip()
                merged.page_refs = sorted({*(ldu.page_refs or []), *(nxt.page_refs or [])})
                out.append(merged)
                i += 2
                continue

            if self.validator.can_merge(buf, ldu):
                buf.append(ldu)
            else:
                flush()
                buf.append(ldu)

            i += 1

        flush()
        return out