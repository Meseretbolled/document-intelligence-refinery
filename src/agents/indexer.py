"""
PageIndex Builder — Stage 4.

Builds BOTH:
  1. A flat page-level index (PageIndexPage nodes) for quick page lookup.
  2. A hierarchical SectionNode tree for LLM-friendly navigation.

If OPENROUTER_API_KEY is set, generates LLM summaries for each section.
Falls back to extractive heuristic summaries otherwise.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set

from src.models.extracted_document import ExtractedDocument, ExtractedBlock
from src.models.ldu import LDU
from src.models.pageindex import BBox, PageIndex, PageIndexItem, PageIndexPage, SectionNode
from src.models.profile import DocumentProfile


# ── Utilities ────────────────────────────────────────────────────────────────

def _shorten(text: str, max_len: int = 180) -> str:
    t = (text or "").strip().replace("\n", " ")
    return t[:max_len - 1].rstrip() + "…" if len(t) > max_len else t


def _pages_from_block(block: ExtractedBlock) -> List[int]:
    prov = getattr(block, "provenance", None)
    if not prov:
        return []
    spans = getattr(prov, "spans", None) or []
    return sorted({s.page for s in spans if isinstance(getattr(s, "page", None), int) and s.page > 0})


def _pages_from_ldu(ldu: LDU) -> List[int]:
    if ldu.page_refs:
        return sorted({p for p in ldu.page_refs if isinstance(p, int) and p > 0})
    prov = getattr(ldu, "provenance", None)
    if not prov:
        return []
    spans = getattr(prov, "spans", None) or []
    return sorted({s.page for s in spans if isinstance(getattr(s, "page", None), int) and s.page > 0})


def _detect_heading_level(text: str) -> Optional[int]:
    """Heuristic heading level detection from text patterns."""
    t = text.strip()
    # Numbered heading like "1.", "1.1", "2.3.4"
    if re.match(r"^\d+\.\s", t):
        dots = t.split(" ")[0].count(".")
        return min(dots + 1, 4)
    # ALL CAPS short line
    if t.isupper() and 3 < len(t) < 80:
        return 1
    # Title case short line
    if len(t) < 80 and t[0].isupper() and not t.endswith("."):
        return 2
    return None


def _extract_entities(text: str, max_ents: int = 8) -> List[str]:
    """Very lightweight entity extraction: capitalized phrases & numbers."""
    entities: List[str] = []
    # Capitalized multi-word phrases
    for m in re.finditer(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+", text):
        if m.group() not in entities:
            entities.append(m.group())
    # Monetary / numeric values
    for m in re.finditer(r"\$[\d,\.]+\s*(?:million|billion|B|M|K)?|\d[\d,\.]*%", text, re.I):
        if m.group() not in entities:
            entities.append(m.group())
    return entities[:max_ents]


def _llm_summary(text: str, title: str) -> str:
    """Call OpenRouter for a 2-3 sentence section summary. Falls back gracefully."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return _extractive_summary(text)
    try:
        import httpx
        prompt = (
            f"Summarize the following document section titled '{title}' "
            f"in 2-3 concise sentences. Focus on key facts, numbers, and findings.\n\n"
            f"{text[:3000]}"
        )
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("MODEL_NAME", "openrouter/auto:free"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 120,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return _extractive_summary(text)


def _extractive_summary(text: str, max_chars: int = 300) -> str:
    """Fallback: first non-trivial sentence(s) up to max_chars."""
    sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    out = ""
    for s in sentences:
        if len(s) > 20:
            out += s + " "
        if len(out) >= max_chars:
            break
    return out.strip()[:max_chars]


# ── Section Tree Builder ──────────────────────────────────────────────────────

def _build_section_tree(ldus: List[LDU], use_llm: bool) -> List[SectionNode]:
    """
    Walk LDUs, detect header LDUs, and build a hierarchical SectionNode tree.
    Non-header LDUs are assigned to the most recent section at the appropriate level.
    """
    if not ldus:
        return []

    # Identify header LDUs and their levels
    sections: List[SectionNode] = []
    current_path: List[SectionNode] = []  # stack of open sections by level

    def close_section(node: SectionNode, last_page: int) -> None:
        node.page_end = max(node.page_end, last_page)

    def current_section() -> Optional[SectionNode]:
        return current_path[-1] if current_path else None

    for ldu in ldus:
        pages = ldu.page_refs or []
        page = pages[0] if pages else 1
        last_page = pages[-1] if pages else page

        if ldu.chunk_type == "header":
            level = _detect_heading_level(ldu.content or "") or 1
            node = SectionNode(
                title=(ldu.content or "").strip()[:120],
                page_start=page,
                page_end=last_page,
                level=level,
                ldu_ids=[ldu.ldu_id],
            )

            # Pop stack back to parent level
            while current_path and current_path[-1].level >= level:
                closed = current_path.pop()
                close_section(closed, page - 1)

            # Attach to parent or root
            if current_path:
                current_path[-1].child_sections.append(node)
            else:
                sections.append(node)

            current_path.append(node)

        else:
            # Assign LDU to current open section
            cs = current_section()
            if cs:
                cs.ldu_ids.append(ldu.ldu_id)
                cs.page_end = max(cs.page_end, last_page)
                ct = ldu.chunk_type
                if ct not in cs.data_types_present:
                    cs.data_types_present.append(ct)

    # Close all remaining open sections
    for node in current_path:
        close_section(node, node.page_end)

    # Build summaries and entities for each section
    ldu_map: Dict[str, LDU] = {ldu.ldu_id: ldu for ldu in ldus}

    def enrich_node(node: SectionNode) -> None:
        combined = " ".join(
            (ldu_map[lid].content or "") for lid in node.ldu_ids if lid in ldu_map
        )
        node.key_entities = _extract_entities(combined)
        node.summary = _llm_summary(combined, node.title) if use_llm else _extractive_summary(combined)
        for child in node.child_sections:
            enrich_node(child)

    use_llm_flag = use_llm and bool(os.getenv("OPENROUTER_API_KEY"))
    for s in sections:
        enrich_node(s)

    # Fallback: if no headers detected, create a single root section
    if not sections and ldus:
        all_pages = [p for ldu in ldus for p in (ldu.page_refs or [1])]
        combined = " ".join((ldu.content or "") for ldu in ldus[:50])
        root = SectionNode(
            title="Document",
            page_start=min(all_pages) if all_pages else 1,
            page_end=max(all_pages) if all_pages else 1,
            level=1,
            ldu_ids=[ldu.ldu_id for ldu in ldus],
            data_types_present=sorted({ldu.chunk_type for ldu in ldus}),
            summary=_extractive_summary(combined),
            key_entities=_extract_entities(combined),
        )
        sections = [root]

    return sections


# ── Main Builder ──────────────────────────────────────────────────────────────

class FinalIndexer:
    def __init__(self) -> None:
        self.has_llm = bool(os.getenv("OPENROUTER_API_KEY"))

    def build(self, profile: DocumentProfile, extracted: ExtractedDocument, ldus: List[LDU]) -> PageIndex:
        return build_page_index(profile, extracted, ldus)


def build_page_index(
    profile: DocumentProfile,
    extracted: ExtractedDocument,
    ldus: List[LDU],
) -> PageIndex:
    """
    Build a full PageIndex with:
      - Flat page-level index (root)
      - Hierarchical section tree (sections)
    """
    doc_id = profile.doc_id
    source_path = profile.source_path

    blocks_by_page: Dict[int, List[ExtractedBlock]] = defaultdict(list)
    ldus_by_page: Dict[int, List[LDU]] = defaultdict(list)
    all_pages: Set[int] = set()

    for block in extracted.blocks:
        for p in _pages_from_block(block):
            blocks_by_page[p].append(block)
            all_pages.add(p)

    for ldu in ldus:
        for p in _pages_from_ldu(ldu):
            ldus_by_page[p].append(ldu)
            all_pages.add(p)

    if not all_pages:
        for p in range(1, max(1, int(profile.page_count or 1)) + 1):
            all_pages.add(p)

    page_models: List[PageIndexPage] = []
    data_types_global: Set[str] = set()

    for page_num in sorted(all_pages):
        items: List[PageIndexItem] = []
        page_types: Set[str] = set()

        for idx, block in enumerate(blocks_by_page.get(page_num, [])):
            btype = block.block_type
            page_types.add(btype)
            data_types_global.add(btype)
            prov = getattr(block, "provenance", None)
            content_hash = getattr(prov, "content_hash", f"{doc_id}-b-{page_num}-{idx}")

            # Extract bbox if available
            bbox = None
            spans = getattr(prov, "spans", []) if prov else []
            for span in spans:
                if hasattr(span, "bbox") and span.bbox and len(span.bbox) == 4:
                    bbox = BBox(x0=span.bbox[0], y0=span.bbox[1], x1=span.bbox[2], y1=span.bbox[3])
                    break

            items.append(PageIndexItem(
                ldu_id="",
                chunk_type=btype,
                snippet=_shorten(block.text or ""),
                content_hash=content_hash,
                bbox=bbox,
                meta={"source": "extracted_block", "page": page_num},
            ))

        for ldu in ldus_by_page.get(page_num, []):
            ctype = ldu.chunk_type
            page_types.add(ctype)
            data_types_global.add(ctype)

            bbox = None
            if ldu.bounding_box and len(ldu.bounding_box) == 4:
                bbox = BBox(x0=ldu.bounding_box[0], y0=ldu.bounding_box[1],
                            x1=ldu.bounding_box[2], y1=ldu.bounding_box[3])

            items.append(PageIndexItem(
                ldu_id=ldu.ldu_id,
                chunk_type=ctype,
                snippet=_shorten(ldu.content or ""),
                content_hash=ldu.content_hash,
                bbox=bbox,
                meta={"source": "ldu", "page": page_num},
            ))

        page_models.append(PageIndexPage(
            page=page_num,
            items=items,
            data_types_present=sorted(page_types),
            char_count=sum(len(it.snippet or "") for it in items),
            item_count=len(items),
        ))

    # Build hierarchical section tree
    use_llm = bool(os.getenv("OPENROUTER_API_KEY"))
    section_tree = _build_section_tree(ldus, use_llm)

    return PageIndex(
        doc_id=doc_id,
        source_path=source_path,
        root=page_models,
        sections=section_tree,
        page_count=len(page_models),
        data_types_present=sorted(data_types_global),
        meta={
            "strategy_used": extracted.strategy_used,
            "confidence": extracted.confidence,
            "summary_mode": "llm" if use_llm else "extractive",
            "section_count": len(section_tree),
        },
    )