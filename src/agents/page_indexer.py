from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

from src.models.extracted_document import ExtractedDocument, ExtractedBlock
from src.models.ldu import LDU
from src.models.pageindex import PageIndex, PageIndexPage, PageIndexItem
from src.models.profile import DocumentProfile


def _pages_from_block(block: ExtractedBlock) -> List[int]:
    prov = getattr(block, "provenance", None)
    if not prov:
        return []

    spans = getattr(prov, "spans", None) or []
    pages: List[int] = []

    for s in spans:
        p = getattr(s, "page", None)
        if isinstance(p, int) and p > 0:
            pages.append(p)

    return sorted(set(pages))


def _pages_from_ldu(ldu: LDU) -> List[int]:
    page_refs = getattr(ldu, "page_refs", None)
    if isinstance(page_refs, list) and page_refs:
        return sorted(set([p for p in page_refs if isinstance(p, int) and p > 0]))

    prov = getattr(ldu, "provenance", None)
    if not prov:
        return []

    spans = getattr(prov, "spans", None) or []
    pages: List[int] = []

    for s in spans:
        p = getattr(s, "page", None)
        if isinstance(p, int) and p > 0:
            pages.append(p)

    return sorted(set(pages))


def _shorten(text: str, max_len: int = 180) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "…"


def build_page_index(profile: DocumentProfile, extracted: ExtractedDocument, ldus: List[LDU]) -> PageIndex:
    """
    Build a simple page-level index from extracted blocks + LDUs.
    Signature is IMPORTANT:
        build_page_index(profile, extracted, ldus)
    """
    doc_id = profile.doc_id
    source_path = profile.source_path

    blocks_by_page: Dict[int, List[ExtractedBlock]] = defaultdict(list)
    ldus_by_page: Dict[int, List[LDU]] = defaultdict(list)
    all_pages: Set[int] = set()

    # Group extracted blocks by page
    for block in extracted.blocks:
        pages = _pages_from_block(block)
        for p in pages:
            blocks_by_page[p].append(block)
            all_pages.add(p)

    # Group LDUs by page
    for ldu in ldus:
        pages = _pages_from_ldu(ldu)
        for p in pages:
            ldus_by_page[p].append(ldu)
            all_pages.add(p)

    # Fallback if nothing had provenance
    if not all_pages:
        for p in range(1, max(1, int(profile.page_count or 1)) + 1):
            all_pages.add(p)

    page_models: List[PageIndexPage] = []
    data_types_present_global: Set[str] = set()

    for page_num in sorted(all_pages):
        items: List[PageIndexItem] = []
        page_types: Set[str] = set()

        # From extracted blocks
        for i, block in enumerate(blocks_by_page.get(page_num, [])):
            btype = block.block_type
            page_types.add(btype)
            data_types_present_global.add(btype)

            prov = getattr(block, "provenance", None)
            spans = getattr(prov, "spans", []) if prov else []
            content_hash = getattr(prov, "content_hash", f"{doc_id}-block-{page_num}-{i}")

            items.append(
                PageIndexItem(
                    ldu_id="",
                    chunk_type=btype,
                    snippet=_shorten(block.text or ""),
                    content_hash=content_hash,
                    bbox=None,
                    meta={"source": "extracted_block", "page": page_num},
                )
            )

        # From LDUs
        for ldu in ldus_by_page.get(page_num, []):
            ctype = getattr(ldu, "chunk_type", "text")
            page_types.add(ctype)
            data_types_present_global.add(ctype)

            items.append(
                PageIndexItem(
                    ldu_id=ldu.ldu_id,
                    chunk_type=ctype,
                    snippet=_shorten(ldu.content or ""),
                    content_hash=ldu.content_hash,
                    bbox=None,
                    meta={"source": "ldu", "page": page_num},
                )
            )

        char_count = sum(len((item.snippet or "")) for item in items)

        page_models.append(
            PageIndexPage(
                page=page_num,
                items=items,
                data_types_present=sorted(page_types),
                char_count=char_count,
                item_count=len(items),
            )
        )

    return PageIndex(
        doc_id=doc_id,
        source_path=source_path,
        root=page_models,
        page_count=len(page_models),
        data_types_present=sorted(data_types_present_global),
        meta={
            "strategy_used": extracted.strategy_used,
            "confidence": extracted.confidence,
        },
    )