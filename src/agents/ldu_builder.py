from __future__ import annotations

from typing import List

from src.models.extracted_document import ExtractedDocument
from src.models.ldu import LDU
from src.utils.hashing import sha256_text


# ── BUG 1 FIX: chunk_type mapping ────────────────────────────────────────────
# Maps ExtractedBlock.block_type  ->  LDU.chunk_type (LDUType literal)
# The SemanticChunkingEngine uses chunk_type to route Rules R1-R5.
# Without this, every block became "text" and all 5 rules were permanently skipped.
BLOCK_TYPE_MAP: dict[str, str] = {
    "text":           "text",
    "table":          "table",
    "figure":         "figure",
    "list":           "list",
    "header":         "header",
    "section_header": "header",   # Docling alias
    "caption":        "figure",   # captions belong to their parent figure (Rule R2)
    "footer":         "text",     # footers are plain text
}

# BUG 3 FIX: full-page bbox fallback in PDF coordinate space (US Letter points)
# Ensures ProvenanceChain always has a spatial anchor even when extraction
# does not produce span-level bounding boxes.
_FULL_PAGE_BBOX: list[float] = [0.0, 0.0, 612.0, 792.0]


def build_ldus(extracted: ExtractedDocument) -> List[LDU]:
    """
    Stage 3: Build one LDU per non-empty ExtractedBlock.

    Each LDU carries:
      - chunk_type  mapped from block.block_type    (BUG 1 fixed)
      - token_count word-split approximation        (BUG 2 fixed)
      - bounding_box with 3-level fallback          (BUG 3 fixed)
      - page_refs from provenance spans
      - content_hash for downstream verification
      - provenance chain preserved from extraction
    """
    ldus: List[LDU] = []

    for i, block in enumerate(extracted.blocks):
        content = (block.text or "").strip()
        if not content:
            continue

        content_hash = sha256_text(content)

        # ── Page refs and span-level bbox ─────────────────────────────────────
        page_refs: list[int] = []
        bbox: list[float] | None = None

        if block.provenance and block.provenance.spans:
            page_refs = sorted({s.page for s in block.provenance.spans if s.page})
            for s in block.provenance.spans:
                if s.bbox:
                    bbox = list(s.bbox)
                    break

        # ── BUG 3 FIX: bbox fallback chain ───────────────────────────────────
        if bbox is None:
            # Some extractors store a block-level bbox attribute directly
            block_bbox = getattr(block, "bbox", None)
            if block_bbox and len(block_bbox) == 4:
                bbox = list(block_bbox)

        if bbox is None:
            # Last resort: full-page coordinates. This ensures every LDU has a
            # valid spatial anchor so the demo can always show page + bbox.
            bbox = _FULL_PAGE_BBOX.copy()

        # ── BUG 1 FIX: map block_type to chunk_type ──────────────────────────
        raw_type = getattr(block, "block_type", "text") or "text"
        chunk_type = BLOCK_TYPE_MAP.get(raw_type, "text")

        # ── BUG 2 FIX: token_count ────────────────────────────────────────────
        # Word-split is fast and sufficient for ChunkRuleConfig.max_chars guards.
        token_count = len(content.split())

        ldu_id = f"{extracted.doc_id}-ldu-{i}-{content_hash[:10]}"

        ldus.append(
            LDU(
                ldu_id=ldu_id,
                doc_id=extracted.doc_id,
                chunk_type=chunk_type,      # BUG 1 FIXED
                content=content,
                token_count=token_count,    # BUG 2 FIXED
                parent_section=None,
                page_refs=page_refs,
                bounding_box=bbox,          # BUG 3 FIXED
                content_hash=content_hash,
                provenance=block.provenance,
            )
        )

    return ldus