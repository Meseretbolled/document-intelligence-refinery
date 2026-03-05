from __future__ import annotations

from typing import List

from src.models.extracted_document import ExtractedDocument
from src.models.ldu import LDU
from src.utils.hashing import sha256_text


def build_ldus(extracted: ExtractedDocument) -> List[LDU]:
    """
    Stage 4: Build LDUs from extracted blocks.

    Final-ready minimum:
    - 1 LDU per ExtractedBlock (traceable + stable)
    - content_hash computed
    - provenance preserved
    """
    ldus: List[LDU] = []

    for i, block in enumerate(extracted.blocks):
        content = (block.text or "").strip()
        if not content:
            continue

        content_hash = sha256_text(content)

        page_refs = []
        bbox = None

        if block.provenance and block.provenance.spans:
            page_refs = sorted({s.page for s in block.provenance.spans if s.page})
            for s in block.provenance.spans:
                if s.bbox:
                    bbox = s.bbox
                    break

        ldu_id = f"{extracted.doc_id}-ldu-{i}-{content_hash[:10]}"

        ldus.append(
            LDU(
                ldu_id=ldu_id,
                doc_id=extracted.doc_id,
                chunk_type="text",
                content=content,
                token_count=0,  # optional; keep 0 for now
                parent_section=None,
                page_refs=page_refs,
                bounding_box=bbox,
                content_hash=content_hash,
                provenance=block.provenance,
            )
        )

    return ldus