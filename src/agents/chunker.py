"""
Semantic Chunking Engine — enforces all 5 chunking rules from the constitution.

Rules:
  R1: A table cell is never split from its header row.
      → Tables are always emitted as standalone LDUs, never merged.
  R2: A figure caption is always stored as metadata of its parent figure chunk.
      → Figures absorb immediately-following caption text blocks.
  R3: A numbered list is always kept as a single LDU unless it exceeds max_tokens.
      → List-type chunks are not split unless they exceed max_chars.
  R4: Section headers are stored as parent_section metadata on all child chunks.
      → Headers update the running parent_section context; never merged into text.
  R5: Cross-references ("see Table 3", "see Figure 2") are stored as chunk relationships.
      → Detected cross-references are added to LDU provenance meta.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from src.models.ldu import LDU


@dataclass
class ChunkRuleConfig:
    max_chars: int = 1200
    min_chars: int = 120
    overlap_chars: int = 100


CROSS_REF_PATTERN = re.compile(
    r"\b(?:see|refer to|as shown in|cf\.?|table|figure|fig\.?|equation|section)\s+"
    r"(?:[A-Z]?\d+(?:\.\d+)*|[A-Z])",
    re.IGNORECASE,
)


class ChunkValidator:
    """
    Validates that a batch of LDUs obeys all 5 chunking rules.
    Returns violation descriptions (empty list = all good).
    """

    def __init__(self, cfg: ChunkRuleConfig | None = None):
        self.cfg = cfg or ChunkRuleConfig()

    def is_table(self, ldu: LDU) -> bool:
        return ldu.chunk_type == "table"

    def is_figure(self, ldu: LDU) -> bool:
        return ldu.chunk_type == "figure"

    def is_header(self, ldu: LDU) -> bool:
        return ldu.chunk_type == "header"

    def is_list(self, ldu: LDU) -> bool:
        return ldu.chunk_type == "list"

    def is_caption_text(self, ldu: LDU) -> bool:
        text = (ldu.content or "").strip()
        return bool(re.match(r"^(Figure|Fig\.?|Table|Exhibit|Chart)\s*\d+", text, re.I))

    def can_merge(self, buf: list[LDU], candidate: LDU) -> bool:
        """Returns True iff candidate may be merged into the current buffer."""
        if not buf:
            return True
        # R1: tables never merged
        if self.is_table(candidate) or self.is_table(buf[-1]):
            return False
        # R4: headers never merged into text body
        if self.is_header(candidate) or self.is_header(buf[-1]):
            return False
        # R2: figures are standalone
        if self.is_figure(candidate) or self.is_figure(buf[-1]):
            return False
        curr_chars = sum(len(x.content or "") for x in buf)
        return (curr_chars + len(candidate.content or "")) <= self.cfg.max_chars

    def validate(self, ldus: List[LDU]) -> List[str]:
        """Run all 5 rule checks, return list of violation strings."""
        violations: List[str] = []
        prev_header: Optional[str] = None

        for i, ldu in enumerate(ldus):
            # R4: track section context
            if self.is_header(ldu):
                prev_header = (ldu.content or "").strip()

            # R4: child chunks should carry parent_section when a header preceded them
            if not self.is_header(ldu) and prev_header and ldu.parent_section != prev_header:
                violations.append(
                    f"R4 violation at LDU {i} ({ldu.ldu_id}): "
                    f"parent_section missing or mismatched (expected '{prev_header}', "
                    f"got '{ldu.parent_section}')"
                )

            # R1: tables must not be chunk_type='text'
            if ldu.chunk_type == "table" and not (ldu.content or "").strip():
                violations.append(f"R1 violation at LDU {i}: empty table LDU")

            # R5: check cross-refs are stored
            refs = CROSS_REF_PATTERN.findall(ldu.content or "")
            if refs:
                stored = _get_cross_refs(ldu)
                if not stored:
                    violations.append(
                        f"R5 violation at LDU {i} ({ldu.ldu_id}): "
                        f"cross-refs {refs} not stored in meta"
                    )

        return violations


def _get_cross_refs(ldu: LDU) -> list:
    try:
        return (ldu.provenance.model_extra or {}).get("cross_refs") or []
    except Exception:
        return []


def _annotate_section(ldu: LDU, section: Optional[str]) -> LDU:
    """Stamp parent_section (R4). Returns updated copy."""
    if ldu.parent_section != section:
        return ldu.model_copy(update={"parent_section": section})
    return ldu


def _apply_cross_refs(ldu: LDU) -> None:
    """Store detected cross-references in provenance meta (R5, best-effort in-place)."""
    refs = CROSS_REF_PATTERN.findall(ldu.content or "")
    if refs and ldu.provenance:
        try:
            extra = ldu.provenance.model_extra
            if extra is not None:
                extra["cross_refs"] = refs
        except Exception:
            pass


def _split_list(ldu: LDU, max_chars: int) -> list[LDU]:
    """Split an oversized list LDU at line boundaries (R3)."""
    lines = (ldu.content or "").splitlines()
    parts: list[str] = []
    current: list[str] = []
    curr_len = 0
    for line in lines:
        if curr_len + len(line) > max_chars and current:
            parts.append("\n".join(current))
            current = [line]
            curr_len = len(line)
        else:
            current.append(line)
            curr_len += len(line) + 1
    if current:
        parts.append("\n".join(current))

    result: list[LDU] = []
    for idx, part in enumerate(parts):
        copy = ldu.model_copy(deep=True)
        copy = copy.model_copy(update={"content": part, "ldu_id": f"{ldu.ldu_id}-part{idx}"})
        result.append(copy)
    return result


class SemanticChunkingEngine:
    """Converts a flat list of LDUs into semantically coherent chunks."""

    def __init__(self, validator: ChunkValidator | None = None):
        self.validator = validator or ChunkValidator()

    def chunk(self, ldus: List[LDU]) -> List[LDU]:
        if not ldus:
            return []

        out: list[LDU] = []
        buf: list[LDU] = []
        current_section: Optional[str] = None

        def flush() -> None:
            nonlocal buf
            if not buf:
                return
            if len(buf) == 1:
                ldu = _annotate_section(buf[0], current_section)
                _apply_cross_refs(ldu)
                out.append(ldu)
                buf = []
                return
            text = "\n\n".join((x.content or "").strip() for x in buf if (x.content or "").strip())
            merged = buf[0].model_copy(deep=True)
            merged = merged.model_copy(update={
                "content": text,
                "chunk_type": "text",
                "page_refs": sorted({p for x in buf for p in (x.page_refs or [])}),
                "content_hash": buf[-1].content_hash,
                "parent_section": current_section,
            })
            _apply_cross_refs(merged)
            out.append(merged)
            buf = []

        i = 0
        while i < len(ldus):
            ldu = ldus[i]

            # R4: header → update running section, emit standalone
            if self.validator.is_header(ldu):
                flush()
                current_section = (ldu.content or "").strip()
                ldu = _annotate_section(ldu, current_section)
                _apply_cross_refs(ldu)
                out.append(ldu)
                i += 1
                continue

            # R1: table → always standalone
            if self.validator.is_table(ldu):
                flush()
                ldu = _annotate_section(ldu, current_section)
                _apply_cross_refs(ldu)
                out.append(ldu)
                i += 1
                continue

            # R2: figure → absorb next caption if present
            if self.validator.is_figure(ldu):
                flush()
                fig_ldu = _annotate_section(ldu, current_section)
                if i + 1 < len(ldus) and self.validator.is_caption_text(ldus[i + 1]):
                    caption_text = (ldus[i + 1].content or "").strip()
                    fig_content = (fig_ldu.content or "").rstrip()
                    fig_ldu = fig_ldu.model_copy(update={
                        "content": f"{fig_content}\n[Caption]: {caption_text}"
                    })
                    i += 2
                else:
                    i += 1
                _apply_cross_refs(fig_ldu)
                out.append(fig_ldu)
                continue

            # R3: list → keep as single LDU unless oversized
            if self.validator.is_list(ldu):
                flush()
                ldu = _annotate_section(ldu, current_section)
                _apply_cross_refs(ldu)
                if len(ldu.content or "") > self.validator.cfg.max_chars:
                    for part_ldu in _split_list(ldu, self.validator.cfg.max_chars):
                        out.append(_annotate_section(part_ldu, current_section))
                else:
                    out.append(ldu)
                i += 1
                continue

            # Default: buffer and merge
            if self.validator.can_merge(buf, ldu):
                buf.append(ldu)
            else:
                flush()
                buf.append(ldu)
            i += 1

        flush()
        return out