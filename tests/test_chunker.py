"""Tests for SemanticChunkingEngine — all 5 chunking rules."""
from __future__ import annotations

from src.agents.chunker import SemanticChunkingEngine, ChunkValidator
from src.models.ldu import LDU
from src.models.provenance import ProvenanceChain, ProvenanceSpan
from src.utils.hashing import sha256_text


def _ldu(content: str, chunk_type: str = "text", page: int = 1) -> LDU:
    h = sha256_text(content)
    return LDU(
        ldu_id=f"test-{h[:8]}",
        doc_id="test_doc",
        chunk_type=chunk_type,
        content=content,
        token_count=len(content.split()),
        page_refs=[page],
        bounding_box=[0.0, 0.0, 612.0, 792.0],
        content_hash=h,
        provenance=ProvenanceChain(
            source_path="test.pdf",
            document_name="test.pdf",
            content_hash=h,
            spans=[ProvenanceSpan(page=page, bbox=None)],
        ),
    )


def _engine() -> SemanticChunkingEngine:
    return SemanticChunkingEngine(ChunkValidator())


# ── R1: Table always emitted standalone ──────────────────────────────────────

def test_R1_table_is_standalone():
    """Table must never be merged with adjacent text blocks."""
    ldus = [
        _ldu("Some introductory text.", "text"),
        _ldu("Col A | Col B\nVal 1 | Val 2", "table"),
        _ldu("Text after the table.", "text"),
    ]
    out = _engine().chunk(ldus)
    table_chunks = [c for c in out if c.chunk_type == "table"]
    assert len(table_chunks) == 1, "Table must produce exactly 1 standalone LDU"
    assert table_chunks[0].content.startswith("Col A")


# ── R2: Figure absorbs immediately-following caption ─────────────────────────

def test_R2_figure_absorbs_caption():
    """A figure LDU must absorb the next line if it looks like a caption."""
    ldus = [
        _ldu("Figure 1", "figure"),
        _ldu("Figure 1: Annual revenue growth trend.", "text"),
        _ldu("Following paragraph.", "text"),
    ]
    out = _engine().chunk(ldus)
    figure_chunks = [c for c in out if c.chunk_type == "figure"]
    assert len(figure_chunks) == 1
    assert "[Caption]" in figure_chunks[0].content, \
        "Figure LDU must contain absorbed caption text"


# ── R3: Oversized list splits at line boundaries, normal list stays whole ────

def test_R3_list_kept_as_single_ldu():
    """A short list stays as one LDU."""
    list_content = "• Item one\n• Item two\n• Item three"
    ldus = [_ldu(list_content, "list")]
    out = _engine().chunk(ldus)
    list_chunks = [c for c in out if c.chunk_type == "list"]
    assert len(list_chunks) == 1, "Short list must stay as a single LDU"


def test_R3_oversized_list_splits():
    """A list exceeding max_chars must be split at line boundaries."""
    line = "• This is a rather long list item with enough text to consume characters.\n"
    long_list = line * 25  # definitely exceeds 1200 chars
    ldus = [_ldu(long_list, "list")]
    out = _engine().chunk(ldus)
    list_chunks = [c for c in out if c.chunk_type == "list"]
    assert len(list_chunks) > 1, "Oversized list must be split"
    for chunk in list_chunks:
        assert len(chunk.content) <= 1200, f"Split chunk exceeds max_chars: {len(chunk.content)}"


# ── R4: Section header stamps parent_section on all children ─────────────────

def test_R4_header_stamps_parent_section():
    """All non-header LDUs following a header must carry that header as parent_section."""
    ldus = [
        _ldu("3. Financial Overview", "header"),
        _ldu("Revenue increased by 12% year-on-year.", "text"),
        _ldu("Expenditure remained flat.", "text"),
    ]
    out = _engine().chunk(ldus)
    children = [c for c in out if c.chunk_type != "header"]
    for child in children:
        assert child.parent_section == "3. Financial Overview", \
            f"parent_section not stamped: got '{child.parent_section}'"


def test_R4_multiple_headers():
    """Children belong to the most recent header, not an earlier one."""
    ldus = [
        _ldu("1. Introduction", "header"),
        _ldu("Background context.", "text"),
        _ldu("2. Methodology", "header"),
        _ldu("We used a mixed-methods approach.", "text"),
    ]
    out = _engine().chunk(ldus)
    method_children = [
        c for c in out
        if c.chunk_type == "text" and "mixed-methods" in c.content
    ]
    assert method_children, "Expected a text chunk about methodology"
    assert method_children[0].parent_section == "2. Methodology"


# ── R5: Cross-references stored in provenance ────────────────────────────────

def test_R5_cross_refs_stored():
    """Any chunk containing 'see Table X' must have cross_refs populated."""
    ldus = [
        _ldu("As shown in Table 3, revenue grew significantly. See Figure 2.", "text"),
    ]
    out = _engine().chunk(ldus)
    assert out, "Expected at least one output chunk"
    chunk = out[0]
    stored = getattr(chunk.provenance, "cross_refs", None) or []
    assert len(stored) > 0, \
        f"cross_refs not stored in provenance. Got: {stored!r}"


# ── ChunkValidator detects violations ────────────────────────────────────────

def test_validator_detects_R4_violation():
    """Validator must flag a chunk whose parent_section doesn't match the last header."""
    header = _ldu("5. Budget Analysis", "header")
    bad_child = _ldu("Some text.", "text")
    bad_child = bad_child.model_copy(update={"parent_section": "WRONG SECTION"})

    validator = ChunkValidator()
    violations = validator.validate([header, bad_child])
    assert any("R4" in v for v in violations), \
        f"Expected R4 violation, got: {violations}"


def test_validator_clean_output():
    """A correctly chunked set produces zero violations."""
    ldus = [
        _ldu("1. Revenue", "header"),
        _ldu("Revenue was $4.2B in Q3.", "text"),
    ]
    out = _engine().chunk(ldus)
    violations = ChunkValidator().validate(out)
    assert violations == [], f"Unexpected violations: {violations}"