from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.settings import settings
from src.utils.io import ensure_dir, write_json
from src.agents.triage import triage_pdf
from src.agents.extractor import ExtractionRouter
from src.agents.ldu_builder import build_ldus
from src.agents.chunker import SemanticChunkingEngine, ChunkValidator
from src.agents.indexer import build_page_index
from src.agents.query_agent import QueryAgent, FactTable, VectorStore


@dataclass
class RefineryOutputs:
    profile_path: Path
    extracted_path: Path
    ldu_path: Path
    page_index_path: Path
    facts_db_path: str

    profile: Dict[str, Any]
    extracted: Dict[str, Any]
    ldus: list
    page_index: Dict[str, Any]
    notes: str
    chunk_violations: List[str]


def run_refinery_on_pdf(pdf_path: str) -> RefineryOutputs:
    """
    Full 5-stage pipeline:
      Stage 1: triage → profile
      Stage 2: extraction → extracted
      Stage 3: LDU build + semantic chunking
      Stage 4: PageIndex (flat + hierarchical section tree)
      Stage 5: QueryAgent setup (vector store + FactTable ingest)
    """
    rules = settings.load_rules()
    router = ExtractionRouter(rules)
    root = settings.project_root

    profiles_dir = root / ".refinery/profiles"
    extracted_dir = root / ".refinery/extracted"
    ldu_dir = root / ".refinery/ldu"
    page_index_dir = root / ".refinery/pageindex"
    chroma_dir = root / ".refinery/chroma"
    facts_db_path = str(root / ".refinery/facts.db")

    for d in [profiles_dir, extracted_dir, ldu_dir, page_index_dir, chroma_dir]:
        ensure_dir(d)

    # Stage 1: Triage
    profile = triage_pdf(pdf_path, rules)
    profile_path = profiles_dir / f"{profile.doc_id}.json"
    write_json(profile_path, profile.model_dump())

    # Stage 2: Extraction
    extracted, notes = router.route(profile)
    extracted_path = extracted_dir / f"{profile.doc_id}.json"
    write_json(extracted_path, extracted.model_dump())

    # Stage 3: LDU build + semantic chunking
    raw_ldus = build_ldus(extracted)
    engine = SemanticChunkingEngine(ChunkValidator())
    ldus = engine.chunk(raw_ldus)

    # Validate (collect but don't block)
    validator = ChunkValidator()
    violations = validator.validate(ldus)

    ldu_path = ldu_dir / f"{profile.doc_id}.json"
    write_json(ldu_path, [l.model_dump() for l in ldus])

    # Stage 4: PageIndex
    page_index = build_page_index(profile, extracted, ldus)
    page_index_path = page_index_dir / f"{profile.doc_id}.json"
    write_json(page_index_path, page_index.model_dump())

    # Stage 5: Persist to vector store + FactTable
    vs = VectorStore(
        collection_name=f"refinery_{profile.doc_id}",
        persist_dir=str(chroma_dir),
    )
    ft = FactTable(db_path=facts_db_path)
    _ = QueryAgent(
        page_index=page_index,
        ldus=ldus,
        fact_table=ft,
        vector_store=vs,
        source_path=pdf_path,
    )

    return RefineryOutputs(
        profile_path=profile_path,
        extracted_path=extracted_path,
        ldu_path=ldu_path,
        page_index_path=page_index_path,
        facts_db_path=facts_db_path,
        profile=profile.model_dump(),
        extracted=extracted.model_dump(),
        ldus=[l.model_dump() for l in ldus],
        page_index=page_index.model_dump(),
        notes=notes or "",
        chunk_violations=violations,
    )


def get_query_agent(pdf_path: str) -> Optional[QueryAgent]:
    """
    Re-hydrate a QueryAgent for an already-processed document.
    Loads ldus + page_index from disk; connects to persisted ChromaDB + SQLite.
    """
    import json
    root = settings.project_root

    # Derive doc_id from profile
    profiles_dir = root / ".refinery/profiles"
    ldu_dir = root / ".refinery/ldu"
    page_index_dir = root / ".refinery/pageindex"

    doc_name = Path(pdf_path).stem
    # Find matching profile
    candidates = list(profiles_dir.glob(f"{doc_name}*.json"))
    if not candidates:
        candidates = list(profiles_dir.glob("*.json"))
    if not candidates:
        return None

    profile_data = json.loads(candidates[0].read_text())
    doc_id = profile_data.get("doc_id", doc_name)

    ldu_file = ldu_dir / f"{doc_id}.json"
    pi_file = page_index_dir / f"{doc_id}.json"

    if not ldu_file.exists() or not pi_file.exists():
        return None

    from src.models.ldu import LDU
    from src.models.pageindex import PageIndex

    ldu_raw = json.loads(ldu_file.read_text())
    ldus = [LDU(**l) for l in ldu_raw]
    page_index = PageIndex(**json.loads(pi_file.read_text()))

    vs = VectorStore(
        collection_name=f"refinery_{doc_id}",
        persist_dir=str(root / ".refinery/chroma"),
    )
    ft = FactTable(db_path=str(root / ".refinery/facts.db"))

    return QueryAgent(
        page_index=page_index,
        ldus=ldus,
        fact_table=ft,
        vector_store=vs,
        source_path=pdf_path,
    )