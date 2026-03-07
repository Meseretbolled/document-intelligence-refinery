from __future__ import annotations

import json
import sqlite3
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


_DEFAULT_QUESTIONS = [
    "What is the total revenue or income reported in this document?",
    "What are the main findings or key conclusions of this report?",
    "What budget allocations or expenditure figures are mentioned?",
]


def run_refinery_on_pdf(pdf_path: str) -> RefineryOutputs:
    """
    Full 5-stage pipeline.

    Stage 1: triage        -> DocumentProfile JSON
    Stage 2: extraction    -> ExtractedDocument JSON
    Stage 3: LDU + chunk   -> LDU JSON list
    Stage 4: PageIndex     -> PageIndex JSON (flat + hierarchical section tree)
    Stage 5: QueryAgent    -> ChromaDB vector store + SQLite FactTable + Q&A pairs
    """
    rules = settings.load_rules()
    router = ExtractionRouter(rules)
    root = settings.project_root

    profiles_dir   = root / ".refinery/profiles"
    extracted_dir  = root / ".refinery/extracted"
    ldu_dir        = root / ".refinery/ldu"
    page_index_dir = root / ".refinery/pageindex"
    chroma_dir     = root / ".refinery/chroma"
    qa_dir         = root / ".refinery/qa"
    facts_db_path  = str(root / ".refinery/facts.db")

    for d in [profiles_dir, extracted_dir, ldu_dir, page_index_dir, chroma_dir, qa_dir]:
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

    violations = ChunkValidator().validate(ldus)
    if violations:
        print(f"  [chunker] {len(violations)} violation(s) for {profile.doc_id}")

    ldu_path = ldu_dir / f"{profile.doc_id}.json"
    write_json(ldu_path, [ldu.model_dump() for ldu in ldus])

    # Stage 4: PageIndex
    page_index = build_page_index(profile, extracted, ldus)
    page_index_path = page_index_dir / f"{profile.doc_id}.json"
    write_json(page_index_path, page_index.model_dump())

    # Stage 5: QueryAgent — ingest happens INSIDE __init__, do NOT call again after
    vs = VectorStore(
        collection_name=f"refinery_{profile.doc_id}",
        persist_dir=str(chroma_dir),
    )
    ft = FactTable(db_path=facts_db_path)

    # Count facts BEFORE init so we can report how many were added this run
    facts_before = _count_facts(facts_db_path, profile.doc_id)

    # QueryAgent.__init__ calls vs.ingest() and ft.extract_and_store() internally
    agent = QueryAgent(
        page_index=page_index,
        ldus=ldus,
        fact_table=ft,
        vector_store=vs,
        source_path=pdf_path,
    )

    # Report actual ingested count (what __init__ added)
    facts_after = _count_facts(facts_db_path, profile.doc_id)
    facts_added = facts_after - facts_before
    # For vector store: QueryAgent ingests all non-existing LDUs. Count = new LDUs.
    print(f"  [vectorstore] {len(ldus)} LDUs ingested for {profile.doc_id}")
    print(f"  [facttable]   {facts_added} facts extracted for {profile.doc_id}")

    # Auto-generate 3 Q&A pairs
    _generate_and_save_qa(agent, profile.doc_id, qa_dir)
    # Write to extraction_ledger.jsonl
    import time as _time
    from src.models.ledger import ExtractionLedgerEvent
    from src.utils.io import append_jsonl

    ledger_event = ExtractionLedgerEvent(
        doc_id=profile.doc_id,
        source_path=pdf_path,
        strategy_used=extracted.strategy_used,
        confidence=extracted.confidence,
        escalated=extracted.strategy_used in ("B", "C"),
        cost_estimate_usd=(extracted.meta or {}).get("estimated_vision_cost_usd", 0.0),
        processing_time_s=round((extracted.meta or {}).get("latency_s", 0.0), 3),
        notes=notes,
        signals={
            "origin_type": profile.origin_type,
            "layout_complexity": profile.layout_complexity,
            "avg_text_chars_per_page": profile.avg_text_chars_per_page,
            "avg_image_area_ratio": profile.avg_image_area_ratio,
        },
    )
    ledger_path = root / ".refinery" / "extraction_ledger.jsonl"
    append_jsonl(ledger_path, ledger_event.model_dump())

    return RefineryOutputs(
        profile_path=profile_path,
        extracted_path=extracted_path,
        ldu_path=ldu_path,
        page_index_path=page_index_path,
        facts_db_path=facts_db_path,
        profile=profile.model_dump(),
        extracted=extracted.model_dump(),
        ldus=[ldu.model_dump() for ldu in ldus],
        page_index=page_index.model_dump(),
        notes=notes or "",
        chunk_violations=violations,
    )


def _count_facts(db_path: str, doc_id: str) -> int:
    """Count rows in facts table for a given doc_id. Returns 0 if table does not exist."""
    try:
        with sqlite3.connect(db_path) as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE doc_id=?", (doc_id,)
            ).fetchone()
            return result[0] if result else 0
    except Exception:
        return 0


def _generate_and_save_qa(
    agent: QueryAgent,
    doc_id: str,
    qa_dir: Path,
    questions: Optional[List[str]] = None,
) -> None:
    qs = questions or _DEFAULT_QUESTIONS
    results = []
    for q in qs:
        try:
            results.append(agent.ask(q, top_k=5))
        except Exception as exc:
            results.append({"question": q, "error": str(exc)})

    out_path = qa_dir / f"{doc_id}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  [qa] {len(results)} Q&A pairs saved -> {out_path}")


def get_query_agent(pdf_path: str) -> Optional[QueryAgent]:
    """Re-hydrate a QueryAgent from previously saved disk artifacts."""
    root = settings.project_root

    profiles_dir   = root / ".refinery/profiles"
    ldu_dir        = root / ".refinery/ldu"
    page_index_dir = root / ".refinery/pageindex"
    chroma_dir     = root / ".refinery/chroma"
    facts_db_path  = str(root / ".refinery/facts.db")

    doc_name = Path(pdf_path).stem
    candidates = list(profiles_dir.glob(f"{doc_name}*.json"))
    if not candidates:
        candidates = list(profiles_dir.glob("*.json"))
    if not candidates:
        return None

    profile_data = json.loads(candidates[0].read_text())
    doc_id = profile_data.get("doc_id", doc_name)

    ldu_file = ldu_dir / f"{doc_id}.json"
    pi_file  = page_index_dir / f"{doc_id}.json"

    if not ldu_file.exists() or not pi_file.exists():
        return None

    from src.models.ldu import LDU
    from src.models.pageindex import PageIndex

    ldus       = [LDU(**l) for l in json.loads(ldu_file.read_text())]
    page_index = PageIndex(**json.loads(pi_file.read_text()))

    vs = VectorStore(
        collection_name=f"refinery_{doc_id}",
        persist_dir=str(chroma_dir),
    )
    ft = FactTable(db_path=facts_db_path)

    return QueryAgent(
        page_index=page_index,
        ldus=ldus,
        fact_table=ft,
        vector_store=vs,
        source_path=pdf_path,
    )