from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.settings import settings
from src.agents.query_agent import QueryAgent, VectorStore, FactTable
from src.models.ldu import LDU
from src.models.pageindex import PageIndex, SectionNode


# ── Question templates by document domain ─────────────────────────────────────
# These are templates — the actual section titles from the PageIndex are
# inserted automatically. You never write questions manually.

FINANCIAL_TEMPLATES = [
    "What is the total {topic} reported in this document?",
    "What are the key figures for {topic}?",
    "What does the document say about {topic}?",
]

SURVEY_TEMPLATES = [
    "What are the main findings about {topic}?",
    "What percentage or score is reported for {topic}?",
    "What challenges or issues are identified regarding {topic}?",
]

AUDIT_TEMPLATES = [
    "What audit findings are reported for {topic}?",
    "What recommendations were made about {topic}?",
    "Were there any irregularities or issues found in {topic}?",
]

BUDGET_TEMPLATES = [
    "What is the allocated budget for {topic}?",
    "What expenditure figures are reported for {topic}?",
    "What is the difference between planned and actual for {topic}?",
]

GENERAL_TEMPLATES = [
    "What does the document say about {topic}?",
    "What are the key findings related to {topic}?",
    "What conclusions are drawn about {topic}?",
]


def _detect_domain(doc_id: str, sections: list) -> str:
    """Detect document domain from doc_id and section titles."""
    doc_lower = doc_id.lower()
    all_titles = " ".join(s.title.lower() for s in sections)
    combined = doc_lower + " " + all_titles

    if any(w in combined for w in ["audit", "auditor", "finding", "irregularit"]):
        return "audit"
    if any(w in combined for w in ["budget", "expenditure", "procurement", "allocation"]):
        return "budget"
    if any(w in combined for w in ["revenue", "profit", "income", "balance sheet", "financial statement"]):
        return "financial"
    if any(w in combined for w in ["survey", "assessment", "performance", "initiative", "awareness"]):
        return "survey"
    return "general"


def _pick_templates(domain: str) -> list[str]:
    return {
        "financial": FINANCIAL_TEMPLATES,
        "audit":     AUDIT_TEMPLATES,
        "budget":    BUDGET_TEMPLATES,
        "survey":    SURVEY_TEMPLATES,
        "general":   GENERAL_TEMPLATES,
    }.get(domain, GENERAL_TEMPLATES)


def _collect_section_topics(sections: list[SectionNode], max_topics: int = 10) -> list[str]:
    """
    Walk section tree and collect meaningful topic strings from titles.
    Filters out short/generic titles like "Introduction", "Appendix", "Contents".
    """
    SKIP = {"introduction", "conclusion", "contents", "appendix", "annex",
            "references", "background", "overview", "summary", "methodology",
            "list of", "acknowledgement", "acronym", "table of"}

    topics = []

    def walk(node: SectionNode, depth: int = 0) -> None:
        title = node.title.strip()
        title_lower = title.lower()

        # Skip generic/short titles
        if len(title) < 8:
            return
        if any(skip in title_lower for skip in SKIP):
            return
        # Skip numbered headings like "1.", "2.3."
        if re.match(r"^\d+[\.\s]", title) and len(title) < 20:
            return

        # Prefer leaf sections (most specific) or L2 sections
        if depth >= 1 or not node.child_sections:
            topics.append(title)

        for child in node.child_sections:
            walk(child, depth + 1)

    for s in sections:
        walk(s)

    # Deduplicate, keep most specific
    seen = set()
    unique = []
    for t in topics:
        key = t.lower()[:40]
        if key not in seen:
            seen.add(key)
            unique.append(t)

    return unique[:max_topics]


def _auto_generate_questions(doc_id: str, page_index: PageIndex) -> list[str]:
    """
    Generate 3 questions automatically from the PageIndex structure.
    No manual question writing needed.
    """
    sections = page_index.sections
    domain = _detect_domain(doc_id, sections)
    templates = _pick_templates(domain)
    topics = _collect_section_topics(sections, max_topics=10)

    # Fallback: if no good section titles, use data_types and doc_id keywords
    if len(topics) < 3:
        doc_words = re.findall(r"[A-Za-z]{4,}", doc_id.replace("_", " ").replace("-", " "))
        topics += [w.replace("_", " ") for w in doc_words if len(w) > 5]

    # If still not enough, use generic document-level question
    if not topics:
        topics = ["the main subject", "key findings", "financial data"]

    # Generate 3 questions from the top 3 topics using the domain templates
    questions = []
    for i in range(min(3, len(topics))):
        topic = topics[i]
        template = templates[i % len(templates)]
        q = template.format(topic=topic)
        questions.append(q)

    # Always ensure exactly 3 questions
    while len(questions) < 3:
        fallback_topics = ["the main conclusions", "key data and figures", "implementation status"]
        template = templates[len(questions) % len(templates)]
        questions.append(template.format(topic=fallback_topics[len(questions) - len(questions)]))

    return questions[:3]


def generate_for_doc(
    doc_id: str,
    ldu_file: Path,
    pi_file: Path,
    source_path: str,
    chroma_dir: Path,
    facts_db: str,
    qa_dir: Path,
) -> dict:
    """Load artifacts, auto-generate questions, run Q&A, save results."""
    ldus       = [LDU(**l) for l in json.loads(ldu_file.read_text())]
    page_index = PageIndex(**json.loads(pi_file.read_text()))

    vs = VectorStore(collection_name=f"refinery_{doc_id}", persist_dir=str(chroma_dir))
    ft = FactTable(db_path=facts_db)

    agent = QueryAgent(
        page_index=page_index,
        ldus=ldus,
        fact_table=ft,
        vector_store=vs,
        source_path=source_path,
    )

    # Auto-generate questions from the document's own PageIndex structure
    questions = _auto_generate_questions(doc_id, page_index)
    print(f"   Auto-generated questions:")
    for q in questions:
        print(f"     - {q}")

    results = []
    for q in questions:
        try:
            result = agent.ask(q, top_k=5)
            results.append(result)
            snippets = len(result.get("answer_snippets", []))
            provs    = len(result.get("provenance_chain", []))
            print(f"   Q: {q[:70]}")
            print(f"      -> {snippets} snippets, {provs} provenance records")
        except Exception as exc:
            results.append({"question": q, "error": str(exc)})
            print(f"   Q: {q[:70]} -- ERROR: {exc}")

    out_path = qa_dir / f"{doc_id}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return {"doc_id": doc_id, "qa_count": len(results), "path": str(out_path)}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", default=None,
                        help="Process only this doc_id (default: all processed docs)")
    args = parser.parse_args()

    root       = settings.project_root
    ldu_dir    = root / ".refinery/ldu"
    pi_dir     = root / ".refinery/pageindex"
    chroma_dir = root / ".refinery/chroma"
    qa_dir     = root / ".refinery/qa"
    facts_db   = str(root / ".refinery/facts.db")

    qa_dir.mkdir(parents=True, exist_ok=True)

    ldu_files = sorted(ldu_dir.glob("*.json"))
    if not ldu_files:
        print("ERROR: No LDU files found. Run the pipeline first:")
        print("  python -m src.main run --input-path data/raw/data/")
        sys.exit(1)

    # Filter to specific doc if requested
    if args.doc_id:
        ldu_files = [f for f in ldu_files if args.doc_id in f.stem]
        if not ldu_files:
            print(f"ERROR: No LDU file found for doc_id containing '{args.doc_id}'")
            sys.exit(1)

    print(f"Found {len(ldu_files)} document(s) to process.\n")
    summaries = []

    for ldu_file in ldu_files:
        doc_id  = ldu_file.stem
        pi_file = pi_dir / f"{doc_id}.json"

        if not pi_file.exists():
            print(f"SKIP {doc_id} -- missing pageindex file (run pipeline first)")
            continue

        profile_file = root / ".refinery/profiles" / f"{doc_id}.json"
        source_path = (
            json.loads(profile_file.read_text()).get("source_path", "")
            if profile_file.exists() else str(ldu_file)
        )

        print(f">> {doc_id}")
        try:
            summary = generate_for_doc(
                doc_id=doc_id,
                ldu_file=ldu_file,
                pi_file=pi_file,
                source_path=source_path,
                chroma_dir=chroma_dir,
                facts_db=facts_db,
                qa_dir=qa_dir,
            )
            summaries.append(summary)
            print(f"   saved {summary['qa_count']} pairs -> {summary['path']}\n")
        except Exception as exc:
            print(f"   FAILED: {exc}\n")

    print("=" * 60)
    print(f"Done: {len(summaries)} doc(s) processed")
    print(f"Total Q&A pairs: {sum(s['qa_count'] for s in summaries)}")
    print(f"Output: {qa_dir}")


if __name__ == "__main__":
    main()