from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()  # loads .env so OPENROUTER_* is available

from pathlib import Path
from typing import Optional

import typer
from rich import print

from src.settings import settings
from src.utils.io import write_json, ensure_dir
from src.agents.triage import triage_pdf
from src.agents.extractor import ExtractionRouter
from src.agents.ldu_builder import build_ldus
from src.agents.page_indexer import build_page_index

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    input_path: str = typer.Option(..., help="Path to a PDF file or a folder containing PDFs."),
    limit: Optional[int] = typer.Option(None, help="Limit number of PDFs processed."),
):
    """
    Week 3 Final: Triage + Routing + Extraction + LDU + PageIndex artifacts.

    Writes:
      - .refinery/profiles/{doc_id}.json
      - .refinery/extracted/{doc_id}.json
      - .refinery/ldu/{doc_id}.json
      - .refinery/page_index/{doc_id}.json
      - .refinery/extraction_ledger.jsonl (append)
    """
    rules = settings.load_rules()
    router = ExtractionRouter(rules)

    root = settings.project_root

    profiles_dir = root / settings.profiles_dir
    extracted_dir = root / settings.extracted_dir

    # NEW final-stage artifact dirs
    ldu_dir = root / ".refinery/ldu"
    page_index_dir = root / ".refinery/page_index"

    ensure_dir(profiles_dir)
    ensure_dir(extracted_dir)
    ensure_dir(ldu_dir)
    ensure_dir(page_index_dir)

    p = Path(input_path)
    pdfs = [p] if p.is_file() else sorted(list(p.glob("*.pdf")))

    if limit:
        pdfs = pdfs[:limit]

    if not pdfs:
        raise typer.BadParameter("No PDFs found at input_path.")

    print(f"[bold]Processing {len(pdfs)} PDF(s)...[/bold]")

    for pdf in pdfs:
        # Stage 1: Triage
        profile = triage_pdf(str(pdf), rules)
        write_json(profiles_dir / f"{profile.doc_id}.json", profile.model_dump())

        # Stage 2/3: Strategy routing + Extraction
        extracted, notes = router.route(profile)
        write_json(extracted_dir / f"{profile.doc_id}.json", extracted.model_dump())

        # Stage 4: LDU building
        ldus = build_ldus(extracted)
        write_json(ldu_dir / f"{profile.doc_id}.json", [l.model_dump() for l in ldus])

        # Stage 5: PageIndex building
        page_index = build_page_index(profile.doc_id, profile.source_path, ldus)
        write_json(page_index_dir / f"{profile.doc_id}.json", page_index.model_dump())

        print(f"✅ {pdf.name} -> strategy {extracted.strategy_used}, confidence={extracted.confidence:.2f}")
        print(f"   artifacts: profile + extracted + ldu + page_index")

        if notes:
            print(f"[yellow]notes:[/yellow] {notes}")


if __name__ == "__main__":
    app()