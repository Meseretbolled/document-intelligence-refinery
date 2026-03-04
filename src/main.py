from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print

from src.settings import settings
from src.utils.io import write_json, ensure_dir
from src.agents.triage import triage_pdf
from src.agents.extractor import ExtractionRouter

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    input_path: str = typer.Option(..., help="Path to a PDF file or a folder containing PDFs."),
    limit: Optional[int] = typer.Option(None, help="Limit number of PDFs processed."),
):
    rules = settings.load_rules()
    router = ExtractionRouter(rules)

    root = settings.project_root
    profiles_dir = root / settings.profiles_dir
    extracted_dir = root / settings.extracted_dir

    ensure_dir(profiles_dir)
    ensure_dir(extracted_dir)

    p = Path(input_path)
    pdfs = [p] if p.is_file() else sorted(list(p.glob("*.pdf")))

    if limit:
        pdfs = pdfs[:limit]

    if not pdfs:
        raise typer.BadParameter("No PDFs found at input_path.")

    print(f"[bold]Processing {len(pdfs)} PDF(s)...[/bold]")

    for pdf in pdfs:
        profile = triage_pdf(str(pdf), rules)
        write_json(profiles_dir / f"{profile.doc_id}.json", profile.model_dump())

        extracted, notes = router.route(profile)
        write_json(extracted_dir / f"{profile.doc_id}.json", extracted.model_dump())

        print(
            f"✅ {pdf.name} -> strategy {extracted.strategy_used}, confidence={extracted.confidence:.2f}"
        )
        if notes:
            print(f"[yellow]notes:[/yellow] {notes}")


if __name__ == "__main__":
    app()