from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path as _Path
import os as _os

# Load .env from project root explicitly — works regardless of CWD
_env_path = _Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

# Warn immediately if key is missing
if not _os.getenv("OPENROUTER_API_KEY", "").strip():
    from rich import print as _rprint
    _rprint("[bold yellow]⚠ WARNING: OPENROUTER_API_KEY is not set.[/bold yellow]")
    _rprint(f"[yellow]  Looked for .env at: {_env_path}[/yellow]")
    _rprint("[yellow]  Scanned PDFs and Amharic documents cannot be extracted without it.[/yellow]")
    _rprint("[yellow]  Get a free key at https://openrouter.ai and add it to your .env[/yellow]\n")

from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from src.service.refinery_service import run_refinery_on_pdf, get_query_agent

console = Console()
app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    input_path: str = typer.Option(..., help="Path to a PDF file or folder of PDFs."),
    limit: Optional[int] = typer.Option(None, help="Max number of PDFs to process."),
):
    """Full 5-stage pipeline for one or more PDFs."""
    p = Path(input_path)
    pdfs = [p] if p.is_file() else sorted(list(p.glob("*.pdf")))
    if limit:
        pdfs = pdfs[:limit]
    if not pdfs:
        raise typer.BadParameter("No PDFs found at input_path.")

    print(f"[bold cyan]Processing {len(pdfs)} PDF(s)...[/bold cyan]\n")
    for pdf in pdfs:
        print(f"[bold]>> {pdf.name}[/bold]")
        try:
            out = run_refinery_on_pdf(str(pdf))
            t = Table(show_header=False, padding=(0, 1))
            t.add_row("Strategy",   str(out.extracted.get("strategy_used")))
            t.add_row("Confidence", f"{out.extracted.get('confidence', 0.0):.2f}")
            t.add_row("Blocks",     str(len(out.extracted.get("blocks") or [])))
            t.add_row("LDUs",       str(len(out.ldus)))
            t.add_row("Violations", str(len(out.chunk_violations)))
            console.print(t)
            if out.notes:
                print(f"  [yellow]notes:[/yellow] {out.notes}")
            if out.chunk_violations:
                print(f"  [red]chunk violations ({len(out.chunk_violations)}):[/red]")
                for v in out.chunk_violations[:5]:
                    print(f"    - {v}")
            print(f"  [green]artifacts written[/green]\n")
        except Exception as exc:
            print(f"  [red]ERROR:[/red] {exc}\n")


@app.command()
def query(
    pdf_path: str = typer.Option(..., help="Path to an already-processed PDF."),
    question: str = typer.Option(..., help="Natural language question."),
):
    """Ask a question against a previously processed document."""
    agent = get_query_agent(pdf_path)
    if agent is None:
        print("[red]Document not processed yet. Run: python -m src.main run --input-path <pdf>[/red]")
        raise typer.Exit(1)
    result = agent.ask(question, top_k=5)
    print(f"\n[bold]Question:[/bold] {result['question']}")
    print(f"[bold]Tools used:[/bold] {', '.join(result['tools_used'])}")
    print("\n[bold]Answer snippets:[/bold]")
    for i, snippet in enumerate(result["answer_snippets"], 1):
        print(f"  {i}. {snippet[:200]}")
    print("\n[bold]Provenance:[/bold]")
    for prov in result["provenance_chain"]:
        spans = prov.get("spans", [])
        page = spans[0].get("page", "?") if spans else "?"
        bbox = spans[0].get("bbox") if spans else None
        h = (prov.get("content_hash") or "")[:12]
        print(f"  - {prov.get('document_name','?')}  page={page}  bbox={bbox}  hash={h}...")
    if result.get("structured_facts"):
        print("\n[bold]Structured facts:[/bold]")
        for f in result["structured_facts"][:5]:
            print(f"  - {f.get('field_name')}: {f.get('value')} ({f.get('unit', '')})")


@app.command()
def audit(
    pdf_path: str = typer.Option(..., help="Path to an already-processed PDF."),
    claim: str = typer.Option(..., help="Factual claim to verify."),
):
    """Audit Mode: verify whether a claim is supported by the document."""
    agent = get_query_agent(pdf_path)
    if agent is None:
        print("[red]Document not processed yet. Run: python -m src.main run --input-path <pdf>[/red]")
        raise typer.Exit(1)
    result = agent.verify_claim(claim)
    color = {"verified": "green", "unverifiable": "yellow", "not_found": "red"}.get(result.verdict, "white")
    print(f"\n[bold]Claim:[/bold] {result.claim}")
    print(f"[bold]Verdict:[/bold] [{color}]{result.verdict.upper()}[/{color}]")
    print(f"[bold]Confidence:[/bold] {result.confidence:.3f}")
    print(f"[bold]Explanation:[/bold] {result.explanation}")
    if result.evidence:
        print("\n[bold]Evidence provenance:[/bold]")
        for ev in result.evidence:
            print(f"  - {ev}")


if __name__ == "__main__":
    app()