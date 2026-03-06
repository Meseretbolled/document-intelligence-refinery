from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import profile
from typing import Any, Dict

from src.settings import settings
from src.utils.io import ensure_dir, write_json
from src.agents.triage import triage_pdf
from src.agents.extractor import ExtractionRouter
from src.agents.ldu_builder import build_ldus
from src.agents.page_indexer import build_page_index


@dataclass
class RefineryOutputs:
    profile_path: Path
    extracted_path: Path
    ldu_path: Path
    page_index_path: Path

    profile: Dict[str, Any]
    extracted: Dict[str, Any]
    ldus: list
    page_index: Dict[str, Any]
    notes: str


def run_refinery_on_pdf(pdf_path: str) -> RefineryOutputs:
    """
    Single entrypoint used by CLI and Streamlit.
    Runs:
      Stage 1: triage -> profile json
      Stage 2: extraction -> extracted json
      Stage 4: ldu -> ldu json
      Stage 5: page index -> page_index json
    """
    rules = settings.load_rules()
    router = ExtractionRouter(rules)

    root = settings.project_root

    # Use settings for these if they exist; otherwise fall back to your known structure
    profiles_dir = root / getattr(settings, "profiles_dir", ".refinery/profiles")
    extracted_dir = root / getattr(settings, "extracted_dir", ".refinery/extracted")

    # These two were missing in your Settings → use canonical paths
    ldu_dir = root / ".refinery/ldu"
    page_index_dir = root / ".refinery/page_index"

    ensure_dir(profiles_dir)
    ensure_dir(extracted_dir)
    ensure_dir(ldu_dir)
    ensure_dir(page_index_dir)

    profile = triage_pdf(pdf_path, rules)
    profile_path = profiles_dir / f"{profile.doc_id}.json"
    write_json(profile_path, profile.model_dump())

    extracted, notes = router.route(profile)
    extracted_path = extracted_dir / f"{profile.doc_id}.json"
    write_json(extracted_path, extracted.model_dump())

    ldus = build_ldus(extracted)
    ldu_path = ldu_dir / f"{profile.doc_id}.json"
    write_json(ldu_path, [l.model_dump() for l in ldus])

    page_index = build_page_index(profile, extracted, ldus)
    page_index_path = page_index_dir / f"{profile.doc_id}.json"
    write_json(page_index_path, page_index.model_dump())

    return RefineryOutputs(
        profile_path=profile_path,
        extracted_path=extracted_path,
        ldu_path=ldu_path,
        page_index_path=page_index_path,
        profile=profile.model_dump(),
        extracted=extracted.model_dump(),
        ldus=[l.model_dump() for l in ldus],
        page_index=page_index.model_dump(),
        notes=notes or "",
    )