from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# ✅ Load .env BEFORE Settings() is instantiated.
# pydantic_settings reads from os.environ at __init__ time, so load_dotenv
# must run first — otherwise OPENROUTER_API_KEY and other vars are invisible
# to any module that imports `settings` directly (refinery_service, vision_vlm,
# triage, etc.) without going through src/main.py.
_env_file = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_env_file, override=True)


class Settings(BaseSettings):
    # Root directory of the repository (…/document-intelligence-refinery)
    project_root: Path = Path(__file__).resolve().parents[1]

    # Config file containing thresholds and escalation rules
    rubric_path: Path = Path("rubric/extraction_rules.yaml")

    # Output artifact directories/files
    refinery_dir: Path = Path(".refinery")
    profiles_dir: Path = refinery_dir / "profiles"
    extracted_dir: Path = refinery_dir / "extracted"
    ldu_dir: Path = refinery_dir / "ldu"
    pageindex_dir: Path = refinery_dir / "pageindex"
    chroma_dir: Path = refinery_dir / "chroma"
    facts_db_path: Path = refinery_dir / "facts.db"
    ledger_path: Path = refinery_dir / "extraction_ledger.jsonl"

    def load_rules(self) -> Dict[str, Any]:
        rubric_file = self.project_root / self.rubric_path
        if not rubric_file.exists():
            raise FileNotFoundError(f"Missing rubric config: {rubric_file}")
        return yaml.safe_load(rubric_file.read_text(encoding="utf-8"))


settings = Settings()