from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Root directory of the repository (…/document-intelligence-refinery)
    project_root: Path = Path(__file__).resolve().parents[1]

    # Config file containing thresholds and escalation rules
    rubric_path: Path = Path("rubric/extraction_rules.yaml")

    # Output artifact directories/files
    refinery_dir: Path = Path(".refinery")
    profiles_dir: Path = refinery_dir / "profiles"
    extracted_dir: Path = refinery_dir / "extracted"
    ledger_path: Path = refinery_dir / "extraction_ledger.jsonl"

    def load_rules(self) -> Dict[str, Any]:
        rubric_file = self.project_root / self.rubric_path
        if not rubric_file.exists():
            raise FileNotFoundError(f"Missing rubric config: {rubric_file}")
        return yaml.safe_load(rubric_file.read_text(encoding="utf-8"))


# ✅ This is what src/main.py expects to import
settings = Settings()

# Load .env if it exists (optional)
load_dotenv()