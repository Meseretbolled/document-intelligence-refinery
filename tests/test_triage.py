from pathlib import Path
from types import SimpleNamespace
import yaml

from src.agents.triage import triage_pdf


def load_rules_from_yaml():
    rules_path = Path(__file__).resolve().parents[1] / "rubric" / "extraction_rules.yaml"
    with open(rules_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_triage_detects_scanned_origin(monkeypatch):
    rules = load_rules_from_yaml()

    def fake_signals(_pdf_path: str):
        return SimpleNamespace(
            page_count=3,
            avg_image_area_ratio=0.90,
            avg_text_chars_per_page=50,
        )

    monkeypatch.setattr("src.agents.triage.compute_pdf_signals", fake_signals)

    profile = triage_pdf("dummy.pdf", rules)

    assert profile.origin_type == "scanned_image"
    assert profile.layout_complexity is not None
    assert profile.cost_tier in (
        "fast_text_sufficient",
        "needs_layout_model",
        "needs_vision_model",
    )


def test_triage_detects_native_origin(monkeypatch):
    rules = load_rules_from_yaml()

    def fake_signals(_pdf_path: str):
        return SimpleNamespace(
            page_count=5,
            avg_image_area_ratio=0.01,
            avg_text_chars_per_page=5000,
        )

    monkeypatch.setattr("src.agents.triage.compute_pdf_signals", fake_signals)

    profile = triage_pdf("dummy.pdf", rules)

    assert profile.origin_type == "native_digital"
    assert profile.layout_complexity is not None
    assert profile.cost_tier == "fast_text_sufficient"