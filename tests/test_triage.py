from types import SimpleNamespace

from src.agents import triage as triage_module


def test_triage_scanned_pdf(monkeypatch):
    monkeypatch.setattr(
        triage_module,
        "compute_pdf_signals",
        lambda _: SimpleNamespace(
            page_count=3,
            avg_text_chars_per_page=0.0,
            avg_image_area_ratio=1.0,
            sampled_pages=3,
        ),
    )
    monkeypatch.setattr(
        triage_module,
        "compute_layout_signals",
        lambda _: SimpleNamespace(
            approx_column_count=1,
            tableish_score=0.10,
            figureish_score=0.90,
        ),
    )
    monkeypatch.setattr(triage_module, "_detect_language", lambda _: (None, 0.0))
    monkeypatch.setattr(triage_module, "_detect_domain", lambda _: None)

    rules = {
        "triage": {
            "scanned_image_threshold": {
                "image_area_ratio_gte": 0.55,
                "text_chars_per_page_lte": 300,
            }
        }
    }

    profile = triage_module.triage_pdf("dummy.pdf", rules)

    assert profile.origin_type == "scanned_image"
    assert profile.layout_complexity == "figure_heavy"
    assert profile.cost_tier == "needs_vision_model"


def test_triage_native_multicolumn(monkeypatch):
    monkeypatch.setattr(
        triage_module,
        "compute_pdf_signals",
        lambda _: SimpleNamespace(
            page_count=155,
            avg_text_chars_per_page=2616.5,
            avg_image_area_ratio=0.0007,
            sampled_pages=10,
        ),
    )
    monkeypatch.setattr(
        triage_module,
        "compute_layout_signals",
        lambda _: SimpleNamespace(
            approx_column_count=2,
            tableish_score=0.12,
            figureish_score=0.05,
        ),
    )
    monkeypatch.setattr(triage_module, "_detect_language", lambda _: ("en", 0.4))
    monkeypatch.setattr(triage_module, "_detect_domain", lambda _: "technical")

    rules = {
        "triage": {
            "scanned_image_threshold": {
                "image_area_ratio_gte": 0.55,
                "text_chars_per_page_lte": 300,
            }
        }
    }

    profile = triage_module.triage_pdf("dummy.pdf", rules)

    assert profile.origin_type == "native_digital"
    assert profile.layout_complexity == "multi_column"
    assert profile.cost_tier == "needs_layout_model"
    assert profile.page_count == 155