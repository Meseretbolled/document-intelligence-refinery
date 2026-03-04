# tests/test_triage.py
import pytest

from src.agents.triage import triage_pdf


def test_triage_detects_scanned_origin(monkeypatch):
    """
    If the PDF has many images and low text density, triage should classify it as scanned.
    """

    # Mock the PDF signals function triage uses internally
    def fake_signals(_pdf_path: str):
        return {
            "page_count": 3,
            "text_chars_total": 50,
            "text_chars_per_page": 16.6,
            "image_count_total": 20,
            "image_pages_ratio": 1.0,
            "avg_words_per_page": 3,
        }

    # Update this import path if your triage.py imports signals differently.
    # Common patterns:
    # - from src.utils.pdf_signals import compute_pdf_signals
    # - from ..utils.pdf_signals import compute_pdf_signals
    monkeypatch.setattr("src.agents.triage.compute_pdf_signals", fake_signals)

    profile = triage_pdf("dummy.pdf")

    assert profile.origin_type in ("scanned", "image_scan", "scanned_pdf")
    assert profile.layout_complexity is not None


def test_triage_detects_native_origin(monkeypatch):
    """
    If the PDF has lots of text and few/no images, triage should classify it as native/digital.
    """

    def fake_signals(_pdf_path: str):
        return {
            "page_count": 5,
            "text_chars_total": 20000,
            "text_chars_per_page": 4000,
            "image_count_total": 0,
            "image_pages_ratio": 0.0,
            "avg_words_per_page": 800,
        }

    monkeypatch.setattr("src.agents.triage.compute_pdf_signals", fake_signals)

    profile = triage_pdf("dummy.pdf")

    assert profile.origin_type in ("native", "digital", "born_digital")
    assert profile.layout_complexity is not None