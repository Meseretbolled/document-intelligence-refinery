# tests/test_confidence.py
import types

from src.strategies.fast_text import FastTextExtractor


class DummyPage:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self):
        return self._text


class DummyPDF:
    def __init__(self, pages_text):
        self.pages = [DummyPage(t) for t in pages_text]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_fasttext_confidence_higher_with_more_text(monkeypatch):
    """
    Confidence should increase when there is more extracted text.
    """

    extractor = FastTextExtractor()

    # Mock pdfplumber.open used inside FastTextExtractor
    def fake_open(_path: str):
        return DummyPDF(["hello world"] * 1)  # small text

    monkeypatch.setattr("src.strategies.fast_text.pdfplumber.open", fake_open)

    low = extractor.extract("dummy.pdf")
    low_conf = low.confidence

    def fake_open_more(_path: str):
        return DummyPDF(["hello world " * 200] * 3)  # lots of text

    monkeypatch.setattr("src.strategies.fast_text.pdfplumber.open", fake_open_more)

    high = extractor.extract("dummy.pdf")
    high_conf = high.confidence

    assert high_conf > low_conf
    assert 0.0 <= low_conf <= 1.0
    assert 0.0 <= high_conf <= 1.0