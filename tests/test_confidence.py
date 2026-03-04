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


def test_fasttext_confidence_higher_with_more_text(monkeypatch, tmp_path):
    extractor = FastTextExtractor()

    # Create a real temporary file so Path.exists() is True
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")  # content doesn't matter; we mock pdfplumber

    # Mock pdfplumber.open used inside FastTextExtractor
    def fake_open(_path: str):
        return DummyPDF(["hello world"])  # small text

    monkeypatch.setattr("src.strategies.fast_text.pdfplumber.open", fake_open)

    extracted_low, _note_low = extractor.extract("doc_low", str(pdf_path))
    low_conf = extracted_low.confidence

    def fake_open_more(_path: str):
        return DummyPDF(["hello world " * 200] * 3)  # lots more text

    monkeypatch.setattr("src.strategies.fast_text.pdfplumber.open", fake_open_more)

    extracted_high, _note_high = extractor.extract("doc_high", str(pdf_path))
    high_conf = extracted_high.confidence

    assert 0.0 <= low_conf <= 1.0
    assert 0.0 <= high_conf <= 1.0
    assert high_conf > low_conf