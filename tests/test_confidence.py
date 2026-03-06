from src.utils.confidence import score_extraction_confidence


def test_confidence_empty():
    assert score_extraction_confidence([]) == 0.0


def test_confidence_good_text():
    blocks = [
        {"text": "Independent Auditor's Report for the year ended 30 June 2020."},
        {"text": "Revenue from contracts with customers was 36,405,691 and profit after tax was 6,333,739."},
        {"text": "Cash and cash equivalents at the end of the year increased compared with the previous period."},
    ]
    assert score_extraction_confidence(blocks) >= 0.55


def test_confidence_bad_text():
    blocks = [
        {"text": "unknown unreadable ### ###"},
        {"text": "unclear"},
    ]
    assert score_extraction_confidence(blocks) < 0.35