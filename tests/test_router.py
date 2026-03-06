from types import SimpleNamespace

from src.agents.extractor import ExtractionRouter
from src.models.extracted_document import ExtractedDocument


def _fake_doc(strategy: str, confidence: float, text: str = "sample text"):
    return ExtractedDocument(
        doc_id="doc1",
        source_path="dummy.pdf",
        strategy_used=strategy,
        confidence=confidence,
        blocks=[
            {
                "block_type": "text",
                "text": text,
                "html": None,
                "provenance": None,
            }
        ],
        meta=None,
    )


def test_router_stops_at_a_when_confident(monkeypatch):
    rules = {
        "confidence": {
            "strategy_a_min_confidence": 0.6,
            "strategy_b_min_confidence": 0.7,
        },
        "escalation": {
            "allow_a_to_b": True,
            "allow_b_to_c": True,
        },
        "cost_estimates_usd": {
            "strategy_a": 0.001,
            "strategy_b": 0.01,
            "strategy_c": 0.1,
        },
    }

    router = ExtractionRouter(rules)

    monkeypatch.setattr(
        router.ex_a,
        "extract",
        lambda doc_id, source_path: (_fake_doc("A", 0.95, "good A result"), "A ok"),
    )
    monkeypatch.setattr(
        router.ex_b,
        "extract",
        lambda doc_id, source_path: (_fake_doc("B", 0.80, "B result"), "B ok"),
    )
    monkeypatch.setattr(
        router.ex_c,
        "extract",
        lambda doc_id, source_path: (_fake_doc("C", 0.70, "C result"), "C ok"),
    )

    profile = SimpleNamespace(
        doc_id="doc1",
        source_path="dummy.pdf",
        page_count=1,
    )

    extracted, notes = router.route(profile)

    assert extracted.strategy_used == "A"
    assert extracted.confidence == 0.95
    assert "A" in notes or "Strategy A" in notes


def test_router_escalates_to_b_when_a_is_low(monkeypatch):
    rules = {
        "confidence": {
            "strategy_a_min_confidence": 0.6,
            "strategy_b_min_confidence": 0.7,
        },
        "escalation": {
            "allow_a_to_b": True,
            "allow_b_to_c": False,
        },
        "cost_estimates_usd": {
            "strategy_a": 0.001,
            "strategy_b": 0.01,
            "strategy_c": 0.1,
        },
    }

    router = ExtractionRouter(rules)

    monkeypatch.setattr(
        router.ex_a,
        "extract",
        lambda doc_id, source_path: (_fake_doc("A", 0.30, "weak A"), "A weak"),
    )
    monkeypatch.setattr(
        router.ex_b,
        "extract",
        lambda doc_id, source_path: (_fake_doc("B", 0.82, "good B"), "B ok"),
    )
    monkeypatch.setattr(
        router.ex_c,
        "extract",
        lambda doc_id, source_path: (_fake_doc("C", 0.70, "C result"), "C ok"),
    )

    profile = SimpleNamespace(
        doc_id="doc1",
        source_path="dummy.pdf",
        page_count=1,
    )

    extracted, notes = router.route(profile)

    assert extracted.strategy_used == "B"
    assert extracted.confidence == 0.82