from __future__ import annotations

from typing import Dict, Any, Tuple

from src.models.document_profile import DocumentProfile
from src.models.extracted_document import ExtractedDocument
from src.models.ledger import ExtractionLedgerEvent
from src.utils.io import append_jsonl
from src.utils.timing import timer
from src.settings import settings

from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout_docling import DoclingLayoutExtractor
from src.strategies.vision_vlm import VisionVLMExtractor


class ExtractionRouter:
    """
    Routes each document to Strategy A/B/C, escalates when confidence is too low,
    and appends a ledger event for traceability.
    """

    def __init__(self, rules: Dict[str, Any]):
        self.rules = rules
        self.a = FastTextExtractor()
        self.b = DoclingLayoutExtractor()
        self.c = VisionVLMExtractor()

    def _cost(self, strategy: str) -> float:
        costs = self.rules.get("cost_estimates_usd", {})
        return float(costs.get(f"strategy_{strategy.lower()}", 0.0))

    def route(self, profile: DocumentProfile) -> Tuple[ExtractedDocument, str]:
        # Rule0: Native-digital fast path (prefer Strategy A when a real text layer exists)
        origin = getattr(profile.origin_type, "value", profile.origin_type)

        if (
            origin == "native_digital"
            and profile.avg_text_chars_per_page >= 30
            and profile.avg_image_area_ratio < 0.2
        ):
            strategy = "A"
        else:
            # initial choice (cost-tier based)
            if profile.cost_tier == "fast_text_sufficient":
                strategy = "A"
            elif profile.cost_tier == "needs_layout_model":
                strategy = "B"
            else:
                strategy = "C"

        conf_rules = self.rules["confidence"]
        min_a = float(conf_rules["strategy_a_min_confidence"])
        min_b = float(conf_rules["strategy_b_min_confidence"])

        escalation = self.rules["escalation"]
        allow_a_to_b = bool(escalation.get("allow_a_to_b", True))
        allow_b_to_c = bool(escalation.get("allow_b_to_c", True))

        notes = ""
        escalated = False

        with timer() as t:
            extracted, n = self._run(strategy, profile.doc_id, profile.source_path)
            if n:
                notes += n

            # escalation logic
            if strategy == "A" and extracted.confidence < min_a and allow_a_to_b:
                escalated = True
                extracted, n2 = self._run("B", profile.doc_id, profile.source_path)
                if n2:
                    notes += f" | {n2}"

            if extracted.strategy_used == "B" and extracted.confidence < min_b and allow_b_to_c:
                escalated = True
                extracted, n3 = self._run("C", profile.doc_id, profile.source_path)
                if n3:
                    notes += f" | {n3}"

        # ledger
        event = ExtractionLedgerEvent(
            doc_id=profile.doc_id,
            source_path=profile.source_path,
            strategy_used=extracted.strategy_used,
            confidence=extracted.confidence,
            escalated=escalated,
            cost_estimate_usd=self._cost(extracted.strategy_used),
            processing_time_s=float(t["seconds"]),
            notes=notes or None,
            signals={
                "origin_type": profile.origin_type,
                "layout_complexity": profile.layout_complexity,
                "avg_text_chars_per_page": profile.avg_text_chars_per_page,
                "avg_image_area_ratio": profile.avg_image_area_ratio,
            },
        )
        append_jsonl(settings.project_root / settings.ledger_path, event.model_dump())

        return extracted, notes

    def _run(self, strategy: str, doc_id: str, source_path: str) -> Tuple[ExtractedDocument, str | None]:
        if strategy == "A":
            return self.a.extract(doc_id, source_path)
        if strategy == "B":
            return self.b.extract(doc_id, source_path)
        return self.c.extract(doc_id, source_path)
