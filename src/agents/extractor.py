from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

from src.engine.budget import BudgetController
from src.engine.cache import ArtifactCache
from src.engine.policy import EscalationPolicy
from src.engine.qa import evaluate_extraction
from src.models.extracted_document import ExtractedDocument
from src.models.profile import DocumentProfile
from src.settings import settings
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout_docling import DoclingLayoutExtractor
from src.strategies.strategy_c_router import StrategyCRouter


@dataclass
class RouteNotes:
    decision_trace: list
    budget_spent_usd: float
    used_cache: bool


class ExtractionRouter:
    def __init__(self, rules):
        self.rules = rules

        mode = os.getenv("REFINERY_MODE", "balanced")
        self.policy = EscalationPolicy.for_mode(mode)

        batch_budget = float(os.getenv("REFINERY_BATCH_BUDGET_USD", "0") or "0")
        doc_budget = float(os.getenv("REFINERY_MAX_DOC_BUDGET_USD", "0") or "0")
        self.budget = BudgetController(
            batch_budget_usd=batch_budget,
            max_doc_budget_usd=doc_budget,
        )

        cache_enabled = (os.getenv("REFINERY_ENABLE_CACHE", "1").strip() == "1")
        self.cache = ArtifactCache(
            settings.project_root / settings.extracted_dir,
            enabled=cache_enabled,
        )

        self.ex_a = FastTextExtractor()
        self.ex_b = DoclingLayoutExtractor()

        # Layered Strategy C: C1 -> C2 -> C3
        allow_c3 = os.getenv("REFINERY_ALLOW_C3", "1").strip() == "1"
        c1_threshold = float(os.getenv("C1_ACCEPT_THRESHOLD", "0.60") or "0.60")
        c2_threshold = float(os.getenv("C2_ACCEPT_THRESHOLD", "0.72") or "0.72")
        self.ex_c = StrategyCRouter(
            c1_threshold=c1_threshold,
            c2_threshold=c2_threshold,
            allow_c3=allow_c3,
        )

    def _estimate_vision_cost(self, profile: DocumentProfile, pages_sent: int = 1) -> float:
        per_page = float(os.getenv("VISION_COST_PER_PAGE_USD", "0.01") or "0.01")
        return float(per_page * max(1, int(pages_sent)))

    def route(self, profile: DocumentProfile) -> Tuple[ExtractedDocument, Optional[str]]:
        trace = []
        doc_spent = 0.0

        # 0) Cache
        cache_hit = self.cache.get_extracted(profile.doc_id)
        if cache_hit.hit and cache_hit.extracted:
            trace.append({"step": "cache", "hit": True, "reason": cache_hit.reason})

            extracted = ExtractedDocument(**cache_hit.extracted)
            qa = evaluate_extraction(cache_hit.extracted)

            extracted.meta = {
                **(getattr(extracted, "meta", {}) or {}),
                "qa_score": qa.qa_score,
                "needs_review": qa.needs_review,
                "qa_reasons": qa.reasons,
                "decision_trace": trace,
            }
            return extracted, "Used cached extracted result"

        trace.append({"step": "cache", "hit": False, "reason": cache_hit.reason})

        # 1) Strategy A
        a_doc, a_note = self.ex_a.extract(profile.doc_id, profile.source_path)
        trace.append(
            {
                "step": "A",
                "confidence": a_doc.confidence,
                "note": a_note,
            }
        )

        if a_doc.confidence >= self.policy.min_conf_a:
            qa = evaluate_extraction(a_doc.model_dump())
            a_doc.meta = {
                **(getattr(a_doc, "meta", {}) or {}),
                "decision_trace": trace,
                "qa_score": qa.qa_score,
                "needs_review": qa.needs_review,
                "qa_reasons": qa.reasons,
            }
            return a_doc, "Accepted Strategy A (high confidence)"

        # 2) Strategy B
        b_doc, b_note = self.ex_b.extract(profile.doc_id, profile.source_path)
        trace.append(
            {
                "step": "B",
                "confidence": b_doc.confidence,
                "note": b_note,
            }
        )

        if b_doc.confidence >= self.policy.min_conf_b:
            qa = evaluate_extraction(b_doc.model_dump())
            b_doc.meta = {
                **(getattr(b_doc, "meta", {}) or {}),
                "decision_trace": trace,
                "qa_score": qa.qa_score,
                "needs_review": qa.needs_review,
                "qa_reasons": qa.reasons,
            }
            return b_doc, "Accepted Strategy B (layout-aware)"

        # 3) Strategy C budget guard
        pages_sent = 2 if getattr(profile, "page_count", 1) >= 2 else 1
        est_cost = self._estimate_vision_cost(profile, pages_sent=pages_sent)

        decision = self.budget.decide(
            doc_spent_usd=doc_spent,
            estimated_cost_usd=est_cost,
        )

        trace.append(
            {
                "step": "C-budget-check",
                "allowed": decision.allowed,
                "reason": decision.reason,
                "estimated_cost_usd": decision.estimated_cost_usd,
                "remaining_batch_usd": decision.remaining_batch_budget_usd,
                "remaining_doc_usd": decision.remaining_doc_budget_usd,
            }
        )

        if not decision.allowed:
            qa = evaluate_extraction(b_doc.model_dump())
            b_doc.meta = {
                **(getattr(b_doc, "meta", {}) or {}),
                "decision_trace": trace,
                "budget_blocked": True,
                "qa_score": min(qa.qa_score, 0.60),
                "needs_review": True,
                "qa_reasons": qa.reasons + ["vision skipped due to budget"],
            }
            return b_doc, f"Skipped Strategy C due to budget: {decision.reason}"

        # 4) Strategy C layered execution: C1 -> C2 -> C3
        c_doc, c_note = self.ex_c.extract(profile.doc_id, profile.source_path)

        strategy_c_level = (
            ((getattr(c_doc, "meta", {}) or {}).get("strategy_c_level"))
            or "unknown"
        )
        strategy_c_route = (
            ((getattr(c_doc, "meta", {}) or {}).get("strategy_c_route"))
            or []
        )

        trace.append(
            {
                "step": "C",
                "confidence": c_doc.confidence,
                "note": c_note,
                "strategy_c_level": strategy_c_level,
                "strategy_c_route": strategy_c_route,
            }
        )

        # Only charge vision budget if C3 actually ran and returned content
        c_meta = getattr(c_doc, "meta", {}) or {}
        if (
            c_doc.confidence > 0
            and len(c_doc.blocks) > 0
            and c_meta.get("strategy_c_level") == "C3"
        ):
            self.budget.charge(est_cost)
            doc_spent += est_cost
            trace.append({"step": "C-charge", "charged_usd": est_cost})

        qa = evaluate_extraction(c_doc.model_dump())
        c_doc.meta = {
            **c_meta,
            "decision_trace": trace,
            "estimated_vision_cost_usd": est_cost if c_meta.get("strategy_c_level") == "C3" else 0.0,
            "doc_spent_usd": doc_spent,
            "batch_spent_usd": self.budget.spent_batch_usd,
            "qa_score": qa.qa_score,
            "needs_review": qa.needs_review or bool(c_meta.get("needs_review", False)),
            "qa_reasons": qa.reasons,
        }

        return c_doc, (c_note or f"Used Strategy C ({strategy_c_level})")