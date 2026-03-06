from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BudgetDecision:
    allowed: bool
    reason: str
    estimated_cost_usd: float
    remaining_batch_budget_usd: float
    remaining_doc_budget_usd: float


class BudgetController:
    """
    Simple, reliable budget guard.
    - batch_budget_usd: total allowed spend for the current run
    - max_doc_budget_usd: cap per document
    """

    def __init__(self, batch_budget_usd: float, max_doc_budget_usd: float):
        self.batch_budget_usd = float(max(batch_budget_usd, 0.0))
        self.max_doc_budget_usd = float(max(max_doc_budget_usd, 0.0))
        self.spent_batch_usd = 0.0

    def remaining_batch(self) -> float:
        return max(0.0, self.batch_budget_usd - self.spent_batch_usd)

    def decide(self, doc_spent_usd: float, estimated_cost_usd: float) -> BudgetDecision:
        doc_spent_usd = float(max(doc_spent_usd, 0.0))
        estimated_cost_usd = float(max(estimated_cost_usd, 0.0))

        remaining_batch = self.remaining_batch()
        remaining_doc = max(0.0, self.max_doc_budget_usd - doc_spent_usd)

        if estimated_cost_usd <= 0.0:
            return BudgetDecision(
                allowed=True,
                reason="estimated_cost=0",
                estimated_cost_usd=0.0,
                remaining_batch_budget_usd=remaining_batch,
                remaining_doc_budget_usd=remaining_doc,
            )

        if estimated_cost_usd > remaining_doc:
            return BudgetDecision(
                allowed=False,
                reason=f"per-doc cap exceeded (need {estimated_cost_usd:.4f}, remaining {remaining_doc:.4f})",
                estimated_cost_usd=estimated_cost_usd,
                remaining_batch_budget_usd=remaining_batch,
                remaining_doc_budget_usd=remaining_doc,
            )

        if estimated_cost_usd > remaining_batch:
            return BudgetDecision(
                allowed=False,
                reason=f"batch budget exceeded (need {estimated_cost_usd:.4f}, remaining {remaining_batch:.4f})",
                estimated_cost_usd=estimated_cost_usd,
                remaining_batch_budget_usd=remaining_batch,
                remaining_doc_budget_usd=remaining_doc,
            )

        return BudgetDecision(
            allowed=True,
            reason="within budget",
            estimated_cost_usd=estimated_cost_usd,
            remaining_batch_budget_usd=remaining_batch,
            remaining_doc_budget_usd=remaining_doc,
        )

    def charge(self, amount_usd: float) -> None:
        amount_usd = float(max(amount_usd, 0.0))
        self.spent_batch_usd += amount_usd