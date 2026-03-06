from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class QAResult:
    qa_score: float  # 0..1
    needs_review: bool
    reasons: List[str]


def evaluate_extraction(extracted: Dict[str, Any]) -> QAResult:
    """
    Lightweight QA to detect bad outputs:
    - empty blocks
    - extremely short content
    - hallucination-like patterns (too many "unknown")
    - missing doc_type for structured outputs (if present)
    """
    reasons: List[str] = []

    blocks = extracted.get("blocks", [])
    if not isinstance(blocks, list) or len(blocks) == 0:
        return QAResult(qa_score=0.0, needs_review=True, reasons=["no blocks"])

    all_text = "\n".join([str(b.get("text") or "") for b in blocks if isinstance(b, dict)])
    text_len = len(all_text.strip())

    score = 1.0

    if text_len < 200:
        score -= 0.35
        reasons.append("very short extracted text")

    if text_len < 50:
        score -= 0.35
        reasons.append("almost empty extracted text")

    unknown_hits = len(re.findall(r"\bunknown\b|\bunsure\b|\bcannot\b", all_text.lower()))
    if unknown_hits >= 5:
        score -= 0.20
        reasons.append("many uncertainty markers")

    # If Strategy C outputs structured header/table, ensure something meaningful exists
    strat = str(extracted.get("strategy_used") or "")
    if strat == "C" and text_len < 120:
        score -= 0.20
        reasons.append("vision output too small")

    score = max(0.0, min(1.0, score))
    needs_review = score < 0.65

    if needs_review and not reasons:
        reasons.append("low QA score")

    return QAResult(qa_score=score, needs_review=needs_review, reasons=reasons)