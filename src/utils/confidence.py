from __future__ import annotations

import re
from typing import Iterable, Mapping, Any


def _normalize_texts(blocks: Iterable[Any]) -> list[str]:
    texts: list[str] = []
    for b in blocks:
        if isinstance(b, str):
            texts.append(b)
        elif isinstance(b, Mapping):
            texts.append(str(b.get("text") or b.get("content") or ""))
        else:
            texts.append(str(getattr(b, "text", "") or getattr(b, "content", "") or ""))
    return [t.strip() for t in texts if t and t.strip()]


def score_extraction_confidence(blocks: Iterable[Any]) -> float:
    """
    Cheap, deterministic quality estimate for extracted content.
    0.0 - 1.0
    """
    texts = _normalize_texts(blocks)
    if not texts:
        return 0.0

    joined = "\n".join(texts)
    total_chars = len(joined)
    alpha_chars = sum(c.isalpha() for c in joined)
    digit_chars = sum(c.isdigit() for c in joined)
    words = re.findall(r"\b[\w\-/%().,:]+\b", joined.lower())
    unique_words = len(set(words))
    uncertainty_hits = len(re.findall(r"\bunknown\b|\bunclear\b|\bunreadable\b|\bunsure\b", joined.lower()))
    junk_hits = len(re.findall(r"[#`]{3,}|\\u[0-9a-fA-F]{4}", joined))

    score = 0.15

    if total_chars >= 100:
        score += 0.15
    if total_chars >= 400:
        score += 0.15
    if total_chars >= 1200:
        score += 0.15

    alpha_ratio = alpha_chars / max(1, total_chars)
    if alpha_ratio >= 0.35:
        score += 0.15
    if alpha_ratio >= 0.55:
        score += 0.10

    if unique_words >= 20:
        score += 0.10
    if unique_words >= 80:
        score += 0.10

    if digit_chars > 0 and re.search(r"\b(total|amount|revenue|tax|profit|year)\b", joined.lower()):
        score += 0.05

    score -= min(0.20, uncertainty_hits * 0.05)
    score -= min(0.15, junk_hits * 0.05)

    return max(0.0, min(1.0, round(score, 3)))