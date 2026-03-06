from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EscalationPolicy:
    """
    Tuning knobs for A/B/C.
    """
    mode: str  # economy | balanced | quality

    # confidence thresholds (stop when met)
    min_conf_a: float = 0.90
    min_conf_b: float = 0.80
    min_conf_c: float = 0.70

    @staticmethod
    def for_mode(mode: str) -> "EscalationPolicy":
        m = (mode or "balanced").strip().lower()
        if m == "economy":
            return EscalationPolicy(mode=m, min_conf_a=0.92, min_conf_b=0.85, min_conf_c=0.70)
        if m == "quality":
            return EscalationPolicy(mode=m, min_conf_a=0.88, min_conf_b=0.78, min_conf_c=0.70)
        return EscalationPolicy(mode="balanced", min_conf_a=0.90, min_conf_b=0.80, min_conf_c=0.70)