from __future__ import annotations

from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel


class ExtractionLedgerEvent(BaseModel):
    doc_id: str
    source_path: str
    strategy_used: Literal["A", "B", "C"]
    confidence: float
    escalated: bool
    cost_estimate_usd: float
    processing_time_s: float
    notes: Optional[str] = None
    signals: Dict[str, Any] = {}