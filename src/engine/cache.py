from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CacheHit:
    hit: bool
    reason: str
    extracted: Optional[Dict[str, Any]] = None


class ArtifactCache:
    """
    Caches extracted results keyed by doc_id (and optionally hash inside file).
    Minimal and robust: if extracted JSON exists, treat as cache hit.
    """

    def __init__(self, extracted_dir: Path, enabled: bool = True):
        self.extracted_dir = extracted_dir
        self.enabled = enabled

    def get_extracted(self, doc_id: str) -> CacheHit:
        if not self.enabled:
            return CacheHit(hit=False, reason="cache disabled")

        path = self.extracted_dir / f"{doc_id}.json"
        if not path.exists():
            return CacheHit(hit=False, reason="no cached extracted")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("doc_id") == doc_id:
                return CacheHit(hit=True, reason="cached extracted found", extracted=data)
            return CacheHit(hit=False, reason="cached extracted invalid")
        except Exception:
            return CacheHit(hit=False, reason="cached extracted unreadable")