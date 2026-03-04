from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Optional

from src.models.extracted_document import ExtractedDocument


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, doc_id: str, source_path: str) -> Tuple[ExtractedDocument, Optional[str]]:
        """
        Returns (ExtractedDocument, notes).
        notes can be used to explain fallback behavior, errors, etc.
        """
        raise NotImplementedError