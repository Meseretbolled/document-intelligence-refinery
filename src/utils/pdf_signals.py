from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pdfplumber


@dataclass
class PDFSignals:
    page_count: int
    avg_text_chars_per_page: float
    avg_image_area_ratio: float


def compute_pdf_signals(pdf_path: str, max_pages: Optional[int] = 10) -> PDFSignals:
    text_counts: List[int] = []
    image_ratios: List[float] = []

    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[: max_pages or len(pdf.pages)]
        for p in pages:
            # text
            text = p.extract_text() or ""
            text_counts.append(len(text))

            # images: approximate total image area / page area
            page_area = (p.width or 1) * (p.height or 1)
            img_area = 0.0
            for img in (p.images or []):
                # pdfplumber gives bbox coords in PDF units
                w = abs((img.get("x1", 0) - img.get("x0", 0)) or 0)
                h = abs((img.get("y1", 0) - img.get("y0", 0)) or 0)
                img_area += float(w * h)
            image_ratios.append(min(1.0, img_area / page_area if page_area else 0.0))

        page_count = len(pages)

    return PDFSignals(
        page_count=page_count,
        avg_text_chars_per_page=sum(text_counts) / max(1, page_count),
        avg_image_area_ratio=sum(image_ratios) / max(1, page_count),
    )