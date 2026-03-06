from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pdfplumber


@dataclass
class PDFSignals:
    page_count: int
    avg_text_chars_per_page: float
    avg_image_area_ratio: float
    sampled_pages: int


def compute_pdf_signals(pdf_path: str, max_pages: Optional[int] = 10) -> PDFSignals:
    """
    Compute lightweight PDF signals for triage.

    Important:
    - page_count = TRUE total number of pages in the PDF
    - averages are computed on a sampled subset for speed
    """
    with pdfplumber.open(pdf_path) as pdf:
        total_page_count = len(pdf.pages)

        sampled_pages = pdf.pages[: max_pages or total_page_count]

        text_counts = []
        image_ratios = []

        for p in sampled_pages:
            # text density
            try:
                text = p.extract_text() or ""
            except Exception:
                text = ""
            text_counts.append(len(text))

            # image area ratio
            page_area = float((p.width or 1) * (p.height or 1))
            images = getattr(p, "images", None) or []

            img_area = 0.0
            for img in images:
                w = abs((img.get("x1", 0) - img.get("x0", 0)) or 0)
                h = abs((img.get("y1", 0) - img.get("y0", 0)) or 0)
                img_area += float(w * h)

            image_ratios.append(img_area / page_area if page_area else 0.0)

    return PDFSignals(
        page_count=total_page_count,
        avg_text_chars_per_page=float(sum(text_counts) / max(1, len(text_counts))),
        avg_image_area_ratio=float(sum(image_ratios) / max(1, len(image_ratios))),
        sampled_pages=len(sampled_pages),
    )