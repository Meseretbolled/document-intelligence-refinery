from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import pdfplumber


@dataclass
class LayoutSignals:
    approx_column_count: int
    tableish_score: float
    figureish_score: float


def _cluster_columns(x_starts: List[float], tol: float = 35.0) -> int:
    """
    Super-light column estimator:
    clusters word x0 positions into columns.
    tol is in PDF points.
    """
    if not x_starts:
        return 1
    xs = sorted(x_starts)
    clusters = [xs[0]]
    for x in xs[1:]:
        if abs(x - clusters[-1]) > tol:
            clusters.append(x)
    return max(1, min(4, len(clusters)))  # cap to avoid crazy counts


def compute_layout_signals(pdf_path: str, max_pages: Optional[int] = 3) -> LayoutSignals:
    """
    Heuristic-only. Enough for interim triage classification.
    """
    col_counts: List[int] = []
    table_scores: List[float] = []
    figure_scores: List[float] = []

    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[: max_pages or len(pdf.pages)]
        for p in pages:
            words = []
            try:
                words = p.extract_words() or []
            except Exception:
                words = []

            x_starts = [float(w.get("x0", 0.0)) for w in words if "x0" in w]
            col_counts.append(_cluster_columns(x_starts))

            # “table-ish”: lots of straight lines/rects OR very regular word rows
            lines = getattr(p, "lines", None) or []
            rects = getattr(p, "rects", None) or []
            grid_primitives = len(lines) + len(rects)

            # normalize by page area-ish
            page_area = float((p.width or 1) * (p.height or 1))
            primitive_density = min(1.0, (grid_primitives / 200.0))

            # “figure-ish”: images count / area
            images = getattr(p, "images", None) or []
            img_area = 0.0
            for img in images:
                w = abs((img.get("x1", 0) - img.get("x0", 0)) or 0)
                h = abs((img.get("y1", 0) - img.get("y0", 0)) or 0)
                img_area += float(w * h)

            img_ratio = min(1.0, img_area / page_area if page_area else 0.0)

            table_scores.append(primitive_density)
            figure_scores.append(img_ratio)

    approx_cols = int(round(sum(col_counts) / max(1, len(col_counts)))) if col_counts else 1
    return LayoutSignals(
        approx_column_count=max(1, approx_cols),
        tableish_score=float(sum(table_scores) / max(1, len(table_scores))),
        figureish_score=float(sum(figure_scores) / max(1, len(figure_scores))),
    )