from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Dict


@contextmanager
def timer() -> Iterator[Dict[str, float]]:
    t0 = time.perf_counter()
    info = {"seconds": 0.0}
    try:
        yield info
    finally:
        info["seconds"] = time.perf_counter() - t0