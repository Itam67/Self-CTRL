"""Shared coins statistics."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


def mode_of_samples(samples: Sequence[float]) -> float:
    """Modal stated bias over K sampled programs (3dp); ties broken toward the mean."""
    s = np.asarray(samples, dtype=float)
    if s.size == 0:
        return float("nan")
    counts = Counter(np.round(s, 3).tolist())
    top = max(counts.values())
    candidates = [v for v, c in counts.items() if c == top]
    if len(candidates) == 1:
        return float(candidates[0])
    return float(min(candidates, key=lambda v: abs(v - float(s.mean()))))


def mean_of_samples(samples: Sequence[float]) -> float:
    """Mean stated bias over K sampled programs — the companion estimator to
    mode_of_samples, reported alongside it wherever R^2 is computed."""
    s = np.asarray(samples, dtype=float)
    if s.size == 0:
        return float("nan")
    return float(s.mean())
