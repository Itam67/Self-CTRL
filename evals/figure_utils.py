from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Bootstrap settings for every error bar in the deck. Fixed seed so a figure is
# byte-identical across reruns.
NBOOT = 10000
BOOTSTRAP_SEED = 0


def apply_serif_font() -> None:

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )


def prop_ci(
    value: float, n: int, *, snap: bool = True
) -> Optional[Tuple[float, float]]:
    """95% percentile-bootstrap CI half-widths (err_lo, err_hi) for `value`.

    The bar is built by resampling the implied k-of-n binary outcomes.

    snap=True  (default) sets p = round(value * n) / n.
    snap=False sets p = clamp(value, 0, 1) instead.
    """
    if not n:
        return None
    n = int(n)
    p = round(value * n) / n if snap else min(max(value, 0.0), 1.0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.sort(rng.binomial(n, p, size=NBOOT) / n)
    return value - means[int(0.025 * NBOOT)], means[int(0.975 * NBOOT)] - value


def load_metrics(run_dir, filename: str) -> dict:
    """Read one metric JSON out of a condition's results dir."""
    with open(Path(run_dir) / filename) as f:
        return json.load(f)
