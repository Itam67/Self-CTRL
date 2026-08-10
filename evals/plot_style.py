"""Shared matplotlib style for the consistency-curves figure.

Usage:
    from evals.plot_style import PALETTE, apply_style
    apply_style()

The PALETTE dict gives named colors so the figure's Validation / Held-out
category / Held-out principle lines share a consistent palette.

Ported from sample_consistency/exps/moral/evals/plot_style.py — only the parts
the consistency-curves figure needs (PALETTE, apply_style) are kept here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


PALETTE = {
    # Semantic roles (preferred — use these in new code).
    "supervised": "#6c757d",   # gray — supervised baseline (passive, no training signal)
    "cons":       "#4878A6",   # blue — consistency-trained set (train + val)
    "held_cat":   "#DD8452",   # orange — held-out at category level
    "held_pri":   "#6BAE5B",   # green — held-out at principle level; also the Holdout color
    "holdout":    "#6BAE5B",   # alias of held_pri

    # Backward-compat raw color keys.
    "blue":       "#4878A6",
    "orange":     "#DD8452",
    "green":      "#6BAE5B",
    "purple":     "#8C72A4",
    "gray":       "#6c757d",
    "subheader":  "#4d4d4d",   # dark gray for subheaders / annotations
}


# Per-role styling, keyed by the `role` field of a manifest condition. Every
# figure colors its bars/points through this table, so a condition keeps the same
# family color across the whole deck and relabelling never recolors anything.
# Each entry is (facecolor, edgecolor).
ROLE_STYLE = {
    "base":           (PALETTE["gray"],   "#3f4347"),
    "expl_baseline":  (PALETTE["blue"],   "#2f5470"),
    "beh_baseline":   (PALETTE["orange"], "#a85f33"),
    "expl":           (PALETTE["blue"],   "#2f5470"),
    "mixed":          (PALETTE["purple"], "#5f4f74"),
    "beh":            (PALETTE["orange"], "#a85f33"),
}


def role_style(role: str):
    """(facecolor, edgecolor) for a manifest role."""
    try:
        return ROLE_STYLE[role]
    except KeyError:
        raise KeyError(
            f"no style for role {role!r}; known roles: {', '.join(ROLE_STYLE)}"
        ) from None


def apply_style():
    """Set rcParams used across all paper figures."""
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "legend.fontsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "axes.axisbelow": True,
    })
