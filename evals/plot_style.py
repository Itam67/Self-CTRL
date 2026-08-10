from __future__ import annotations

import matplotlib.pyplot as plt

PALETTE = {
    "blue": "#4878A6",
    "orange": "#DD8452",
    "green": "#6BAE5B",
    "purple": "#8C72A4",
    "gray": "#6c757d",
    # Eval-set semantics, used by the consistency-curves figure.
    "cons": "#4878A6",  # consistency-trained set (train + val)
    "held_cat": "#DD8452",  # held out at category level
    "held_pri": "#6BAE5B",  # held out at principle level
}


# Per-run styling
ROLE_STYLE = {
    # moral domain
    "base": (PALETTE["gray"], "#3f4347"),
    "expl_baseline": (PALETTE["blue"], "#2f5470"),
    "beh_baseline": (PALETTE["orange"], "#a85f33"),
    "expl": (PALETTE["blue"], "#2f5470"),
    "mixed": (PALETTE["purple"], "#5f4f74"),
    "beh": (PALETTE["orange"], "#a85f33"),
    # coins domain
    "sft_baseline": (PALETTE["gray"], "#3f4347"),
    "cons_trained": (PALETTE["blue"], "#2f5470"),
    "oracle": (PALETTE["green"], "#4a7a3f"),
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
    plt.rcParams.update(
        {
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
        }
    )
