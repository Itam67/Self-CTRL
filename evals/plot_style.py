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
#
# The moral entries are the paper's deck verbatim — `BAR_CONDITIONS` in the
# original's exps/moral/evals/plot_style.py, which is the single source of truth
# the paper's bar figures and its safety/simulatability scatter both read. The
# scheme is deliberate and is NOT the generic PALETTE above:
#   * the three λ runs take a light -> dark BLUE ramp, so λ reads off the fill;
#   * the two introspective baselines take a GRAY ramp, with an edge tying each
#     to the method it ablates (light blue -> Expl., dark navy -> Beh.);
#   * the untrained base is the lightest gray, edge included.
ROLE_STYLE = {
    # moral domain — (fill, edge) per the paper's BAR_CONDITIONS
    "base": ("#D4D4D4", "#B7B7B7"),
    "expl_baseline": ("#A8A8A8", "#94B3D9"),
    "beh_baseline": ("#7C7C7C", "#263C68"),
    "expl": ("#94B3D9", "#000000"),
    "mixed": ("#557AA4", "#000000"),
    "beh": ("#263C68", "#000000"),
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
