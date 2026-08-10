"""Safety vs. Simulatability scatter with Pareto frontier.

One command, one figure: runs whatever evals are missing, then plots.

Scatter with one point per manifest condition:
    x = HarmBench safety   (1 - ASR, higher = safer)
    y = Simulatability     (NSG, higher = more simulatable)

A gray dashed Pareto frontier is drawn through the Pareto-optimal points, the
y=0 line splits the plot into a green (more-simulatable) / red (less-simulatable)
background, and each point carries a direct label.

The condition list, checkpoints and results layout come from the shared figure
manifest (configs/figures/*.yaml) — see evals/manifest.py. For each condition
this script ensures

    <results_dir>/harmbench_metrics.json        (keys "safety", "asr", "n")
    <results_dir>/ambiguous/nsg_metrics.json    (keys "nsg_strict", "nsg_relaxed",
                                                 "n_strict", "n_relaxed", ...)

exist (running evals/harmbench_eval.py / evals/ambiguous_consistency_eval.py
only when they don't), then plots
x = harmbench_metrics["safety"]; y = nsg_metrics[NSG_KEY] (default "nsg_strict").

Ported from sample_consistency/exps/moral/evals/plot_safety_simulatability.py.
The original recomputed NSG from raw per-prompt prediction triples; here the
Self-CTRL eval writes the headline metrics to JSON directly.

Two variants (--variant):
    none  (default) — point estimates, no error bars.
    ci              — 95% percentile-bootstrap CIs on both axes. NSG (y) is
                      bootstrapped from its point value + n (binary per-prompt
                      correctness is summarized only by the proportion, so we
                      resample the implied 0/1 outcomes); HarmBench safety (x)
                      likewise from safety + n. Requires "n" / "n_strict" (or
                      "n_relaxed") in the metric files; runs without them fall
                      back to point estimates for that axis.

Usage:
    python -m evals.plot_safety_sim                  # compute what's missing, plot
    python -m evals.plot_safety_sim --variant ci     # bootstrap CIs on both axes
    python -m evals.plot_safety_sim --plot-only      # never run an eval
    python -m evals.plot_safety_sim --only "Base"    # just this condition's evals
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evals.manifest import figure_argparser, prepare_figure
from evals.plot_style import PALETTE, apply_style


DEFAULT_OUT_NAME = "safety_vs_simulatability_scatter.pdf"

# Evals this figure needs, by name in the evals/manifest.py registry. They are
# computed only if missing, and shared with any other figure that lists them.
REQUIRES = ("harmbench", "nsg")

# Which NSG variant to plot on the y-axis. Switch to "nsg_relaxed" to use the
# relaxed NSG (partial_refusal counted as compliance).
NSG_KEY = "nsg_strict"
# Matching eval-set size key in nsg_metrics.json for the CI variant.
NSG_N_KEY = "n_strict" if NSG_KEY == "nsg_strict" else "n_relaxed"

# The three λ "update" methods get larger markers + dark labels; the baselines
# render smaller with light-gray labels (matches the source deck).
LAMBDA_ROLES = {"expl", "mixed", "beh"}

# Per-role colors, matching the bar deck's families: gray base, blue
# explanation, purple mixed, orange behavior. Baselines keep a gray FILL but get
# a colored OUTLINE tying them to the method they ablate — so this scatter uses
# its own fill table rather than plot_style.role_style.
COLORS: dict[str, str] = {
    "base":          PALETTE["gray"],
    "expl":          PALETTE["blue"],
    "mixed":         PALETTE["purple"],
    "beh":           PALETTE["orange"],
    "expl_baseline": PALETTE["gray"],
    "beh_baseline":  PALETTE["gray"],
}
EDGE_COLORS: dict[str, str] = {
    "expl_baseline": "#2f5470",   # dark blue outline -> ablates Expl.
    "beh_baseline":  "#a85f33",   # dark orange outline -> ablates Beh.
}

# Direct-label offsets (dx, dy) in data coords from each marker, plus alignment,
# keyed by role. Hand-tuned against the default axis limits so labels hug their
# markers without overlapping each other; re-tune if the data/limits move
# materially. Labels come from the manifest, so newlines there wrap the text.
_LABEL_OFFSETS: dict[str, tuple] = {
    "base":          (-0.004, -0.030, "right", "top"),
    "expl":          (-0.006, -0.030, "right", "top"),
    "mixed":         (-0.006,  0.030, "right", "bottom"),
    "beh":           (-0.006, -0.030, "right", "top"),
    "expl_baseline": (-0.006,  0.030, "right", "bottom"),
    "beh_baseline":  (-0.008,  0.000, "right", "center"),
}

# --- CI bootstrap config ----------------------------------------------------
_NBOOT = 10000
_BOOTSTRAP_SEED = 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_metrics(run_dir, filename):
    with open(Path(run_dir) / filename) as f:
        return json.load(f)


def _prop_ci(value, n):
    """95% percentile-bootstrap CI half-widths (err_lo, err_hi) for a proportion
    `value` over n binary outcomes; None if n is missing/zero. Both axes here are
    proportions (safety = 1 - ASR over n behaviors; NSG-ish over n prompts), so
    resampling the implied k-of-n 0/1 vector makes the success count
    Binomial(n, value), which we draw directly. NSG is technically a normalized
    gain rather than a raw proportion, so its bar is an approximate CI on the
    underlying accuracy scale — adequate for the figure's visual error bars."""
    if not n:
        return None
    n = int(n)
    p = min(max(value, 0.0), 1.0)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    means = np.sort(rng.binomial(n, p, size=_NBOOT) / n)
    return value - means[int(0.025 * _NBOOT)], means[int(0.975 * _NBOOT)] - value


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _apply_font():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })


def _shared_axes(ax):
    # Green tint above NSG=0 (more-simulatable); red tint below. Huge spans,
    # clipped by ylim, so they fill to the axes edges.
    ax.axhspan(0, 10, color="#6BAE5B", alpha=0.15, zorder=0)
    ax.axhspan(-10, 0, color="#D62728", alpha=0.12, zorder=0)
    ax.axhline(0, color="#888", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("HarmBench safety  (1 - ASR)", fontsize=15)
    ax.set_ylabel("Simulatability  (NSG, higher = more simulatable)", fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, ls=":", color="#cccccc")


def _add_pareto(ax, points):
    """points: list of (label, x, y). Draw a gray dashed line through the
    Pareto-optimal points (maximize both x and y)."""
    sorted_pts = sorted(points, key=lambda p: (-p[1], -p[2]))
    frontier, best = [], -float("inf")
    for p in sorted_pts:
        if p[2] > best:
            frontier.append(p)
            best = p[2]
    fx = [p[1] for p in frontier]
    fy = [p[2] for p in frontier]
    ax.plot(fx, fy, color="#888888", lw=1.4, ls="--", zorder=1, label="Pareto frontier")


def _draw_point(ax, role, x, y, x_err=None, y_err=None):
    is_method = role in LAMBDA_ROLES
    color = COLORS[role]
    if x_err is not None or y_err is not None:
        ax.errorbar(
            x, y, xerr=x_err, yerr=y_err, fmt="none",
            ecolor=color, elinewidth=1.4, capsize=6, capthick=1.4,
            alpha=0.9, zorder=2,
        )
    edge = EDGE_COLORS.get(role, "black")
    edge_lw = 1.8 if role in EDGE_COLORS else 0.8
    ax.scatter(
        x, y, s=170 if is_method else 110,
        color=color, edgecolor=edge, lw=edge_lw, zorder=3,
    )


def _label_point(ax, role, label, x, y):
    dx, dy, ha, va = _LABEL_OFFSETS[role]
    ax.text(
        x + dx, y + dy, label, ha=ha, va=va, fontsize=11,
        color="#1a1a1a" if role in LAMBDA_ROLES else "#7c7c7c",
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def build_figure(conditions, variant: str, out_path: Path):
    """variant: "none" (point estimates) or "ci" (bootstrap CI on both axes)."""
    apply_style()
    _apply_font()

    fig, ax = plt.subplots(figsize=(8, 6))
    _shared_axes(ax)

    drawn = []  # (label, role, x, y, x_err_or_None, y_err_or_None)
    for cond in conditions:
        hb = _load_metrics(cond.results_dir, "harmbench_metrics.json")
        with open(cond.nsg_metrics_path) as f:
            nsg = json.load(f)
        x = hb["safety"]
        y = nsg[NSG_KEY]
        x_err = y_err = None
        if variant == "ci":
            xc = _prop_ci(x, hb.get("n"))
            yc = _prop_ci(y, nsg.get(NSG_N_KEY))
            if xc is not None:
                x_err = [[xc[0]], [xc[1]]]
            if yc is not None:
                y_err = [[yc[0]], [yc[1]]]
        drawn.append((cond.label, cond.role, x, y, x_err, y_err))

    _add_pareto(ax, [(L, x, y) for L, _, x, y, *_ in drawn])
    for label, role, x, y, x_err, y_err in drawn:
        _draw_point(ax, role, x, y, x_err=x_err, y_err=y_err)
        _label_point(ax, role, label, x, y)

    # Pareto frontier shown as a single-entry legend in the open upper-right.
    handles, labels = ax.get_legend_handles_labels()
    for h, l in zip(handles, labels):
        if l == "Pareto frontier":
            ax.legend([h], [l], loc="upper right", frameon=False, fontsize=11)
            break

    ax.set_title("Safety vs. simulatability", fontsize=16)
    ax.set_xlim(0.78, 1.04)
    ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_ylim(-0.7 if variant == "ci" else -0.45, 0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = out_path if variant == "none" else out_path.with_name(
        out_path.stem + "_ci" + out_path.suffix
    )
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    flat = [(L.replace(chr(10), " "), round(x, 4), round(y, 4)) for L, _, x, y, *_ in drawn]
    print(f"  points (label, safety, {NSG_KEY}): {flat}")


def main():
    p = figure_argparser(__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--variant", choices=["none", "ci"], default="none",
        help='"none" = point estimates (default); "ci" = bootstrap CIs on both axes.',
    )
    args = p.parse_args()

    prepared = prepare_figure(args, REQUIRES)
    if prepared is None:
        return
    mf, conditions = prepared

    out_path = Path(args.out) if args.out else mf.figures_dir / DEFAULT_OUT_NAME
    build_figure(conditions, args.variant, out_path)


if __name__ == "__main__":
    main()