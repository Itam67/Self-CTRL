"""Consistency (jury eval score) over training: one panel per training run.

Builds a 1xN horizontal layout from the manifest's `panels`, sized so each panel
is readable at paper-figure scale. The panels share a y-axis and a single
legend. Plots Avg consistency (phi) vs Examples seen, with Validation /
Held-out category / Held-out principle lines.

PURE PLOTTING — unlike the other figure scripts this one runs no evals, because
the curves come from the eval snapshots the trainer already wrote during
training:
    <run_dir>/eval_snapshots/eval_*.json          (validation)
    <run_dir>/eval_snapshots/holdout_eval_*.json  (held-out)

Panels (and their run dirs) come from the shared figure manifest
configs/figures/*.yaml — see evals/manifest.py. A panel may list several run
dirs; they are merged step-by-step, earlier dirs winning on overlapping steps,
so a continuation run extends the curve past where the original stopped.

Usage:
    python -m evals.plot_consistency_curves
    python -m evals.plot_consistency_curves --manifest configs/figures/<family>.yaml
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evals.manifest import figure_argparser, prepare_figure
from evals.plot_style import PALETTE, apply_style
from evals.viz_utils import (
    load_all_snapshots,
    load_holdout_snapshots,
    _series_from_snapshots,
    _split_holdout_snapshots,
)


DEFAULT_OUT_NAME = "consistency_curves_combined.pdf"

# This figure runs no evals — the curves come from the trainer's own eval
# snapshots, so there is nothing to compute or share.
REQUIRES = ()


def _set_size(snaps, predicate=None):
    for snap in snaps.values():
        recs = snap.get("records", [])
        if predicate:
            recs = [r for r in recs if predicate(r)]
        if recs:
            return len(recs)
    return 0


def _merge_snaps(dirs, loader):
    """Merge a loader's step-keyed snapshots across dirs; earlier dirs win on
    overlapping steps (so a continuation's re-eval doesn't override the original)."""
    merged = {}
    for d in dirs:
        for step, snap in loader(Path(d)).items():
            merged.setdefault(step, snap)
    return merged


def _draw_panel(ax, run_dirs, xmax=None):
    """Render val + held-out curves for one run (or merged across several dirs).
    xmax clips every series so all panels can share one x-range.
    Returns the series list so the caller can build a single shared legend."""
    dirs = [run_dirs] if isinstance(run_dirs, (str, Path)) else list(run_dirs)
    val_snaps = _merge_snaps(dirs, load_all_snapshots)
    holdout_snaps = _merge_snaps(dirs, load_holdout_snapshots)
    if xmax is not None:
        val_snaps = {s: v for s, v in val_snaps.items() if s <= xmax}
        holdout_snaps = {s: v for s, v in holdout_snaps.items() if s <= xmax}

    series = []  # (xs, ys, color, label, filled)
    val_x, val_y = _series_from_snapshots(val_snaps, "avg_consistency")
    if val_x:
        n_val = _set_size(val_snaps)
        series.append((val_x, val_y, PALETTE["cons"], f"Validation  (n={n_val})", False))
    if holdout_snaps:
        broad, fine = _split_holdout_snapshots(holdout_snaps)
        bx, by = _series_from_snapshots(broad, "avg_consistency")
        fx, fy = _series_from_snapshots(fine, "avg_consistency")
        if bx:
            n_b = _set_size(holdout_snaps, lambda r: r.get("holdout_type") == "broad_category")
            series.append((bx, by, PALETTE["held_cat"], f"Held-out category  (n={n_b})", True))
        if fx:
            n_f = _set_size(holdout_snaps, lambda r: r.get("holdout_type") == "fine_principle")
            series.append((fx, fy, PALETTE["held_pri"], f"Held-out principle  (n={n_f})", True))

    for xs, ys, color, label, filled in series:
        ax.plot(
            xs, ys,
            marker="o", markersize=8,
            linewidth=2.4, color=color,
            markerfacecolor=color if filled else "white",
            markeredgecolor=color, markeredgewidth=1.6,
            label=label, zorder=2, alpha=0.92,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)   # 1.0 markers no longer clipped
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="both", labelsize=19)
    ax.grid(True, alpha=0.18, linewidth=0.7)
    return series


def main():
    args = figure_argparser(__doc__.split("\n\n", 1)[0]).parse_args()
    prepared = prepare_figure(args, REQUIRES)
    if prepared is None:
        return
    mf, _ = prepared

    panels = mf.panels
    if not panels:
        raise SystemExit(f"{mf.path} defines no `panels:` — nothing to plot.")

    missing = [p.title for p in panels
               if not any(Path(d, "eval_snapshots").exists() for d in p.run_dirs)]
    if missing:
        raise SystemExit(
            "No eval_snapshots/ found for panel(s): " + ", ".join(missing) +
            f"\nPoint `panels:` in {mf.path} at run dirs the trainer wrote."
        )

    apply_style()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
    })

    out_path = Path(args.out) if args.out else mf.figures_dir / DEFAULT_OUT_NAME

    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 5), sharey=True)
    axes = np.atleast_1d(axes)

    # Common x-range: cap every panel at the last checkpoint of the final panel's
    # run (the behavior/bw=1 run in the paper layout), so a merged/continued run
    # is cut to match.
    cap = max(load_all_snapshots(Path(panels[-1].run_dirs[-1])).keys())

    legend_series = None
    for ax, panel in zip(axes, panels):
        series = _draw_panel(ax, panel.run_dirs, xmax=cap)
        ax.set_title(panel.title, fontsize=26, pad=12)
        ax.set_xlabel("Examples seen", fontsize=23)
        ax.set_xlim(0, cap * 1.03)
        if legend_series is None:
            legend_series = series

    axes[0].set_ylabel(r"Avg consistency ($\mathrm{\phi}$)", fontsize=23)

    # Build proxy handles for a single shared legend.
    proxy_handles = []
    for _, _, color, label, filled in legend_series:
        line, = plt.plot(
            [], [],
            marker="o", markersize=7, linewidth=2.2, color=color,
            markerfacecolor=color if filled else "white",
            markeredgecolor=color, markeredgewidth=1.5,
            label=label,
        )
        proxy_handles.append(line)

    fig.legend(
        handles=proxy_handles,
        labels=[s[3] for s in legend_series],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=len(proxy_handles),
        frameon=False,
        fontsize=20,
        columnspacing=2.4,
        handletextpad=0.6,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()