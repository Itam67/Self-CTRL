"""Counterfactual Consistency grouped-bar figure."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from evals.manifest import figure_argparser, prepare_figure
from evals.figure_utils import apply_serif_font, prop_ci
from evals.plot_style import apply_style, role_style

DEFAULT_OUT_NAME = "cf_consistency_bars.pdf"

# Evals this figure needs, by name in the evals/manifest.py registry. They are
# computed only if missing, and shared with any other figure that lists them.
REQUIRES = ("cf",)

# Both bars in a condition share the role fill/edge; the Compliance bar is
# differentiated by the white hatch overlay.
BAR_LW = 1.4


def _load_cf_metrics(run_dir):
    """Load the per-category cf metrics block from a condition directory.

    The eval filename carries a model slug + N (cf_consistency_<slug>_n<N>_percat
    .json), so glob the stable suffix and take the single match (newest by mtime
    if several variants are present)."""
    matches = glob.glob(str(Path(run_dir) / "cf_consistency_*_percat.json"))
    matches = [m for m in matches if not m.endswith("_cache.json")]
    if not matches:
        raise FileNotFoundError(f"no cf_consistency_*_percat.json under {run_dir}")
    matches.sort(key=lambda m: Path(m).stat().st_mtime)
    with open(matches[-1]) as f:
        return json.load(f)["metrics"]


# --- 95% CI error bars ------------------------------------------------------
# Each bar is a per-side accuracy over the pooled boundary prompts on that side
# (~80 = 8 categories x 10/side), treated as the independent sampling unit. The
# 95% CI is a nonparametric percentile bootstrap over those binary outcomes.
# Resampling a k-of-n binary vector makes the success count Binomial(n, k/n), so
# we draw that count from each bar's value and its on-disk per-side n
# (n_refuse_prompts / n_comply_prompts). Bootstrap means are bounded in [0, 1],
# so the comply side never exceeds 1.0.
# Set SHOW_CI = False to render the original error-bar-free figure.
SHOW_CI = True


def main():
    args = figure_argparser(__doc__.split("\n\n", 1)[0]).parse_args()
    prepared = prepare_figure(args, REQUIRES)
    if prepared is None:
        return
    mf, conditions = prepared

    apply_style()
    apply_serif_font()
    # Only this figure uses hatched bars.
    plt.rcParams.update({"hatch.color": "white", "hatch.linewidth": 1.6})

    labels = [c.label for c in conditions]
    run_dirs = [c.results_dir for c in conditions]
    colors = [role_style(c.role) for c in conditions]

    mets = [_load_cf_metrics(d) for d in run_dirs]
    refuse = [m["refuse_accuracy"] for m in mets]
    comply = [m["comply_accuracy_relaxed"] for m in mets]
    refuse_n = [m.get("n_refuse_prompts") for m in mets]
    comply_n = [m.get("n_comply_prompts") for m in mets]
    fills = [c[0] for c in colors]
    edges = [c[1] for c in colors]

    n = len(labels)
    xs = np.arange(n)
    w = 0.36
    off = 0.21

    def _ci_label_y(ax, x, v, nside):
        """Draw a per-side CI whisker (if any) and return the value-label y."""
        err = prop_ci(v, nside) if SHOW_CI else None
        if err is None:
            return v + 0.012
        ax.errorbar(
            x,
            v,
            yerr=[[err[0]], [err[1]]],
            fmt="none",
            ecolor="#2a2a2a",
            elinewidth=1.4,
            capsize=5,
            capthick=1.4,
            zorder=4,
        )
        return v + err[1] + 0.012

    fig, ax = plt.subplots(figsize=(14, 6))
    for x, fill, edge, vr, vc, nr, nc in zip(
        xs, fills, edges, refuse, comply, refuse_n, comply_n
    ):
        ax.bar(
            x - off, vr, w, facecolor=fill, edgecolor=edge, linewidth=BAR_LW, zorder=2
        )
        ax.bar(
            x + off,
            vc,
            w,
            facecolor=fill,
            edgecolor="white",
            hatch="////",
            linewidth=0,
            zorder=2,
        )
        ax.bar(
            x + off, vc, w, facecolor="none", edgecolor=edge, linewidth=BAR_LW, zorder=3
        )
        yr = _ci_label_y(ax, x - off, vr, nr)
        yc = _ci_label_y(ax, x + off, vc, nc)
        ax.text(
            x - off,
            yr,
            f"{vr:.2f}",
            ha="center",
            va="bottom",
            fontsize=16,
            color="#2a2a2a",
        )
        ax.text(
            x + off,
            yc,
            f"{vc:.2f}",
            ha="center",
            va="bottom",
            fontsize=16,
            color="#2a2a2a",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=17)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel("Counterfactual consistency", fontsize=20, labelpad=12)
    ax.set_title("Counterfactual Consistency", fontsize=24, pad=14)
    ax.grid(False)
    ax.tick_params(axis="y", labelsize=17)
    ax.tick_params(axis="x", pad=6)

    legend = [
        Patch(facecolor="#9e9e9e", edgecolor="#555555", label="Refusal Acc"),
        Patch(
            facecolor="#9e9e9e", edgecolor="white", hatch="////", label="Compliance Acc"
        ),
    ]
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        fontsize=17,
        frameon=False,
    )

    out_path = Path(args.out) if args.out else mf.figures_dir / DEFAULT_OUT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    flat = [l.replace("\n", " ") for l in labels]
    print(f"Saved: {out_path}")
    print(f"  refuse_accuracy: {dict(zip(flat, [round(v, 4) for v in refuse]))}")
    print(f"  comply_relaxed:  {dict(zip(flat, [round(v, 4) for v in comply]))}")


if __name__ == "__main__":
    main()
