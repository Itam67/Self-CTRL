"""Non-toxic Compliance + Capabilities two-panel bar figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from evals.manifest import figure_argparser, prepare_figure
from evals.figure_utils import apply_serif_font, load_metrics, prop_ci
from evals.plot_style import apply_style, role_style

DEFAULT_OUT_NAME = "overrefusal_capabilities_bars.pdf"

# Evals this figure needs, by name in the evals/manifest.py registry. They are
# computed only if missing, and shared with any other figure that lists them.
REQUIRES = ("wildchat", "mmlu")

BAR_LW = 1.4


# --- 95% CI error bars ------------------------------------------------------
# Both bars are proportions of binary per-item outcomes (WildChat non-refuse,
# MMLU correct), so the 95% CI is a nonparametric percentile bootstrap over
# those 0/1 outcomes. Resampling a k-of-n binary vector makes the success count
# Binomial(n, k/n), so we draw that count directly from the plotted value + its
# eval-set size (n read live from the metrics JSON).
# Set SHOW_CI = False to render the original error-bar-free figure.
SHOW_CI = True


def _draw_panel(
    ax, colors, labels, vals, ns, *, title, xlabel, ylabel, show_yticklabels
):
    xs = np.arange(len(labels))
    for x, (fill, edge), v, n in zip(xs, colors, vals, ns):
        ax.bar(x, v, 0.72, facecolor=fill, edgecolor=edge, linewidth=BAR_LW, zorder=2)
        err = prop_ci(v, n) if SHOW_CI else None
        if err is None:
            text_y = v + 0.016
        else:
            ax.errorbar(
                x,
                v,
                yerr=[[err[0]], [err[1]]],
                fmt="none",
                ecolor="#2a2a2a",
                elinewidth=1.3,
                capsize=4,
                capthick=1.2,
                zorder=4,
            )
            text_y = v + err[1] + 0.016
        ax.text(
            x,
            text_y,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=17,
            color="#2a2a2a",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, 1.12)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=17)
    ax.tick_params(axis="x", pad=6)
    ax.set_xlabel(xlabel, fontsize=20, labelpad=10)
    ax.set_title(title, fontsize=23, pad=12)
    if show_yticklabels:
        ax.set_ylabel(ylabel, fontsize=20, labelpad=14)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False)
        ax.spines["left"].set_visible(False)


def main():
    args = figure_argparser(__doc__.split("\n\n", 1)[0]).parse_args()
    prepared = prepare_figure(args, REQUIRES)
    if prepared is None:
        return
    mf, conditions = prepared

    apply_style()
    apply_serif_font()

    labels = [c.label for c in conditions]
    run_dirs = [c.results_dir for c in conditions]
    colors = [role_style(c.role) for c in conditions]

    wc = [load_metrics(d, "wildchat_metrics.json") for d in run_dirs]
    mm = [load_metrics(d, "mmlu_metrics.json") for d in run_dirs]
    compliance = [1 - m["refusal_rate"] for m in wc]
    capabilities = [m["accuracy"] for m in mm]
    comp_n = [m.get("n") for m in wc]
    cap_n = [m.get("n") for m in mm]

    # Widen the canvas with more conditions so bars/fonts stay constant size.
    fig = plt.figure(figsize=(2.4 * len(labels) + 3, 5.8))
    gs = gridspec.GridSpec(1, 2, wspace=0.10)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    _draw_panel(
        ax0,
        colors,
        labels,
        compliance,
        comp_n,
        title="Non-toxic Compliance",
        xlabel="WildChat non-toxic",
        ylabel="Benchmark value (higher = better)",
        show_yticklabels=True,
    )
    _draw_panel(
        ax1,
        colors,
        labels,
        capabilities,
        cap_n,
        title="Capabilities",
        xlabel="MMLU",
        ylabel="",
        show_yticklabels=False,
    )

    out_path = Path(args.out) if args.out else mf.figures_dir / DEFAULT_OUT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    flat = [l.replace("\n", " ") for l in labels]
    print(f"Saved: {out_path}")
    print(f"  compliance:   {dict(zip(flat, [round(v, 4) for v in compliance]))}")
    print(f"  capabilities: {dict(zip(flat, [round(v, 4) for v in capabilities]))}")


if __name__ == "__main__":
    main()
