"""Safety vs. Simulatability scatter, with the three λ conditions connected."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evals.manifest import figure_argparser, prepare_figure
from evals.figure_utils import apply_serif_font, load_metrics, prop_ci
from evals.plot_style import apply_style, role_style

DEFAULT_OUT_NAME = "safety_vs_simulatability_scatter.pdf"

# Evals this figure needs, by name in the evals/manifest.py registry. They are
# computed only if missing, and shared with any other figure that lists them.
REQUIRES = ("harmbench", "nsg")

# Which NSG variant to plot on the y-axis. Switch to "nsg_relaxed" to use the
NSG_KEY = "nsg_relaxed"
# Matching eval-set size key in nsg_metrics.json for the CI variant.
NSG_N_KEY = "n_strict" if NSG_KEY == "nsg_strict" else "n_relaxed"

# The three λ "update" methods get larger markers + dark labels
LAMBDA_ROLES = {"expl", "mixed", "beh"}

# Order the connecting line walks the λ sweep: λ=0 -> 0.5 -> 1.0. Roles absent
# from the manifest are skipped, so a partial deck still draws what it has.
LAMBDA_LINE_ORDER = ("expl", "mixed", "beh")
LAMBDA_LINE_LABEL = "λ sweep (0 → 0.5 → 1.0)"
PARETO_LINE_LABEL = "Pareto frontier"

# Introspective baselines: drawn as a gray marker under the role-colored outline
# of the method they ablate (dark blue -> ablates Expl., dark orange -> Beh.).
BASELINE_ROLES = {"expl_baseline", "beh_baseline"}


def _role_colors(role: str) -> tuple[str, str, float]:
    """(facecolor, edgecolor, edge linewidth) for a role in this figure.

    Colors come from the shared plot_style palette, whose moral roles are the
    paper's BAR_CONDITIONS verbatim. Matching the paper's scatter: the ablation
    baselines keep their own gray-ramp fill and carry a thicker colored outline
    tying them to the method they ablate; every other role takes a plain black
    outline.
    """
    face, edge = role_style(role)
    if role in BASELINE_ROLES:
        return face, edge, 1.8
    return face, "black", 0.8


# Direct-label offsets (dx, dy) in data coords from each marker
_LABEL_OFFSETS: dict[str, tuple] = {
    "base": (-0.004, -0.030, "right", "top"),
    "expl": (-0.006, -0.030, "right", "top"),
    "mixed": (-0.006, 0.030, "right", "bottom"),
    "beh": (-0.006, -0.030, "right", "top"),
    "expl_baseline": (-0.006, 0.030, "right", "bottom"),
    "beh_baseline": (-0.008, 0.000, "right", "center"),
}


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


def _add_lambda_line(ax, points):
    """points: list of (role, x, y). Connect the three λ update methods in λ
    order (0 -> 0.5 -> 1.0).

    This traces the sweep the figure is about, so it is drawn through all three
    regardless of Pareto dominance. The previous version drew the Pareto
    frontier instead, which silently dropped any dominated λ — and a dominated
    behavior-training point is precisely what the reader needs to see."""
    by_role = {role: (x, y) for role, x, y in points}
    line = [by_role[r] for r in LAMBDA_LINE_ORDER if r in by_role]
    if len(line) < 2:
        return
    ax.plot(
        [p[0] for p in line],
        [p[1] for p in line],
        color="#888888",
        lw=1.4,
        ls="--",
        zorder=1,
        label=LAMBDA_LINE_LABEL,
    )


def _add_pareto_frontier(ax, points):
    """points: list of (role, x, y), all conditions. Draw the frontier the
    paper's scatter drew: walk the points from safest leftward, keeping each
    point that improves on the best simulatability seen so far (sort by -x,
    then -y; keep running-max y). Drawn alongside the λ sweep — the sweep shows
    the path through λ, the frontier shows who is undominated."""
    pts = sorted(((x, y) for _, x, y in points), key=lambda p: (-p[0], -p[1]))
    frontier, best_y = [], None
    for x, y in pts:
        if best_y is None or y > best_y:
            frontier.append((x, y))
            best_y = y
    if len(frontier) < 2:
        return
    ax.plot(
        [p[0] for p in frontier],
        [p[1] for p in frontier],
        color="#555555",
        lw=1.6,
        ls=(0, (5, 2)),
        zorder=1,
        label=PARETO_LINE_LABEL,
    )


def _draw_point(ax, role, x, y, x_err=None, y_err=None):
    is_method = role in LAMBDA_ROLES
    color, edge, edge_lw = _role_colors(role)
    if x_err is not None or y_err is not None:
        ax.errorbar(
            x,
            y,
            xerr=x_err,
            yerr=y_err,
            fmt="none",
            ecolor=color,
            elinewidth=1.4,
            capsize=6,
            capthick=1.4,
            alpha=0.9,
            zorder=2,
        )
    ax.scatter(
        x,
        y,
        s=170 if is_method else 110,
        color=color,
        edgecolor=edge,
        lw=edge_lw,
        zorder=3,
    )


def _label_point(ax, role, label, x, y):
    dx, dy, ha, va = _LABEL_OFFSETS[role]
    ax.text(
        x + dx,
        y + dy,
        label,
        ha=ha,
        va=va,
        fontsize=11,
        color="#1a1a1a" if role in LAMBDA_ROLES else "#7c7c7c",
    )


def build_figure(conditions, variant: str, out_path: Path):
    """variant: "none" (point estimates) or "ci" (bootstrap CI on both axes)."""
    apply_style()
    apply_serif_font()

    fig, ax = plt.subplots(figsize=(8, 6))
    _shared_axes(ax)

    drawn = []  # (label, role, x, y, x_err_or_None, y_err_or_None)
    for cond in conditions:
        hb = load_metrics(cond.results_dir, "harmbench_metrics.json")
        with open(cond.nsg_metrics_path) as f:
            nsg = json.load(f)
        x = hb["safety"]
        y = nsg[NSG_KEY]
        x_err = y_err = None
        if variant == "ci":
            xc = prop_ci(x, hb.get("n"), snap=False)
            yc = prop_ci(y, nsg.get(NSG_N_KEY), snap=False)
            if xc is not None:
                x_err = [[xc[0]], [xc[1]]]
            if yc is not None:
                y_err = [[yc[0]], [yc[1]]]
        drawn.append((cond.label, cond.role, x, y, x_err, y_err))

    xy = [(role, x, y) for _, role, x, y, *_ in drawn]
    _add_lambda_line(ax, xy)
    _add_pareto_frontier(ax, xy)
    for label, role, x, y, x_err, y_err in drawn:
        _draw_point(ax, role, x, y, x_err=x_err, y_err=y_err)
        _label_point(ax, role, label, x, y)

    # Both connecting lines in one legend in the open upper-right.
    handles, labels = ax.get_legend_handles_labels()
    line_entries = [
        (h, l) for h, l in zip(handles, labels)
        if l in (LAMBDA_LINE_LABEL, PARETO_LINE_LABEL)
    ]
    if line_entries:
        ax.legend(
            [h for h, _ in line_entries],
            [l for _, l in line_entries],
            loc="upper right",
            frameon=False,
            fontsize=11,
        )

    ax.set_title("Safety vs. simulatability", fontsize=16)
    ax.set_xlim(0.78, 1.04)
    ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
    # y-limits follow the data (error-bar extents included), padded so the
    # direct labels beside the extreme points stay inside the axes.
    y_lo = min(
        y - (y_err[0][0] if y_err else 0.0) for _, _, _, y, _, y_err in drawn
    )
    y_hi = max(
        y + (y_err[1][0] if y_err else 0.0) for _, _, _, y, _, y_err in drawn
    )
    pad = max(0.08, 0.10 * (y_hi - y_lo))
    ax.set_ylim(y_lo - pad, y_hi + pad)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = (
        out_path
        if variant == "none"
        else out_path.with_name(out_path.stem + "_ci" + out_path.suffix)
    )
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    flat = [
        (L.replace(chr(10), " "), round(x, 4), round(y, 4)) for L, _, x, y, *_ in drawn
    ]
    print(f"  points (label, safety, {NSG_KEY}): {flat}")


def main():
    p = figure_argparser(__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--variant",
        choices=["none", "ci"],
        default="none",
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
