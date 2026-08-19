"""Safety vs. Simulatability scatter, with the three λ conditions connected."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evals.manifest import REPO_ROOT, figure_argparser, prepare_figure
from evals.figure_utils import (
    BOOTSTRAP_SEED,
    NBOOT,
    apply_serif_font,
    load_metrics,
    prop_ci,
)
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


_RELAXED_BINARY = {
    "full_compliance": "comply",
    "partial_refusal": "comply",  # relaxed mapping, matching nsg_relaxed
    "full_refusal": "refuse",
}


def _nsg_bootstrap_ci(cond, mf, point):
    """95% CI for NSG-relaxed via a paired percentile bootstrap over prompts.

    NSG = (a_we - a_zs) / (1 - a_zs) is a ratio of two accuracies measured on
    the SAME prompts — not a k-of-n proportion, so prop_ci's binomial model is
    the wrong family (and clamps negative NSG, which this figure expects).
    Resampling prompts and recomputing the whole statistic per resample keeps
    the two arms' dependence intact. Returns (lo_err, hi_err) or None when the
    per-item records aren't on disk."""
    ambig = cond.nsg_metrics_path.parent
    gemini_model = mf.eval_cfg.get("nsg", {}).get("gemini_model", "gemini-2.5-flash")
    zs_path = REPO_ROOT / "evals/moral/ambiguous_cache/zero_shot_predictions.json"
    try:
        gt = json.load(open(ambig / "gt_classifications.json"))["classifications"]
        pe = json.load(open(ambig / "predictions_with_expl.json"))["predictions"]
        zs = json.load(open(zs_path))[gemini_model]["predictions"]
    except (FileNotFoundError, KeyError):
        return None

    we_ok, zs_ok = [], []
    for cat, labels in gt.items():
        for i, lab in enumerate(labels):
            b = _RELAXED_BINARY.get(lab)
            if b is None:  # judge-failed / unparseable gt — excluded from both arms
                continue
            try:
                we_ok.append(pe[cat][i] == b)
                zs_ok.append(zs[cat][i] == b)
            except (KeyError, IndexError):
                continue
    n = len(we_ok)
    if n < 2:
        return None
    we_ok = np.asarray(we_ok, dtype=float)
    zs_ok = np.asarray(zs_ok, dtype=float)

    # Sanity: the point estimate must reproduce from the records we resample.
    a_we, a_zs = we_ok.mean(), zs_ok.mean()
    check = (a_we - a_zs) / max(1.0 - a_zs, 1e-9)
    if abs(check - point) > 0.02:
        print(
            f"  WARNING: per-item NSG {check:.4f} != metrics {point:.4f} for "
            f"{ambig} — skipping its CI"
        )
        return None

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(NBOOT, n))
    bw = we_ok[idx].mean(axis=1)
    bz = zs_ok[idx].mean(axis=1)
    boot = (bw - bz) / np.clip(1.0 - bz, 1e-9, None)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return max(0.0, point - float(lo)), max(0.0, float(hi) - point)


def _repel_labels(fig, ax, texts, max_passes=8):
    """Nudge direct labels apart vertically until their bounding boxes stop
    overlapping (data positions of the markers are untouched)."""
    for _ in range(max_passes):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [t.get_window_extent(renderer=renderer) for t in texts]
        moved = False
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                bi, bj = boxes[i], boxes[j]
                if bi.overlaps(bj):
                    # Push the lower one further down by the overlap + padding.
                    lower, upper = (i, j) if bi.y0 <= bj.y0 else (j, i)
                    overlap = boxes[upper].y0 - boxes[lower].y1  # negative
                    shift_px = overlap - 3
                    t = texts[lower]
                    x, y = t.get_position()
                    inv = ax.transData.inverted()
                    dy = (
                        inv.transform((0, 0))[1] - inv.transform((0, -shift_px))[1]
                    )
                    t.set_position((x, y - abs(dy)))
                    moved = True
        if not moved:
            return


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
    return ax.text(
        x + dx,
        y + dy,
        label,
        ha=ha,
        va=va,
        fontsize=11,
        color="#1a1a1a" if role in LAMBDA_ROLES else "#7c7c7c",
    )


def build_figure(conditions, variant: str, out_path: Path, mf=None):
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
            # x is a k-of-n proportion (1 - ASR over n behaviors): binomial
            # bootstrap is the right family there.
            xc = prop_ci(x, hb.get("n"), snap=False)
            if xc is not None:
                x_err = [[xc[0]], [xc[1]]]
            # y (NSG) is a paired ratio statistic: paired percentile bootstrap
            # over the per-prompt records (see _nsg_bootstrap_ci).
            yc = _nsg_bootstrap_ci(cond, mf, y)
            if yc is not None:
                y_err = [[yc[0]], [yc[1]]]
        drawn.append((cond.label, cond.role, x, y, x_err, y_err))

    xy = [(role, x, y) for _, role, x, y, *_ in drawn]
    _add_lambda_line(ax, xy)
    _add_pareto_frontier(ax, xy)
    texts = []
    for label, role, x, y, x_err, y_err in drawn:
        _draw_point(ax, role, x, y, x_err=x_err, y_err=y_err)
        texts.append(_label_point(ax, role, label, x, y))

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
    # Extra headroom at the top: the legend lives in the upper right, and the
    # high-λ conditions (safety ≈ 1) put points and labels under that corner —
    # the reserved band keeps them from colliding with it.
    pad_top = max(0.22, 0.25 * (y_hi - y_lo))
    ax.set_ylim(y_lo - pad, y_hi + pad_top)

    # With final limits in place, separate any colliding direct labels.
    _repel_labels(fig, ax, texts)

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
    build_figure(conditions, args.variant, out_path, mf=mf)


if __name__ == "__main__":
    main()
