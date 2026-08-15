"""Coins calibration / self-consistency figure."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from evals.coins.stats import mean_of_samples, mode_of_samples
from evals.manifest import figure_argparser, prepare_figure
from evals.figure_utils import BOOTSTRAP_SEED, NBOOT, apply_serif_font
from evals.plot_style import apply_style

DEFAULT_OUT_NAME = "coins_calibration.pdf"
COINS_MANIFEST = "configs/figures/coins.yaml"

# Evals this figure needs, by name in the evals/manifest.py registry.
REQUIRES = ("coins",)

# Per-COIN colours (the within-panel grouping). Condition-level styling lives in
# plot_style.ROLE_STYLE and isn't used here — every panel is one condition.
GROUP_STYLE = {
    "sft": ("#6c757d", "Fully supervised (FS)"),
    "cons": ("#4878A6", "Consistency-trained (EC)"),
    "holdout": ("#6BAE5B", "Held out (H)"),
}
GROUP_ORDER = ("sft", "cons", "holdout")

# A different statistic from figure_utils.prop_ci: that one bootstraps a
# proportion over n binary outcomes, this one bootstraps the MEAN of K sampled
# program biases. Shares the deck's bootstrap constants so all error bars in the
# paper use the same resample count and seed.
_BOOT_RNG = np.random.default_rng(BOOTSTRAP_SEED)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion — well behaved near 0/1,
    where the normal approximation is not."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def _bootstrap_mean_ci(samples):
    s = np.asarray(samples, dtype=float)
    if s.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(s.mean())
    if s.size < 2 or float(s.std(ddof=0)) == 0.0:
        return mean, mean, mean
    idx = _BOOT_RNG.integers(0, s.size, size=(NBOOT, s.size))
    boot = s[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi)


def _r2(pred, target):
    """R^2 against the identity line (see domains/coins/train.py)."""
    pairs = [
        (p, t)
        for p, t in zip(pred, target)
        if p is not None and t is not None and not math.isnan(p)
    ]
    if len(pairs) < 2:
        return float("nan")
    pred, target = zip(*pairs)
    mean_t = sum(target) / len(target)
    ss_res = sum((p - t) ** 2 for p, t in zip(pred, target))
    ss_tot = sum((t - mean_t) ** 2 for t in target)
    return 1 - ss_res / ss_tot if ss_tot else float("nan")


def _panel(ax, records, x_key, *, wilson: bool, title=None, ylabel=None, xlabel=None):
    """Scatter of stated bias (MODE of the K samples — what the paper plots)
    against x_key. R^2 is annotated for both estimators, mode and mean, but the
    plotted points are always the mode."""
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="#999999", zorder=1)

    r2_by_group = {}
    for group in GROUP_ORDER:
        rows = [r for r in records if r["group"] == group]
        if not rows:
            continue
        color, _ = GROUP_STYLE[group]

        xs, ys, ys_mean = [], [], []
        xerr_lo, xerr_hi, yerr_lo, yerr_hi = [], [], [], []
        for r in rows:
            samples = r.get("pred_bias_samples") or []
            if samples:
                y = mode_of_samples(samples)
                y_mean = mean_of_samples(samples)
                _, ylo, yhi = _bootstrap_mean_ci(samples)
            else:
                y = r["pred_bias"] if r["pred_bias"] is not None else float("nan")
                y_mean = y
                ylo = yhi = y
            if wilson and r["rollout_n"]:
                x, xlo, xhi = _wilson_ci(int(r["rollout_h"]), int(r["rollout_n"]))
            else:
                x = r[x_key]
                xlo = xhi = x
            if x is None or y is None or (isinstance(y, float) and math.isnan(y)):
                continue
            xs.append(x)
            ys.append(y)
            ys_mean.append(y_mean)
            xerr_lo.append(max(0.0, x - xlo))
            xerr_hi.append(max(0.0, xhi - x))
            yerr_lo.append(max(0.0, y - ylo))
            yerr_hi.append(max(0.0, yhi - y))

        if not xs:
            continue
        ax.errorbar(
            xs,
            ys,
            xerr=np.vstack([xerr_lo, xerr_hi]) if wilson else None,
            yerr=np.vstack([yerr_lo, yerr_hi]),
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            alpha=0.45,
            zorder=2,
        )
        ax.scatter(xs, ys, s=42, color=color, edgecolor="white", lw=0.7, zorder=3)
        r2_by_group[group] = {
            "r2_mode": _r2(ys, xs),
            "r2_mean": _r2(ys_mean, xs),
        }

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":", color="#dddddd")
    if title:
        ax.set_title(title, fontsize=17, pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)

    txt = "\n".join(
        f"$R^2_{{{g.upper()}}}$ mode={v['r2_mode']:.2f} mean={v['r2_mean']:.2f}"
        for g, v in r2_by_group.items()
        if not math.isnan(v["r2_mode"])
    )
    if txt:
        ax.text(
            0.03,
            0.97,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="#2a2a2a",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85),
        )
    return r2_by_group


def build_figure(loaded, out_path: Path):
    apply_style()
    apply_serif_font()

    n = len(loaded)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 9.2), squeeze=False)

    for col, (label, records) in enumerate(loaded):
        # Plotted points use the mode-of-K estimator (the paper's statistic);
        # the axis label says so, and the corner text reports mean-of-K R^2
        # alongside for comparison.
        _panel(
            axes[0][col],
            records,
            "gt_bias",
            wilson=False,
            title=label,
            ylabel="Stated bias (mode of K)" if col == 0 else None,
            xlabel="True coin bias",
        )
        _panel(
            axes[1][col],
            records,
            "rollout_bias",
            wilson=True,
            ylabel="Stated bias (mode of K)" if col == 0 else None,
            xlabel="Empirical rollout bias",
        )

    axes[0][0].text(
        -0.28,
        0.5,
        "Calibration",
        transform=axes[0][0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=16,
    )
    axes[1][0].text(
        -0.28,
        0.5,
        "Self-consistency",
        transform=axes[1][0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=16,
    )

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="none",
            markersize=9,
            markerfacecolor=GROUP_STYLE[g][0],
            markeredgecolor="white",
            label=GROUP_STYLE[g][1],
        )
        for g in GROUP_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0.02, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# wandb mirroring (additive — disk outputs are unchanged)
# ---------------------------------------------------------------------------


def _estimators(record):
    """(pred_mode, pred_mean) for a per-coin record, tolerating older JSONs
    that predate the explicit pred_mode/pred_mean fields."""
    samples = record.get("pred_bias_samples") or []
    mode = record.get("pred_mode")
    mean = record.get("pred_mean")
    if mode is None:
        mode = mode_of_samples(samples) if samples else record.get("pred_bias")
    if mean is None:
        mean = mean_of_samples(samples) if samples else record.get("pred_bias")
    return mode, mean


def _wandb_eval_required(raw_manifest) -> bool:
    """Same opt-out contract as consistency.trainer.wandb_required, reading the
    flag from the manifest instead of a hydra config (imported locally so
    --plot-only keeps working without torch)."""
    if os.environ.get("WANDB_MODE", "").lower() in ("disabled", "offline"):
        return False
    return bool((raw_manifest.get("wandb") or {}).get("required", True))


def log_figure_to_wandb(mf, conditions, loaded, out_path: Path) -> None:
    """One wandb run per figure evaluation: summary R^2 for every
    column x group x estimator, the per-coin tables, and the figure file."""
    with open(mf.path) as f:
        raw_manifest = yaml.safe_load(f) or {}
    required = _wandb_eval_required(raw_manifest)

    try:
        import wandb
    except Exception as exc:
        if required:
            raise RuntimeError(
                "wandb is required but could not be imported. Install wandb, "
                "or opt out explicitly with WANDB_MODE=offline (still records "
                "locally), WANDB_MODE=disabled, or wandb.required: false in "
                "the manifest."
            ) from exc
        print(f"wandb unavailable, skipping figure logging: {exc}")
        return

    slugs = [
        (cond.run_dir.name if cond.run_dir is not None else "base") for cond in conditions
    ]
    run_name = f"fig_{mf.path.stem}_" + "-".join(
        f"{slug}@{cond.ckpt}" for slug, cond in zip(slugs, conditions)
    )
    wandb_cfg = raw_manifest.get("wandb") or {}
    try:
        run = wandb.init(
            project=wandb_cfg.get("project", "self_ctrl_coins"),
            name=run_name,
            job_type="figure",
            config=raw_manifest,
        )
    except Exception as exc:
        if required:
            raise RuntimeError(
                "wandb.init failed and wandb logging is required. Fix the "
                "wandb setup, or opt out explicitly with WANDB_MODE=offline / "
                "WANDB_MODE=disabled / wandb.required: false in the manifest."
            ) from exc
        print(f"wandb.init failed, skipping figure logging: {exc}")
        return

    summary = {}
    for slug, (_, records) in zip(slugs, loaded):
        table = wandb.Table(
            columns=[
                "coin",
                "group",
                "gt_bias",
                "pred_mode",
                "pred_mean",
                "rollout_bias",
                "n_valid",
            ]
        )
        for r in records:
            mode, mean = _estimators(r)
            table.add_data(
                r["coin"],
                r["group"],
                r["gt_bias"],
                mode,
                mean,
                r["rollout_bias"],
                len(r.get("pred_bias_samples") or []),
            )
        wandb.log({f"{slug}/per_coin": table})

        for group in GROUP_ORDER + ("all",):
            rows = [r for r in records if group == "all" or r["group"] == group]
            if not rows:
                continue
            modes, means = zip(*(_estimators(r) for r in rows))
            gt = [r["gt_bias"] for r in rows]
            rollout = [r["rollout_bias"] for r in rows]
            summary[f"{slug}/{group}/r2_gt_mode"] = _r2(modes, gt)
            summary[f"{slug}/{group}/r2_gt_mean"] = _r2(means, gt)
            summary[f"{slug}/{group}/r2_rollout_mode"] = _r2(modes, rollout)
            summary[f"{slug}/{group}/r2_rollout_mean"] = _r2(means, rollout)

    wandb.log(summary)

    if out_path.exists():
        if out_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            wandb.log({"figure": wandb.Image(str(out_path))})
        else:
            wandb.save(str(out_path))

    run.finish()


def main():
    parser = figure_argparser(__doc__.split("\n\n", 1)[0])
    # The coins deck has its own manifest; the shared default points at moral.
    parser.set_defaults(manifest=COINS_MANIFEST)

    args = parser.parse_args()
    prepared = prepare_figure(args, REQUIRES)
    if prepared is None:
        return
    mf, conditions = prepared

    loaded = []
    for cond in conditions:
        with open(cond.results_dir / "per_coin_debug.json") as f:
            loaded.append((cond.label, json.load(f)["records"]))

    out_path = Path(args.out) if args.out else mf.figures_dir / DEFAULT_OUT_NAME
    build_figure(loaded, out_path)
    log_figure_to_wandb(mf, conditions, loaded, out_path)


if __name__ == "__main__":
    main()
