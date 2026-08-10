"""Shared figure manifest. Stores which runs are in the paper, and where their metrics live."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every role a figure knows how to style. A manifest using anything else is a
# typo, and one that silently fell through to a default color would be worse
# than an error.
ROLES = (
    # moral domain
    "base",
    "expl",
    "mixed",
    "beh",
    "expl_baseline",
    "beh_baseline",
    # coins domain
    "sft_baseline",
    "cons_trained",
    "oracle",
)


def flat_label(label: str) -> str:
    """Single-line form of a bar label (they carry newlines to wrap captions)."""
    return " ".join(label.split())


_CKPT_DIR_RE = re.compile(r"ckpt_(\d+)")


def get_checkpoint_dirs(run_dir: Path, ckpt_filter=None):
    """Find ckpt_<step> adapter dirs under run_dir. Returns sorted [(step, dir), ...].

    Step 0 is excluded as it denotes the untrained base model, which every eval
    loads with no adapter rather than from a checkpoint dir.
    """
    ckpts = []
    for d in run_dir.iterdir():
        if not d.is_dir():
            continue
        m = _CKPT_DIR_RE.match(d.name)
        if m:
            step = int(m.group(1))
            if step == 0:
                continue
            if ckpt_filter is None or step in ckpt_filter:
                ckpts.append((step, d))
    return sorted(ckpts, key=lambda x: x[0])


@dataclass
class Condition:
    """A single checkpoint, or the base model."""

    label: str
    role: str
    run_dir: Optional[Path]  # adapter dir (trainer save_dir), None for base
    ckpt: Optional[int]  # headline checkpoint step, None for base
    results_dir: Path  # canonical dir holding this condition's metrics

    @property
    def is_base(self) -> bool:
        return self.run_dir is None

    @property
    def lora_path(self) -> Optional[str]:
        """Adapter to load, or None to evaluate the untouched base model."""
        if self.is_base:
            return None
        return str(self.run_dir / f"ckpt_{self.ckpt}")

    @property
    def nsg_metrics_path(self) -> Path:
        return self.results_dir / "ambiguous" / "nsg_metrics.json"


@dataclass
class Panel:
    """One panel of the consistency-curves figure (a training run, not a ckpt)."""

    title: str
    run_dirs: list  # merged step-by-step; earlier dirs win on overlap


@dataclass
class Manifest:
    path: Path
    model_name: str
    models_root: Path
    results_root: Path
    figures_dir: Path
    conditions: list
    panels: list
    eval_cfg: dict = field(default_factory=dict)

    @property
    def model_slug(self) -> str:
        return self.model_name.replace("/", "_")

    def by_label(self, label: str) -> Condition:
        """Look up a condition by label.

        Labels carry newlines so bar captions wrap, which makes them awkward to
        type; matching is therefore whitespace-insensitive, so --only "Expl.
        baseline" finds "Expl.\\nbaseline"."""
        want = " ".join(label.split())
        for c in self.conditions:
            if c.label == label or " ".join(c.label.split()) == want:
                return c
        known = ", ".join(repr(flat_label(c.label)) for c in self.conditions)
        raise KeyError(
            f"no condition labelled {label!r} in {self.path}. Known: {known}"
        )

    def cf_filename(self) -> str:
        """Filename evals/moral/cf.py writes, for the manifest's settings."""
        cf = self.eval_cfg.get("cf", {})
        slug = (
            str(cf.get("cf_model", "gemini-2.5-flash"))
            .replace("/", "_")
            .replace(".", "-")
        )
        return f"cf_consistency_{slug}_n{int(cf.get('n_prompts', 10))}_percat.json"


def _resolve(root: Path, value) -> Path:
    """Resolve a manifest path against the repo root when it isn't absolute."""
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def load_manifest(path) -> Manifest:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        raw = yaml.safe_load(f)

    model_name = raw["model_name"]
    models_root = _resolve(REPO_ROOT, raw.get("models_root", "models/moral"))
    results_root = _resolve(REPO_ROOT, raw.get("results_root", "results/moral"))
    figures_dir = _resolve(REPO_ROOT, raw.get("figures_dir", "results/final_figures"))
    model_slug = model_name.replace("/", "_")

    conditions = []
    for entry in raw.get("conditions", []):
        role = entry["role"]
        if role not in ROLES:
            raise ValueError(
                f"{path}: condition {entry['label']!r} has unknown role {role!r}. "
                f"Expected one of {', '.join(ROLES)}."
            )
        run_dir_name = entry.get("run_dir")
        if run_dir_name is None:
            run_dir, ckpt = None, None
            results_dir = results_root / "_base_evals" / model_slug
        else:
            if entry.get("ckpt") is None:
                raise ValueError(
                    f"{path}: condition {entry['label']!r} sets run_dir but no ckpt "
                    "(the headline checkpoint step is required)."
                )
            run_dir = _resolve(models_root, run_dir_name)
            ckpt = int(entry["ckpt"])
            results_dir = results_root / Path(run_dir_name).name / f"ckpt_{ckpt}"
        conditions.append(
            Condition(
                label=entry["label"],
                role=role,
                run_dir=run_dir,
                ckpt=ckpt,
                results_dir=results_dir,
            )
        )

    # Panels read eval_snapshots/, which the trainer writes under results_dir —
    # so these resolve against results_root, not models_root.
    panels = [
        Panel(
            title=p["title"],
            run_dirs=[_resolve(results_root, d) for d in p["run_dirs"]],
        )
        for p in raw.get("panels", [])
    ]

    return Manifest(
        path=path,
        model_name=model_name,
        models_root=models_root,
        results_root=results_root,
        figures_dir=figures_dir,
        conditions=conditions,
        panels=panels,
        eval_cfg=raw.get("eval", {}) or {},
    )


DEFAULT_MANIFEST = "configs/figures/moral_llama.yaml"


def figure_argparser(description: str) -> argparse.ArgumentParser:
    """Argparse parent shared by every figure script."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"Figure manifest YAML (default {DEFAULT_MANIFEST}).",
    )
    p.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="LABEL",
        help="Compute only this condition's evals, then exit without plotting. "
        "Repeatable. Use this to fan the eval pass out over SLURM array tasks.",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Never run an eval; plot from the metric JSONs already on disk and "
        "skip (with a warning) any condition still missing them.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute evals even when their metric JSON already exists.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Override the output figure path.",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Print which evals are already computed for every condition — the "
        "whole deck, not just this figure's — then exit.",
    )
    return p


def select_conditions(mf: Manifest, only) -> list:
    """Resolve --only labels (or all conditions when it isn't given)."""
    if not only:
        return list(mf.conditions)
    return [mf.by_label(label) for label in only]


# Eval registry
#
# ONE entry per eval, and it is the only place that knows two things: where that
# eval's metrics land for a condition, and how to run it. Figures declare the
# evals they need BY NAME (see REQUIRES in each plot_*.py), so:
#
#   * the "is it already computed?" check and the "run it" call can never
#     disagree about the path — they read the same spec;
#   * two figures that need the same eval share the result automatically. The
#     output path depends only on (condition, eval), never on which figure asked,
#     so whichever figure runs first pays for it and the rest hit the cache.
#
# Adding an eval to a second figure is therefore a one-word change to that
# figure's REQUIRES, with no risk of recomputation.


@dataclass
class EvalSpec:
    """How to locate and (lazily) run one eval for one condition."""

    name: str
    # (cond, mf) -> Path of the metrics file this eval writes.
    locate: Callable
    # (cond, mf) -> None. Imports the eval module lazily so --plot-only needs no torch.
    run: Callable

    def path(self, cond: "Condition", mf: "Manifest") -> Path:
        return self.locate(cond, mf)

    def ensure(self, cond: "Condition", mf: "Manifest", force=False) -> Path:
        """Compute this eval unless its metrics file is already on disk."""
        out = self.path(cond, mf)
        if out.exists() and not force:
            print(f"  [{flat_label(cond.label)}] {self.name}: cached ({out})")
            return out
        print(f"  [{flat_label(cond.label)}] {self.name}: running...")
        self.run(cond, mf)
        if not out.exists():
            raise RuntimeError(
                f"{self.name} finished but wrote no {out}. If the eval's output "
                "layout changed, update its EvalSpec in evals/manifest.py."
            )
        return out


def _in_results(filename):
    """locate() for evals that write straight into the condition's results dir."""
    return lambda cond, mf: cond.results_dir / filename


def _run_harmbench(cond, mf):
    from evals.moral.harmbench import run_harmbench

    cfg = mf.eval_cfg.get("harmbench", {})
    run_harmbench(
        model_name=mf.model_name,
        lora_path=cond.lora_path,
        output_dir=str(cond.results_dir),
        max_prompts=int(cfg.get("max_prompts", 200)),
        judge_model=cfg.get("judge_model", "gemini-2.5-flash"),
        seed=int(cfg.get("seed", 42)),
    )


def _run_wildchat(cond, mf):
    from evals.moral.wildchat import run_wildchat_refusal

    cfg = mf.eval_cfg.get("wildchat", {})
    run_wildchat_refusal(
        model_name=mf.model_name,
        lora_path=cond.lora_path,
        output_dir=str(cond.results_dir),
        max_prompts=int(cfg.get("max_prompts", 300)),
        judge_model=cfg.get("judge_model", "gemini-2.5-flash"),
        seed=int(cfg.get("seed", 42)),
    )


def _run_mmlu(cond, mf):
    from evals.moral.mmlu import run_mmlu

    cfg = mf.eval_cfg.get("mmlu", {})
    run_mmlu(
        model_name=mf.model_name,
        lora_path=cond.lora_path,
        output_dir=str(cond.results_dir),
        max_questions=int(cfg.get("max_questions", 200)),
        seed=int(cfg.get("seed", 42)),
    )


def _run_nsg(cond, mf):
    from evals.moral.nsg import ALL_PHASES, run_eval

    cfg = mf.eval_cfg.get("nsg", {})
    run_eval(
        run_dir=str(cond.run_dir) if cond.run_dir else None,
        ckpt_filter={cond.ckpt} if cond.ckpt is not None else {0},
        phases=set(ALL_PHASES),
        gemini_model=cfg.get("gemini_model", "gemini-2.5-flash"),
        no_cache=False,
        model_name=mf.model_name,
        lora_path=cond.lora_path,
        gen_batch_size=int(cfg.get("gen_batch_size", 8)),
        out_dir=cond.nsg_metrics_path.parent,
    )


def _run_cf(cond, mf):
    from evals.moral.cf import run as run_cf

    cfg = mf.eval_cfg.get("cf", {})
    run_cf(
        run_dir=str(cond.run_dir) if cond.run_dir else str(mf.models_root),
        ckpt_filter={cond.ckpt} if cond.ckpt is not None else {0},
        cf_model=cfg.get("cf_model", "gemini-2.5-flash"),
        judge_model=cfg.get("judge_model", "gemini-2.5-flash"),
        n_prompts=int(cfg.get("n_prompts", 10)),
        use_cache=True,
        gen_batch_size=int(cfg.get("gen_batch_size", 8)),
        model_name=mf.model_name,
        lora_path=cond.lora_path,
        out_dir=cond.results_dir,
    )


def _run_coins(cond, mf):
    from evals.coins.per_coin import run_per_coin

    cfg = mf.eval_cfg.get("coins", {})
    run_per_coin(
        model_name=mf.model_name,
        lora_path=cond.lora_path,
        output_dir=str(cond.results_dir),
        n_programs=int(cfg.get("n_programs", 10)),
        n_flips=int(cfg.get("n_flips", 100)),
        max_new_tokens=int(cfg.get("max_new_tokens", 150)),
        temp=float(cfg.get("temp", 1.0)),
        top_p=float(cfg.get("top_p", 0.9)),
        seed=int(cfg.get("seed", 42)),
        gen_batch_size=int(cfg.get("gen_batch_size", 8)),
    )


EVALS = {
    spec.name: spec
    for spec in [
        EvalSpec("harmbench", _in_results("harmbench_metrics.json"), _run_harmbench),
        EvalSpec("wildchat", _in_results("wildchat_metrics.json"), _run_wildchat),
        EvalSpec("mmlu", _in_results("mmlu_metrics.json"), _run_mmlu),
        EvalSpec("nsg", lambda cond, mf: cond.nsg_metrics_path, _run_nsg),
        EvalSpec("cf", lambda cond, mf: cond.results_dir / mf.cf_filename(), _run_cf),
        EvalSpec("coins", _in_results("per_coin_debug.json"), _run_coins),
    ]
}


def get_eval(name: str) -> EvalSpec:
    try:
        return EVALS[name]
    except KeyError:
        raise KeyError(
            f"unknown eval {name!r}; registered: {', '.join(EVALS)}"
        ) from None


def gather(conditions, eval_names, mf: Manifest, *, plot_only: bool, force: bool):
    """Ensure every condition's required evals, and drop the ones not ready.

    eval_names are keys into EVALS. Returns (kept_conditions, missing) where
    `missing` is [(label, reason), ...]. A condition is kept only when EVERY
    file the figure needs is present, so a half-evaluated condition can never
    silently render as a partial bar.
    """
    specs = [get_eval(n) for n in eval_names]
    kept, missing = [], []
    for cond in conditions:
        problems = []
        for spec in specs:
            try:
                path = (
                    spec.path(cond, mf)
                    if plot_only
                    else spec.ensure(cond, mf, force=force)
                )
            except Exception as exc:  # eval failed — report, don't abort the figure
                problems.append(f"{spec.name}: {exc}")
                continue
            if not Path(path).exists():
                problems.append(f"missing {path}")
        if problems:
            missing.append((cond.label, "; ".join(problems)))
        else:
            kept.append(cond)
    return kept, missing


def prepare_figure(args, eval_names):
    """The shared front half of every figure script.

    Loads the manifest, services --status/--only, ensures the figure's required
    evals for the selected conditions, and reports what got skipped.

    Returns (manifest, conditions) to plot, or None when the caller should exit
    without plotting (--status, --only, or nothing left to draw).
    """
    mf = load_manifest(args.manifest)

    if args.status:
        report_status(mf)
        return None

    conditions = select_conditions(mf, args.only)
    conditions, missing = gather(
        conditions, eval_names, mf, plot_only=args.plot_only, force=args.force
    )
    report_missing(missing)

    if args.only and eval_names:
        print("--only given: evals done, skipping the figure.")
        return None
    if not conditions and eval_names:
        print("No conditions have metrics yet — nothing to plot.")
        return None
    return mf, conditions


def report_missing(missing) -> None:
    if not missing:
        return
    print("\nSkipped conditions (metrics not available):")
    for label, reason in missing:
        print(f"  - {flat_label(label)}: {reason}")
    print()


def manifest_evals(mf: Manifest) -> list:
    """The evals this manifest is about — the ones it configures under `eval:`.

    A coins manifest shouldn't report on HarmBench and a moral one shouldn't
    report on per-coin programs. Falls back to every registered eval when a
    manifest declares no eval settings at all.
    """
    named = [n for n in EVALS if n in mf.eval_cfg]
    return named or list(EVALS)


def report_status(mf: Manifest) -> None:
    """Print, for every condition, which of this deck's evals are on disk.

    Covers the whole deck rather than one figure's slice, so `--status` from any
    plot script answers "what still needs computing anywhere?".
    """
    names = manifest_evals(mf)
    width = max(len(flat_label(c.label)) for c in mf.conditions)
    print(f"Eval status for {mf.path}:\n")
    print(f"  {'condition':<{width}}  " + "  ".join(f"{n:>9}" for n in names))
    for cond in mf.conditions:
        marks = [
            f"{'ok' if EVALS[n].path(cond, mf).exists() else '-':>9}" for n in names
        ]
        print(f"  {flat_label(cond.label):<{width}}  " + "  ".join(marks))
    print()
