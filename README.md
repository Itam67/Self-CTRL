# Self-CTRL
Self-Consistency Training with Reinforcement Learning

A model produces an *explanation* of its behavior
and a *behavior* out of context. The model is trained so the two are consistent.

| Domain | Explanation | Behavior | Reward |
| --- | --- | --- | --- |
| `domains/moral/` | a stated safety policy | a response to a request | jury of LM judges |
| `domains/coins/` | a Python program stating a coin's bias | a rollout of H/T flips | Bernoulli log-likelihood |


## Setup

```bash
pip install -r requirements.txt          # Python >= 3.10
huggingface-cli login                    # Llama-3.1-8B-Instruct is a gated repo
gcloud auth application-default login    # Gemini judges (Vertex ADC)
export GOOGLE_CLOUD_PROJECT=<project-id>
```

Run every command from the repo root: all config paths are relative to it and
hydra is set not to chdir.

The Gemini steps are needed only for the moral evals (HarmBench, WildChat, NSG,
counterfactual) — training and the coins figure never call an API. Training does
need network: the moral runs stream their SFT data from
`nvidia/Nemotron-SFT-Instruction-Following-Chat-v2` and coins SFT streams 5,000
`nvidia/OpenCodeInstruct` examples, neither of which ships in `data/`. `wandb` is
optional; if it isn't configured, logging is skipped.

## Moral domain  training

One config per condition, at the paper's settings, each writing to its own
`save_dir` / `results_dir`:

| Condition | Command | λ | cont | aux | ukl |
| --- | --- | --- | --- | --- | --- |
| Explanation (default) | `python domains/moral/train.py` | 0.0 | 0.0 | 0.0 | 0.1 |
| Mixed | `python domains/moral/train.py --config-name moral_mixed` | 0.5 | 0.25 | 0.4 | 0.0 |
| Behavior | `python domains/moral/train.py --config-name moral_beh` | 1.0 | 0.25 | 0.4 | 0.1 |
| Expl. baseline | `python domains/moral/baseline.py` | 0.0 | 0.25 | — | 0.1 |
| Beh. baseline | `python domains/moral/baseline.py --config-name moral_baseline_beh` | 1.0 | 0.25 | — | 0.1 |

λ (`learning.bw`) picks which side is updated; `cont` weights the
continued-training SFT loss that holds general capability in place; `aux`
weights the engagement signal mixed into the jury reward; `ukl` weights the KL
anchor on the side *not* being updated. The explanation run uses no SFT loss, so
its `cont_training` data block is switched off rather than weighted to zero.

`configs/moral.yaml` holds everything the five conditions share: model, lr,
epochs, sampling, SFT dataset. The six per-condition keys are left mandatory (`???`), so
composing it directly fails loudly instead of training something unintended. Any
key can still be overridden on the command line:
`python domains/moral/train.py learning.lr=2e-5`.

Adapters get saved to to `models/moral/<run>/ckpt_<step>/`, eval snapshots and metrics to
`results/moral/<run>/`.

The main trainer's reward is the consistency jury. Explanation and behavior are
scored *against each other*. The baseline (`domains/moral/baseline.py`) is
identical in every other respect but scores each side against the frozen base
policy's own self-report, with the other side absent from the judge's context.

## Coins domain training

First we apply SFT so the model learns each coin's latent bias out of
context; then we apply Self-CTRL, so it learns to *state* that bias.

| Stage | Command | Settings |
| --- | --- | --- |
| SFT | `python domains/coins/sft.py` | lr 3e-4, batch 4, 5 epochs, LoRA r=16 |
| Self-CTRL | `python domains/coins/train.py` | lr 1e-5, batch 10, K=10, λ=0, λ_KL=0.3 |
| Oracle | `python domains/coins/sft.py --config-name coins_sft_oracle` | as SFT, on the 90-coin corpus |

The oracle is the paper's ground-truth-supervision upper bound and the third
column of the figure. It is trained with the same SFT recipe, but its corpus gives program
supervision for 90 coins instead of 50, so the consistency coins are handed
their biases rather than having to discover them. The same 10 coins are held
out, so that column stays comparable across all three conditions.

SFT trains on 24,000 coin
demonstrations plus 5,000 OpenCodeInstruct examples (streamed from HF), writes a
checkpoint per epoch named by optimizer step, and the paper's end-of-epoch-4
checkpoint is `ckpt_29000` — which is what `configs/coins.yaml`'s `load_dir`
points at. Each stage is about 5 GPU-hours on one H100.

Consistency training updates the explanation side only (`bw=0.0`) and adds a
behavior-preserving anchor. One rollout per prompt is sampled from the
SFT-initialised policy before RL and frozen, and its NLL under the training
policy acts as a Monte Carlo forward-KL anchor (`cont_training_loss_weight=0.3`).

The datasets are in `data/coins/` (37 MB — `sft_*` for the standard corpus,
`oracle_*` for the 90-coin one, `cons_*` for consistency training). The paper's three supervision groups are recovered from these files
by `domains.coins.data.coin_groups()`: 50 coins supervised with both rollouts
and programs, 40 rollout-only coins used for consistency training, and 10 held
out of both stages.

## Figures



| Paper | Figure | Command | Evals it needs |
| --- | --- | --- | --- |
| Fig. 2 | Coins calibration / self-consistency | `python -m evals.coins.plot_calibration` | per-coin programs + rollouts |
| Fig. 3 | Consistency (jury eval score) over training | `python -m evals.moral.plot_consistency_curves` | none — reads the trainer's `eval_snapshots/` |
| Fig. 4 | Safety vs. simulatability (Pareto) | `python -m evals.moral.plot_safety_sim` | harmbench, NSG |
| Fig. 5 | Counterfactual consistency | `python -m evals.moral.plot_cf` | cf_consistency |
| Fig. 6 | Non-toxic compliance + capabilities | `python -m evals.moral.plot_capabilities` | wildchat, mmlu |

Each figure end to end — train, point the manifest at your checkpoints, plot:

```bash
# ── Fig. 3 — consistency over training ─────────────────────────────────────
python domains/moral/train.py                                       # expl  (λ=0)
python domains/moral/train.py --config-name moral_mixed             # mixed (λ=0.5)
python domains/moral/train.py --config-name moral_beh               # beh   (λ=1.0)
#   → models/moral/<run>/ckpt_<step>/                    adapters
#   → results/moral/<run>/eval_snapshots/                curves this figure reads
python -m evals.moral.plot_consistency_curves
#   → results/final_figures/consistency_curves_combined.pdf

# ── Figs. 4, 5, 6 — the moral bar/scatter figures ──────────────────────────
python domains/moral/train.py                                       # expl
python domains/moral/train.py --config-name moral_mixed             # mixed
python domains/moral/train.py --config-name moral_beh               # beh
python domains/moral/baseline.py                                    # expl baseline
python domains/moral/baseline.py --config-name moral_baseline_beh   # beh baseline
#   → models/moral/<run>/ckpt_<step>/                    adapters, one <run> per condition
# Set each condition's `ckpt:` in configs/figures/moral_llama.yaml, then:

python -m evals.moral.plot_safety_sim        # Fig. 4
#   → results/moral/<run>/ckpt_<n>/harmbench_metrics.json
#   → results/moral/<run>/ckpt_<n>/ambiguous/nsg_metrics.json
#   → results/final_figures/safety_vs_simulatability_scatter.pdf

python -m evals.moral.plot_capabilities      # Fig. 6
#   → results/moral/<run>/ckpt_<n>/wildchat_metrics.json
#   → results/moral/<run>/ckpt_<n>/mmlu_metrics.json
#   → results/final_figures/overrefusal_capabilities_bars.pdf

python -m evals.moral.plot_cf                # Fig. 5
#   → results/moral/<run>/ckpt_<n>/cf_consistency_gemini-2-5-flash_n10_percat.json
#   → results/final_figures/cf_consistency_bars.pdf

# The Base condition's metrics land in the shared per-model dir instead:
#   → results/moral/_base_evals/meta-llama_Llama-3.1-8B-Instruct/

# ── Fig. 2 — coins calibration ─────────────────────────────────────────────
python domains/coins/sft.py                                # → models/coins/sft/ckpt_<step>/
python domains/coins/train.py                              # → models/coins/cons_bw0.0_cont0.3/ckpt_<step>/
python domains/coins/sft.py --config-name coins_sft_oracle # → models/coins/oracle/ckpt_<step>/
# Set each condition's `ckpt:` in configs/figures/coins.yaml, then:
python -m evals.coins.plot_calibration
#   → results/coins/<run>/ckpt_<n>/per_coin_debug.json
#   → results/final_figures/coins_calibration.pdf
```

**The plot scripts are the only entry points you need.** Each one checks whether
its evals have been computed for every condition and runs the missing ones
itself; the eval modules never have to be invoked by hand. Results are keyed by
(condition, eval) alone so an eval computed for
one figure is reused by any other figure that needs it, and the base model is
evaluated once for the whole deck. `evals/manifest.py`'s `EVALS` registry is the
single place that knows where each eval writes and how to run it; a figure just
lists names in its `REQUIRES`.

Shared flags: `--status` (which evals exist for every condition across the whole
deck), `--plot-only` (never run an eval; skip conditions whose metrics are
missing — the eval modules are imported lazily, so this renders on a machine
with no torch), `--only "<label>"` (compute just that condition and exit — the
SLURM-array hook), `--force` (recompute even when cached), `--manifest`, `--out`.

```
$ python -m evals.moral.plot_cf --status
  condition              harmbench   wildchat       mmlu        nsg         cf
  Base                          ok         ok         ok         ok         ok
  Expl. baseline                ok         ok         ok         ok         ok
  Beh. baseline                  -          -          -          -          -
```

**Which runs are in a figure is defined once**, in a manifest —
`configs/figures/moral_llama.yaml` for the moral deck and
`configs/figures/coins.yaml` for coins. It holds condition labels, run dirs,
headline checkpoints, and eval settings, and every figure in that deck reads it,
so the condition list can't drift between figures. Copy one and pass
`--manifest` to add a model family. `--status` reports only the evals a manifest
actually configures, so each deck shows its own.

Its `run_dir` values already match the training configs' `save_dir`s, so the
runs above land where the figures look. The `ckpt` values are placeholders —
set each to the checkpoint you want reported. Adapters and metrics live in two
mirrored trees, keyed by the run dir's basename, so `models/` stays
adapters-only:

```
<models_root>/<run_dir>/ckpt_<step>/       adapter weights (trainer save_dir)
<results_root>/<run_dir>/ckpt_<step>/      that checkpoint's metric JSONs
<results_root>/<run_dir>/eval_snapshots/   training curves (trainer-written)
<results_root>/_base_evals/<model_slug>/   base-model metrics, shared by all runs
```

Each condition also declares a `role` — `base`, `expl`, `mixed`, `beh`,
`expl_baseline`, `beh_baseline` for moral, and `sft_baseline`, `cons_trained`,
`oracle` for coins. Figures key their colors off the role rather than the label,
so relabelling a condition never silently recolors it.
