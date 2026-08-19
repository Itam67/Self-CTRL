# Self-CTRL
Self-Consistency Training with Reinforcement Learning

[![arXiv](https://img.shields.io/badge/arXiv-2606.18327-b31b1b.svg)](https://arxiv.org/abs/2606.18327)

This repository contains code for "[Self-CTRL: Self-Consistency Training with
Reinforcement Learning](https://arxiv.org/abs/2606.18327)".

Self-CTRL trains language models to make their stated *explanations* of their behavior consistent with the *behavior* they actually produce.

| Domain | Explanation | Behavior | Reward |
| --- | --- | --- | --- |
| `domains/moral/` | a stated safety policy | a response to a request | jury of LM judges |
| `domains/coins/` | a Python program stating a coin's bias | a rollout of H/T flips | Bernoulli log-likelihood |

We include instructions for adapting this codebase to new domains/problems. 
## Setup

```bash
pip install -r requirements.txt          # Python >= 3.10
huggingface-cli login                    # Llama-3.1-8B-Instruct is a gated repo
gcloud auth application-default login    # Gemini judges (Vertex ADC)
export GOOGLE_CLOUD_PROJECT=<project-id>
```

Run every command from the repo root. Gemini authentication is only required for the moral-domain evaluations. Training streams SFT data from Hugging Face; `wandb` is optional.

## Moral domain training

We currently have one config per a training type each writing to its own
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

Adapters get saved to `models/moral/<run>/ckpt_<step>/`, checkpoints and metrics to
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

The coin datasets are in `data/coins/`. SFT uses 24,000 coin demonstrations plus 5,000 streamed OpenCodeInstruct examples. The paper's splits contain 50 fully supervised coins, 40 rollout-only coins used for consistency training, and 10 held-out coins.

## Applying Self-CTRL to new domains

The core trainer (`consistency/trainer.py`) is domain agnostic. To instantiate a new domain requires creating a 
`ConsistencyConfig` of callables plus a Hydra config. Currently, `domains/coins/` and
`domains/moral/` are two complete examples. 

To apply Self-CTRL to a new
problem, define what an *explanation* and a *behavior* are for your task, then
provide:

| Callable | Signature | Returns |
| --- | --- | --- |
| `load_data` | `(cfg)` | train/val loaders yielding (behavior_prompts, explanation_prompts) batches, plus an `extra` dict to store additional domain specific information |
| `collect_behaviors` | `(model, tok, prompts, cfg, extra)` | behaviors `[B][K]`, their NLLs `[B, K]`, `extra` |
| `collect_explanations` | `(model, tok, prompts, cfg, extra)` | explanations `[B][K]`, NLLs `[B, K]`, `valid_mask` (or `None`), `extra` |
| `reward_fn` | `(behaviors, explanations, model, tok, cfg, extra)` | rewards `[B, K]` scoring behavior–explanation consistency |

`cont_training_loss_fn`, `untouched_kl_fn`, `eval_fn`, and `verbose_logging`
are optional regularizers and hooks. 

```python
cons_cfg = ConsistencyConfig(load_data=..., collect_behaviors=...,
                             collect_explanations=..., reward_fn=...)

ConsistencyTrainer(model, tokenizer, optimizer, device, cfg, cons_cfg).train()
```

Config-wise, you can inherit the shared keys (`learning.bw` = λ, `sampling.*`, K via
`n_beams`, `save_dir`/`results_dir`) from an existing domain's YAML — see
`configs/coins.yaml` or `configs/moral.yaml` for the annotated set.

## Figures

| Figure | Description | Command | Required evals |
| --- | --- | --- | --- |
| Fig. 2 | Coins calibration / self-consistency | `python -m evals.coins.plot_calibration` | per-coin programs + rollouts |
| Fig. 3 | Consistency (jury eval score) over training | `python -m evals.moral.plot_consistency_curves` | none — reads the trainer's `eval_snapshots/` |
| Fig. 4 | Safety vs. simulatability (Pareto) | `python -m evals.moral.plot_safety_sim` | harmbench, NSG |
| Fig. 5 | Counterfactual consistency | `python -m evals.moral.plot_cf` | cf_consistency |
| Fig. 6 | Non-toxic compliance + capabilities | `python -m evals.moral.plot_capabilities` | wildchat, mmlu |

Set the checkpoints for each condition in `configs/figures/moral_llama.yaml` or `configs/figures/coins.yaml`, then run the plotting commands above. The plot scripts automatically run any missing evaluations and save the resulting figures to `results/final_figures/`.

Useful flags are `--status`, `--plot-only`, `--only "<label>"`, `--force`, `--manifest`, and `--out`.

## Citation

```bibtex
@misc{pres2026selfctrl,
  title         = {Self-CTRL: Self-Consistency Training with Reinforcement Learning},
  author        = {Pres, Itamar and Ruis, Laura and Ghebreselassie, Melat and Li, Belinda Z. and Andreas, Jacob},
  year          = {2026},
  eprint        = {2606.18327},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2606.18327},
  url           = {https://arxiv.org/abs/2606.18327},
}
```
