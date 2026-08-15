"""Per-coin calibration eval for the coins domain (post-hoc, one checkpoint)."""

import functools
import json
from pathlib import Path

print = functools.partial(print, flush=True)

import torch

from consistency.model_utils import (
    load_model_tok,
    sample_conts_gen,
    sample_k_conts_gen,
)
from domains.coins.data import DATA_DIR, coin_groups
from domains.coins.program import program_biases
from evals.coins.stats import mean_of_samples, mode_of_samples


BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_N_PROGRAMS = 10
DEFAULT_N_FLIPS = 100
DEFAULT_MAX_NEW_TOKENS = 150
DEFAULT_TEMP = 1.0
DEFAULT_TOP_P = 0.9

_TEMPLATE_COIN = "__COIN__"


def prompt_templates(cons_path=None):
    """The (verb, rollout) prompt pair with the coin name placeheld.

    Taken from the shipped consistency data so the eval can't drift from how the
    model was trained. Keys on the record's own `coin` field rather than
    searching the text: coin labels nest ("Coin_A" is a prefix of "Coin_AZ"), so
    a containment match could rewrite the wrong span.
    """
    cons_path = Path(cons_path or DATA_DIR / "cons_train.jsonl")
    with open(cons_path) as f:
        rec = json.loads(f.readline())

    coin, verb, rollout = rec["coin"], rec["verb_prompt"], rec["rollout_prompt"]
    if coin not in verb or coin not in rollout:
        raise RuntimeError(f"{coin} missing from its own prompts in {cons_path}")

    # The consistency data originally carried one rollout paraphrase that
    # misspelled "separated" (record 0 held it); the shipped JSONLs have been
    # corrected to match the SFT corpora (see DIVERGENCES.md), so the record
    # templates verbatim.

    return verb.replace(coin, _TEMPLATE_COIN), rollout.replace(coin, _TEMPLATE_COIN)


def _build_metrics(records):
    """Per-group R^2 of the stated bias against each reference, plus coverage."""
    def r2(pred, target):
        pairs = [(p, t) for p, t in zip(pred, target) if p is not None and t is not None]
        if len(pairs) < 2:
            return float("nan")
        pred, target = zip(*pairs)
        mean_t = sum(target) / len(target)
        ss_res = sum((p - t) ** 2 for p, t in zip(pred, target))
        ss_tot = sum((t - mean_t) ** 2 for t in target)
        return 1 - ss_res / ss_tot if ss_tot else float("nan")

    metrics = {}
    for group in ("sft", "cons", "holdout", None):
        rows = [r for r in records if group is None or r["group"] == group]
        rows = [r for r in rows if r["pred_bias"] is not None]
        key = group or "all"
        if not rows:
            continue
        stated_mode = [r["pred_mode"] for r in rows]
        stated_mean = [r["pred_mean"] for r in rows]
        gt = [r["gt_bias"] for r in rows]
        rollout = [r["rollout_bias"] for r in rows]
        # Unsuffixed keys kept for backward compat; they are the mode-of-K
        # numbers, same as r2_*_mode_*.
        metrics[f"r2_gt_{key}"] = r2(stated_mode, gt)
        metrics[f"r2_rollout_{key}"] = r2(stated_mode, rollout)
        metrics[f"r2_gt_mode_{key}"] = r2(stated_mode, gt)
        metrics[f"r2_gt_mean_{key}"] = r2(stated_mean, gt)
        metrics[f"r2_rollout_mode_{key}"] = r2(stated_mode, rollout)
        metrics[f"r2_rollout_mean_{key}"] = r2(stated_mean, rollout)
        metrics[f"n_{key}"] = len(rows)

    unparsed = sum(1 for r in records if not r["pred_bias_samples"])
    metrics["n_coins"] = len(records)
    metrics["n_no_parseable_program"] = unparsed
    return metrics


def run_per_coin(
    model_name=BASE_MODEL,
    lora_path=None,
    output_dir=None,
    n_programs=DEFAULT_N_PROGRAMS,
    n_flips=DEFAULT_N_FLIPS,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    temp=DEFAULT_TEMP,
    top_p=DEFAULT_TOP_P,
    seed=42,
    gen_batch_size=8,
):
    """Sample programs + a rollout for every coin. Returns the payload dict;
    writes per_coin_debug.json under output_dir if given."""
    group_of, bias_of = coin_groups()
    verb_tmpl, rollout_tmpl = prompt_templates()

    coins = sorted(group_of)
    verb_prompts = [verb_tmpl.replace(_TEMPLATE_COIN, c) for c in coins]
    rollout_prompts = [rollout_tmpl.replace(_TEMPLATE_COIN, c) for c in coins]

    print(f"Loading {model_name} (adapter={lora_path or 'none — base model'})...")
    model, tokenizer = load_model_tok(
        model_name, mode="full", adapter_path=lora_path, is_trainable=False
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    with torch.no_grad():
        print(f"Sampling {n_programs} programs for each of {len(coins)} coins...")
        programs, _ = sample_k_conts_gen(
            model,
            tokenizer,
            prompts=verb_prompts,
            k=n_programs,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_p=top_p,
            greedily=False,
            batch_size=gen_batch_size,
            seed=seed,
        )
        print(f"Sampling one {n_flips}-flip rollout per coin...")
        rollouts, _, _ = sample_conts_gen(
            model,
            tokenizer,
            rollout_prompts,
            max_new_tokens=n_flips * 2,
            # The empirical rate has to reflect the policy's own distribution,
            # so no temperature or nucleus truncation here.
            temperature=1.0,
            top_p=1.0,
            greedily=False,
        )

    records = []
    for coin, candidates, rollout in zip(coins, programs, rollouts):
        try:
            biases, mask = program_biases(candidates, coin, seed=seed)
            samples = [b for b, ok in zip(biases, mask.tolist()) if ok]
        except RuntimeError:
            samples = []
        h, t = rollout.count("H"), rollout.count("T")
        records.append(
            {
                "coin": coin,
                "group": group_of[coin],
                "gt_bias": bias_of[coin],
                "pred_bias_samples": samples,
                # The mode over the K samples, which is what plot_calibration
                # plots. Taking samples[0] instead would make the metrics printed
                # here a different (much noisier) statistic from the figure's.
                # Kept for backward compat; identical to pred_mode.
                "pred_bias": mode_of_samples(samples) if samples else None,
                # Both estimators, explicitly named. The figure plots the mode;
                # the mean is reported alongside everywhere R^2 is computed.
                "pred_mode": mode_of_samples(samples) if samples else None,
                "pred_mean": mean_of_samples(samples) if samples else None,
                "rollout_h": h,
                "rollout_n": h + t,
                "rollout_bias": (h / (h + t)) if (h + t) else None,
            }
        )

    metrics = _build_metrics(records)
    payload = {
        "base_model_name": model_name,
        "lora_path": lora_path,
        "n_programs": n_programs,
        "n_flips": n_flips,
        # Sampling settings are part of the result: the same checkpoint scores
        # very differently at different program temperatures, and two runs land
        # on the same output path.
        "temp": temp,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "metrics": metrics,
        "records": records,
    }

    print(f"  coins={metrics['n_coins']}  "
          f"no parseable program={metrics['n_no_parseable_program']}")
    for group in ("sft", "cons", "holdout"):
        if f"r2_gt_mode_{group}" in metrics:
            print(f"  {group:>8}: "
                  f"R2(stated|gt) mode={metrics[f'r2_gt_mode_{group}']:.4f} "
                  f"mean={metrics[f'r2_gt_mean_{group}']:.4f}  "
                  f"R2(stated|rollout) mode={metrics[f'r2_rollout_mode_{group}']:.4f} "
                  f"mean={metrics[f'r2_rollout_mean_{group}']:.4f}  "
                  f"n={metrics[f'n_{group}']}")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "per_coin_debug.json", "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  -> {out / 'per_coin_debug.json'}")

    return payload


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name", default=BASE_MODEL)
    p.add_argument("--lora_path", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--n_programs", type=int, default=DEFAULT_N_PROGRAMS)
    p.add_argument("--n_flips", type=int, default=DEFAULT_N_FLIPS)
    # Program-sampling temperature. The original ran all three figure columns at
    # 1.0 — its `coin_sft_eval_*.yaml` say 0.3, but every oracle run overrode it
    # on the command line (see its outputs/*/.hydra/overrides.yaml). 1.0 is the
    # value to reproduce; this flag exists so the choice is explicit and logged.
    p.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    p.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gen_batch_size", type=int, default=8)
    args = p.parse_args()

    run_per_coin(
        model_name=args.model_name,
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        n_programs=args.n_programs,
        n_flips=args.n_flips,
        temp=args.temp,
        top_p=args.top_p,
        seed=args.seed,
        gen_batch_size=args.gen_batch_size,
    )
