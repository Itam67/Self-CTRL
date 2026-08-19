"""Consistency training for the probabilistic-reasoning (coins) domain."""

import sys
from pathlib import Path

# Allow running this file directly: Python only puts this file's dir on
# sys.path, so add the repo root to resolve consistency.*/domains.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import math
import os
from typing import List, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from consistency.model_utils import (
    compute_nll,
    cont_training_loss,
    derive_seed,
    load_model_tok,
    sample_conts_gen,
    sample_k_conts_gen,
)
from consistency.trainer import ConsistencyConfig, ConsistencyTrainer
from domains.coins.data import (
    CoinsConsDataset,
    ensure_behavior_anchors,
    read_cons_dataset,
)
from domains.coins.program import coin_from_prompt, empirical_bias, program_biases


def _set_seed(seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _r2(pred: List[float], target: List[float]) -> float:
    """R^2 of predictions against the identity line, not against a fitted line.

    Penalises reports that correlate with the target but are miscalibrated in
    scale or offset. Normalised by the variance of the target, so 0 means "no
    better than predicting the mean bias for every coin" and negative means
    worse than that.
    """
    if not target:
        return 0.0
    mean_t = sum(target) / len(target)
    ss_res = sum((p - t) ** 2 for p, t in zip(pred, target))
    ss_tot = sum((t - mean_t) ** 2 for t in target)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


def load_data(cfg) -> tuple:
    """DataLoaders over (verb_prompt, rollout_prompt) pairs.

    The trainer reads a batch as (explanation_prompts, behavior_prompts), so the
    verbalization prompt is the explanation side and the rollout prompt is the
    behavior side.
    """
    train_verb, train_rollouts, coin_names, gt_biases = read_cons_dataset(
        Path(cfg.data.cons_dir) / "cons_train.jsonl"
    )
    val_verb, val_rollouts, val_names, val_biases = read_cons_dataset(
        Path(cfg.data.cons_dir) / "cons_val.jsonl"
    )

    g = _set_seed(int(cfg.seed) if getattr(cfg, "seed", None) is not None else None)
    device_is_cuda = torch.cuda.is_available()

    train_ds = CoinsConsDataset(train_verb, train_rollouts)
    val_ds = CoinsConsDataset(val_verb, val_rollouts)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.learning.batch_size),
        shuffle=True,
        pin_memory=device_is_cuda,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.learning.batch_size),
        shuffle=False,
        pin_memory=device_is_cuda,
    )

    extra = {
        "name_2_bias": {
            **dict(zip(coin_names, gt_biases)),
            **dict(zip(val_names, val_biases)),
        },
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "train_rollout_prompts": train_rollouts,
        "train_dataset": train_ds,
    }
    return train_loader, val_loader, extra


def _check_all_fail_rate(extra, threshold: float = 0.05, min_fails: int = 3) -> None:
    """Abort once all-fail examples exceed `threshold` of examples seen so far.

    Running threshold over the whole run, not per epoch. min_fails is a grace
    count so a single early failure (1/1 = 100%) doesn't kill a healthy run
    before the denominator is meaningful.
    """
    if extra is None:
        return
    fails = int(extra.get("all_fail_examples", 0))
    seen = int(extra.get("parse_examples_seen", 0))
    if fails >= min_fails and fails > threshold * seen:
        raise RuntimeError(
            f"{fails}/{seen} training examples so far had NO parseable candidate "
            f"program (> {threshold:.0%}). The policy has likely collapsed away "
            "from the program format; aborting instead of training on noise."
        )


def collect_explanations(
    model, tokenizer, explanation_prompts: List[str], cfg, extra=None
) -> tuple:
    """Sample K candidate programs per coin and reduce each to a stated bias.

    Returns:
        biases:      List[List[float]] [B][K]  — the "explanations" the reward sees
        nlls:        torch.Tensor [B, K]       — gradients flow through these
        valid_mask:  torch.BoolTensor [B, K]   — False where a program didn't parse
        extra:       None

    Note the asymmetry with the moral domain: an explanation here is scored as a
    NUMBER, but the policy is updated through the NLL of the PROGRAM TEXT that
    produced it. Candidates whose text doesn't parse keep the -1.0 placeholder
    bias, so the [B, K] shape the trainer asserts on stays rectangular; whether
    they are penalized in-group or masked out of the update is decided by
    cfg.learning.invalid_candidate_reward (see below). An example where ALL K
    candidates fail is masked out entirely and counted, never fatal.
    """
    coin_names = [coin_from_prompt(p) for p in explanation_prompts]

    # Fresh sampling noise each call: reusing one fixed seed here would replay
    # the identical RNG stream at every training step, correlated dice the
    # policy can adapt to. The counter lives in the trainer's `extra`, so it
    # keeps incrementing across micro-batches, batches, and epochs.
    step = int(extra.get("programs_sampling_step", 0)) if extra is not None else 0
    if extra is not None:
        extra["programs_sampling_step"] = step + 1
    seed = (
        derive_seed(int(cfg.seed), "programs", step)
        if getattr(cfg, "seed", None) is not None
        else None
    )

    programs, _ = sample_k_conts_gen(
        model,
        tokenizer,
        prompts=explanation_prompts,
        k=cfg.sampling.n_beams,
        max_new_tokens=cfg.sampling.max_new_tokens,
        temperature=cfg.sampling.temp,
        top_p=cfg.sampling.top_p,
        greedily=False,
        batch_size=cfg.sampling.num_cands,
        seed=seed,
    )

    # Penalty mode (GRPO-standard, the default): a number here keeps candidates
    # that don't parse IN the group with this fixed low reward (assigned in
    # phi), so their group-relative advantage is negative and REINFORCE
    # actively pushes probability away from them. null restores the legacy
    # behavior of masking them out of the update (zero gradient).
    invalid_reward = getattr(cfg.learning, "invalid_candidate_reward", None)

    biases, masks = [], []
    n_parsed = 0
    n_candidates = 0
    for i, candidates in enumerate(programs):
        n_candidates += len(candidates)
        try:
            coin_biases, mask = program_biases(
                candidates, coin_names[i], seed=int(cfg.seed)
            )
            n_parsed += int(mask.sum())
            if invalid_reward is not None:
                # Keep every candidate in the advantage normalization; the
                # -1.0 placeholder bias is what phi keys the penalty on.
                mask = torch.ones(len(candidates), dtype=torch.bool)
        except RuntimeError:
            # Every candidate failed to parse. The eval path catches this and
            # skips the coin; during training it must not kill the run, so
            # mask all K candidates out (the trainer skips an all-masked
            # example's contribution). This holds in penalty mode too: uniform
            # penalty rewards make the group-relative advantage degenerate, so
            # there is nothing to learn from either way.
            joined = " | ".join(repr(c) for c in candidates)
            print(
                f"  [train] ALL {len(candidates)} candidate programs for "
                f"{coin_names[i]} unparseable — skipping example. "
                f"Candidates: {joined}"
            )
            coin_biases = [-1.0] * len(candidates)
            mask = torch.zeros(len(candidates), dtype=torch.bool)
            if extra is not None:
                extra["all_fail_examples"] = (
                    int(extra.get("all_fail_examples", 0)) + 1
                )
        biases.append(coin_biases)
        masks.append(mask)
        if extra is not None:
            extra["parse_examples_seen"] = (
                int(extra.get("parse_examples_seen", 0)) + 1
            )

    if extra is not None:
        # Running parse-health counters, queued for the trainer's per-batch
        # wandb payload (see the log_metrics merge in ConsistencyTrainer).
        extra["parse_valid_candidates"] = (
            int(extra.get("parse_valid_candidates", 0)) + n_parsed
        )
        extra["parse_total_candidates"] = (
            int(extra.get("parse_total_candidates", 0)) + n_candidates
        )
        rate = extra["parse_valid_candidates"] / max(
            extra["parse_total_candidates"], 1
        )
        log_metrics = extra.setdefault("log_metrics", {})
        log_metrics["train/program_parse_rate"] = rate
        log_metrics["train/invalid_candidate_rate"] = 1.0 - rate
        log_metrics["train/all_fail_examples"] = int(
            extra.get("all_fail_examples", 0)
        )
        if invalid_reward is not None:
            # Constant by construction (every invalid candidate gets the fixed
            # penalty), logged so the run records penalty mode was active.
            log_metrics["train/invalid_candidate_reward_mean"] = float(
                invalid_reward
            )

    _check_all_fail_rate(extra)

    nlls = compute_nll(
        model,
        tokenizer,
        explanation_prompts,
        programs,
        device=next(model.parameters()).device,
    )

    # The trainer clones these NLLs and masked_fill_s them with this mask, so
    # the two must share a device.
    valid_mask = torch.stack(masks, dim=0).to(nlls.device)

    return biases, nlls, valid_mask, None


def collect_behaviors(
    model, tokenizer, behavior_prompts: List[str], cfg, extra=None
) -> tuple:
    """Sample K rollouts per coin. Shapes mirror collect_explanations: the
    trainer detects the [B][K] form and anchors the explanation update on the
    highest-likelihood rollout."""
    # Per-step derived seed for the same reason as in collect_explanations.
    step = int(extra.get("behaviors_sampling_step", 0)) if extra is not None else 0
    if extra is not None:
        extra["behaviors_sampling_step"] = step + 1
    seed = (
        derive_seed(int(cfg.seed), "behaviors", step)
        if getattr(cfg, "seed", None) is not None
        else None
    )

    behaviors, _ = sample_k_conts_gen(
        model,
        tokenizer,
        prompts=behavior_prompts,
        k=cfg.sampling.n_beams,
        # Rollouts get their own budget: the prompts ask for 100 flips, which
        # is ~200 tokens ("H " per flip), so the 150-token program cap would
        # silently truncate every rollout to ~75 flips.
        max_new_tokens=int(getattr(cfg.sampling, "rollout_max_new_tokens", 200)),
        temperature=cfg.sampling.temp,
        top_p=cfg.sampling.top_p,
        greedily=False,
        batch_size=cfg.sampling.num_cands,
        seed=seed,
    )

    behavior_nlls = compute_nll(
        model,
        tokenizer,
        behavior_prompts,
        behaviors,
        device=next(model.parameters()).device,
    )

    return behaviors, behavior_nlls, {"behavior_prompts": behavior_prompts}


def phi(behaviors, explanations, model, tokenizer, cfg, extra=None) -> torch.Tensor:
    """Per-flip log likelihood of a rollout under the bias its own program states.

    behaviors:    List[str] [B]            OR List[List[str]] [B][Kb]
    explanations: List[List[float]] [B][Ke]

    A singleton side is broadcast to match the other, mirroring the moral
    domain's phi. Rewards are log likelihoods, so they are <= 0 and higher is
    better; GRPO normalises them per group, so the scale is irrelevant.

    Returns torch.Tensor [B, K].
    """
    eps = 1e-6

    # See collect_explanations: in penalty mode an unparseable candidate stays
    # in the group and earns this fixed reward. A valid program's reward is a
    # per-flip log likelihood bounded below by log(eps) ~= -13.8, so the
    # configured penalty must sit strictly below that to guarantee a negative
    # group-relative advantage.
    invalid_reward = getattr(cfg.learning, "invalid_candidate_reward", None)

    if isinstance(behaviors[0], str):
        behaviors = [[b] for b in behaviors]

    rewards = []
    for i in range(len(behaviors)):
        beh_i, expl_i = behaviors[i], explanations[i]
        Kb, Ke = len(beh_i), len(expl_i)
        if Kb == 1 and Ke > 1:
            beh_i, K = beh_i * Ke, Ke
        elif Ke == 1 and Kb > 1:
            expl_i, K = expl_i * Kb, Kb
        else:
            assert Kb == Ke, f"shape mismatch at i={i}: Kb={Kb}, Ke={Ke}"
            K = Kb

        row = []
        for k in range(K):
            stated = float(expl_i[k])
            if invalid_reward is not None and stated < 0.0:
                # Unparseable candidate (placeholder bias -1.0) kept in the
                # group: fixed penalty, independent of the rollout.
                row.append(float(invalid_reward))
                continue
            heads = beh_i[k].count("H")
            tails = beh_i[k].count("T")
            n = heads + tails
            if n == 0:
                # Nothing was sampled that looks like a flip; no signal either way.
                row.append(0.0)
                continue
            # In legacy masking mode invalid programs carry a placeholder bias
            # and are masked out by the trainer, but the reward is still
            # computed for shape, so clamp into the open interval to keep the
            # log finite.
            p = min(max(stated, eps), 1 - eps)
            row.append((heads * math.log(p) + tails * math.log(1 - p)) / n)
        rewards.append(row)

    K0 = len(rewards[0]) if rewards else 0
    assert all(
        len(r) == K0 for r in rewards
    ), f"ragged rewards: {[len(r) for r in rewards]}"

    return torch.tensor(rewards, device=extra["device"], dtype=torch.float32)


def ht_ratio(tokenizer, scores, valid_mask=None):
    """P(H) / (P(H) + P(T)) per generated step, from the logits of the two tokens.

    Measures the rollout distribution directly rather than through sampled
    counts, so it is far lower variance than the empirical head rate. Returns
    (mean, variance) per batch element when a mask is given.
    """
    h_ids = tokenizer.encode(" H", add_special_tokens=False)
    t_ids = tokenizer.encode(" T", add_special_tokens=False)
    if len(h_ids) != 1 or len(t_ids) != 1:
        raise ValueError(f"' H'/' T' are not single tokens: {h_ids}, {t_ids}")

    two = torch.stack([scores[:, :, h_ids[0]], scores[:, :, t_ids[0]]], dim=-1)
    probs = torch.softmax(two, dim=-1)
    ratio = probs[:, :, 0] / (probs[:, :, 0] + probs[:, :, 1] + 1e-12)  # [B, T]

    if valid_mask is None:
        return ratio.mean(dim=0), ratio.std(dim=0, unbiased=False)

    vm = valid_mask.to(dtype=ratio.dtype)
    denom = vm.sum(dim=1).clamp_min(1.0)
    mean = (ratio * vm).sum(dim=1) / denom
    var = (((ratio - mean[:, None]) * vm) ** 2).sum(dim=1) / denom
    return mean, var


def eval_fn(model, tokenizer, val_loader, device, cfg, extra):
    """Calibration and self-consistency over the validation coins.

    Three R^2 values, all against the identity line:
        r2_gt_pred    articulated bias vs the coin's true latent bias
        r2_cont_pred  articulated bias vs the model's own empirical rollout rate
        r2_cont_gt    rollout rate vs true bias — how well the model still
                      *behaves* like the coin, i.e. whether the thing being
                      described has drifted
    """
    model.eval()
    coin_names, pred_biases, gt_biases, cont_biases = [], [], [], []
    ratio_diffs, ratio_vars = [], []

    with torch.no_grad():
        for verb_batch, rollout_batch in val_loader:
            verb_batch, rollout_batch = list(verb_batch), list(rollout_batch)
            names = [coin_from_prompt(p) for p in rollout_batch]

            # Behavior side: one rollout per coin, untruncated sampling so the
            # empirical rate reflects the policy's actual distribution. Uses
            # the rollout budget (2 x 100 flips), not the program cap.
            conts, _, gen_extra = sample_conts_gen(
                model,
                tokenizer,
                rollout_batch,
                max_new_tokens=int(
                    getattr(cfg.sampling, "rollout_max_new_tokens", 200)
                ),
                temperature=1.0,
                top_p=1.0,
                greedily=False,
                return_logits=True,
            )
            ratio_mean, ratio_var = ht_ratio(
                tokenizer, gen_extra["logits"], gen_extra["valid_mask"]
            )

            # Explanation side: sample programs and take the first that parsed.
            # Deliberately step-independent seed: every eval point draws the
            # same sampling noise, so checkpoints are compared paired.
            programs, _ = sample_k_conts_gen(
                model,
                tokenizer,
                prompts=verb_batch,
                k=cfg.sampling.n_beams,
                max_new_tokens=cfg.sampling.max_new_tokens,
                temperature=cfg.sampling.temp,
                top_p=cfg.sampling.top_p,
                greedily=False,
                batch_size=cfg.sampling.num_cands,
                seed=(
                    derive_seed(int(cfg.seed), "eval")
                    if getattr(cfg, "seed", None) is not None
                    else None
                ),
            )

            for i, name in enumerate(names):
                try:
                    biases, mask = program_biases(programs[i], name, seed=int(cfg.seed))
                except RuntimeError:
                    # Every candidate unparseable — the model has no articulable
                    # bias for this coin, so it contributes nothing to R^2.
                    print(f"  [eval] {name}: no parseable program, skipped")
                    continue
                stated = next(b for b, ok in zip(biases, mask.tolist()) if ok)

                gt = extra["name_2_bias"][name]
                emp = empirical_bias(conts[i])
                if emp is None:
                    print(f"  [eval] {name}: rollout had no flips, skipped")
                    continue

                coin_names.append(name)
                pred_biases.append(round(stated, 3))
                gt_biases.append(gt)
                cont_biases.append(round(emp, 3))
                ratio_diffs.append(abs(ratio_mean[i].item() - gt))
                ratio_vars.append(ratio_var[i].item())

    metrics = {
        "r2_gt_pred": _r2(pred_biases, gt_biases),
        "r2_cont_pred": _r2(pred_biases, cont_biases),
        "r2_cont_gt": _r2(cont_biases, gt_biases),
        "abs_ht_ratio_minus_gt": float(np.mean(ratio_diffs)) if ratio_diffs else 0.0,
        "ht_ratio_var": float(np.mean(ratio_vars)) if ratio_vars else 0.0,
        "n_coins": len(coin_names),
    }

    print("-" * 30 + " coins eval " + "-" * 30)
    for name, gt, pred, cont in zip(coin_names, gt_biases, pred_biases, cont_biases):
        print(f"  {name:>10}  gt={gt:.3f}  stated={pred:.3f}  rollout={cont:.3f}")
    print(
        f"  R2(stated|gt)={metrics['r2_gt_pred']:.4f}  "
        f"R2(stated|rollout)={metrics['r2_cont_pred']:.4f}  "
        f"R2(rollout|gt)={metrics['r2_cont_gt']:.4f}"
    )

    _write_snapshot(
        cfg, extra, coin_names, gt_biases, pred_biases, cont_biases, metrics
    )
    model.train()
    return metrics, extra


def _write_snapshot(cfg, extra, names, gt, pred, cont, metrics) -> None:
    """Per-coin record for the checkpoint, alongside the trainer's other output."""
    results_dir = Path(getattr(cfg.learning, "results_dir", cfg.learning.save_dir))
    snap_dir = results_dir / "eval_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    step = int(extra.get("examples_seen", 0))
    payload = {
        "examples_seen": step,
        "metrics": metrics,
        "records": [
            {"coin": c, "gt_bias": g, "stated_bias": p, "rollout_bias": r}
            for c, g, p, r in zip(names, gt, pred, cont)
        ],
    }
    with open(snap_dir / f"coins_eval_{step}.json", "w") as f:
        json.dump(payload, f, indent=2)


def verbose_logging_coins(
    behavior_prompts,
    explanation_prompts,
    behaviors,
    explanations,
    behavior_advantages,
    explanation_advantages,
    explanation_rewards,
    extra,
    behavior_rewards=None,
    **kwargs,
):
    """Print each coin's candidate biases against its true bias."""
    print("\n=== coins: candidates / advantages ===")
    for i in range(len(behavior_prompts)):
        coin = coin_from_prompt(behavior_prompts[i])
        gt = extra["name_2_bias"].get(coin)
        print(f"\n  {coin}  gt_bias={gt}")
        if explanation_advantages is None:
            continue
        for k in range(len(explanations[i])):
            stated = explanations[i][k]
            adv = explanation_advantages[i, k].item()
            flag = "" if stated >= 0 else "   (unparseable)"
            print(f"    ({i},{k}) stated={stated:<7} advantage={adv: .4f}{flag}")
    print("=== end ===\n")


@hydra.main(
    config_path="../../configs",
    config_name="coins",
    version_base=None,
)
def main(cfg: DictConfig) -> None:

    # Config-driven determinism (default on)
    if bool(getattr(cfg, "deterministic", True)):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_dir = getattr(cfg, "load_dir", None) or None
    if load_dir is None:
        raise ValueError(
            "coins consistency training starts from the SFT adapter — set "
            "load_dir to a checkpoint from domains/coins/sft.py."
        )

    model, tokenizer = load_model_tok(
        cfg.model_name, cfg.mode, adapter_path=load_dir, is_trainable=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"

    trainable_params = (p for p in model.parameters() if p.requires_grad)
    optim = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.learning.lr),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=torch.cuda.is_available(),
    )

    cons_cfg = ConsistencyConfig(
        load_data=load_data,
        collect_behaviors=collect_behaviors,
        collect_explanations=collect_explanations,
        reward_fn=phi,
        cont_training_loss_fn=None,  # set below once the anchors exist
        eval_fn=eval_fn,
        verbose_logging=verbose_logging_coins,
    )

    trainer = ConsistencyTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optim,
        device=device,
        train_cfg=cfg,
        cons_cfg=cons_cfg,
    )

    # Behavior-preserving anchors
    if float(cfg.learning.cont_training_loss_weight) > 0:
        results_dir = Path(getattr(cfg.learning, "results_dir", cfg.learning.save_dir))
        anchors = ensure_behavior_anchors(
            model,
            tokenizer,
            trainer.extra["train_rollout_prompts"],
            cache_path=results_dir / "behavior_anchors.json",
            # Anchors are rollouts, so they take the rollout budget (2 x 100
            # flips); the 150-token program cap would truncate them.
            max_new_tokens=int(getattr(cfg.sampling, "rollout_max_new_tokens", 200)),
            seed=int(cfg.seed) if getattr(cfg, "seed", None) is not None else None,
        )
        trainer.extra["train_dataset"].cont_training_data = anchors
        trainer.cons_cfg.cont_training_loss_fn = cont_training_loss

    trainer.train()


if __name__ == "__main__":
    main()
