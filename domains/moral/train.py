import sys
from pathlib import Path

# Allow running this file directly (`python domains/moral/train.py`): Python only puts
# this file's dir on sys.path, so add the repo root to resolve consistency.*/domains.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import hydra
import torch
import numpy as np
from omegaconf import DictConfig
from typing import List, Optional
from torch.utils.data import DataLoader
import wandb
from contextlib import nullcontext

from consistency.trainer import ConsistencyConfig, ConsistencyTrainer
from consistency.model_utils import (
    load_model_tok,
    sample_conts_gen,
    sample_k_conts_gen,
    compute_nll,
    cont_training_loss,
)
from consistency.data_utils import load_cont_training_data
from domains.moral.data import (
    make_spec_eval_dataset,
    category_paraphrase,
    holdout_type,
)
from domains.moral.judges import (
    ETHICAL_FRAMEWORKS,
    CLASSIFICATION_TEMPLATE,
    ENGAGEMENT_TEMPLATE,
    MORAL_SYSTEM_PROMPT,
    soft_score_compliance,
)


### Helper functions ###
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


### Moral-specific implementations ###
def load_data_spec_eval(cfg) -> tuple:
    """
    Loads the moral evaluation dataset and returns DataLoaders for training and validation.

    Returns:
        train_loader: DataLoader for the training dataset
        val_loader: DataLoader for the validation dataset
        extra: dict containing additional information such as labels and principles
    """

    # Load continual training data if specified in the configuration
    cont_training_data = None
    if getattr(cfg, "cont_training", None) is not None:
        cont_training_data = load_cont_training_data(cfg, seed=int(cfg.seed))

    # Create consistency datasets for training and validation
    train_ds = make_spec_eval_dataset("train", cfg)
    val_ds = make_spec_eval_dataset("val", cfg, max_samples=100)

    # Add continual training data to the training dataset if available
    if cont_training_data is not None:
        train_ds.cont_training_data = cont_training_data

    g = _set_seed(int(cfg.seed) if getattr(cfg, "seed", None) is not None else None)
    device_is_cuda = torch.cuda.is_available()

    # Create DataLoaders for training and validation datasets
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

    # Prepare additional information for logging and evaluation
    train_categories = [category_paraphrase(p) for p in train_ds.principle]
    val_categories = [category_paraphrase(p) for p in val_ds.principle]
    extra = {
        # Primary label (category paraphrase) — what logging/eval reads.
        "train/labels": train_categories,
        "val/labels": val_categories,
        # Principle slug, kept alongside for analysis: labels are category-level,
        # so this is the only per-record trace of which principle it came from.
        "train/labels_principle": list(train_ds.principle),
        "val/labels_principle": list(val_ds.principle),
        "prompt_to_principle": {
            **dict(zip(train_ds.behavior_prompts, train_categories)),
            **dict(zip(val_ds.behavior_prompts, val_categories)),
        },
    }

    return train_loader, val_loader, extra


### Sample collection from policy ###
def collect_behaviors(model, tokenizer, moral_prompts: List[str], cfg, extra) -> tuple:
    """
    Collects K candidate behaviors for each moral prompt using the model.

    Returns:
        behaviors: List[List[str]]  shape [B][K]
        judgement_nlls: torch.Tensor [B, K]
    """

    # Generate K candidates per prompt (mirrors collect_moral_explanations)
    behaviors, _ = sample_k_conts_gen(
        model,
        tokenizer,
        prompts=moral_prompts,
        k=cfg.sampling.n_beams,
        max_new_tokens=cfg.sampling.max_new_tokens,
        temperature=cfg.sampling.temp,
        top_p=cfg.sampling.top_p,
        greedily=False,
        batch_size=cfg.sampling.num_cands,
        seed=cfg.seed,
    )

    # Recompute nll of candidates [B, K]
    behavior_nlls = compute_nll(
        model,
        tokenizer,
        moral_prompts,
        behaviors,
        device=next(model.parameters()).device,
    )

    extra["behavior_prompts"] = moral_prompts

    return behaviors, behavior_nlls, extra


def collect_explanations(
    model, tokenizer, explanation_prompts: List[str], cfg, extra=None
) -> tuple:
    """
    Collects K candidate explanations for each explanation prompt using the model.

    Returns:
        explanations: List[List[str]]  shape [B][K]
        explanations_nlls: torch.Tensor [B, K]
    """

    B = len(explanation_prompts)
    system_prompts = [MORAL_SYSTEM_PROMPT] * B

    # Generate K candidates per prompt (mirrors collect_behaviors)
    explanations, _ = sample_k_conts_gen(
        model,
        tokenizer,
        prompts=explanation_prompts,
        k=cfg.sampling.n_beams,
        max_new_tokens=cfg.sampling.max_new_tokens,
        temperature=cfg.sampling.temp,
        top_p=cfg.sampling.top_p,
        greedily=False,
        batch_size=cfg.sampling.num_cands,
        seed=cfg.seed,
        system_prompts=system_prompts,
    )

    # Recompute nll of candidates [B, K]
    explanations_nlls = compute_nll(
        model,
        tokenizer,
        explanation_prompts,
        explanations,
        device=next(model.parameters()).device,
        system_prompt=MORAL_SYSTEM_PROMPT,
    )

    return explanations, explanations_nlls, None, None


def phi_moral_jury_soft(behaviors, explanations, model, tokenizer, cfg, extra):
    """
    Soft jury-based consistency reward: same as phi_moral_jury but uses
    P(yes) / (P(yes) + P(no)) from logits instead of hard yes/no decisions.
    Scans up to 8 generated tokens and picks the step with highest Yes/No coverage.

    Batches all B*K pairs across all n_judges in a single call to
    soft_score_compliance (which handles sub-batching internally).

    Returns:
      rewards: torch.Tensor [B, K]  (higher is better, range [0, 1])
    """

    behavior_prompts = extra["behavior_prompts"]
    device = next(model.parameters()).device
    B = len(behavior_prompts)
    assert len(behaviors) == B

    # Normalize both to List[List[str]]
    if isinstance(behaviors[0], str):
        behaviors = [[b] for b in behaviors]
    if isinstance(explanations[0], str):
        explanations = [[e] for e in explanations]

    # Broadcast to matching lengths
    for i in range(B):
        Ke, Kb = len(explanations[i]), len(behaviors[i])
        if Ke == 1 and Kb > 1:
            explanations[i] = explanations[i] * Kb
        elif Kb == 1 and Ke > 1:
            behaviors[i] = behaviors[i] * Ke

    if getattr(cfg, "single_judge", False):
        judge_system_prompts = [None]
    else:
        judge_system_prompts = [fw_text for _, fw_text in ETHICAL_FRAMEWORKS]
    n_judges = len(judge_system_prompts)

    # Flatten all (i, k) pairs × n_judges into one big batch
    all_system_prompts = []
    all_user_prompts = []
    k_per_examples = []  # track K per example for reshaping
    for i in range(B):
        K = len(explanations[i])
        k_per_examples.append(K)
        for k in range(K):
            user_content = CLASSIFICATION_TEMPLATE.format(
                principle=explanations[i][k],
                prompt=behavior_prompts[i],
                response=behaviors[i][k],
            )
            all_system_prompts.extend(judge_system_prompts)
            all_user_prompts.extend([user_content] * n_judges)

    adapter_ctx = (
        model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    )

    torch.cuda.empty_cache()
    with torch.no_grad(), adapter_ctx:
        all_soft_scores = soft_score_compliance(
            model,
            tokenizer,
            system_prompts=all_system_prompts,
            user_prompts=all_user_prompts,
            device=device,
        )

    # Reshape: every n_judges consecutive scores belong to one (i, k) pair
    idx = 0
    rewards = []
    for i in range(B):
        row_rewards = []
        for k in range(k_per_examples[i]):
            pair_scores = all_soft_scores[idx : idx + n_judges]
            row_rewards.append(sum(pair_scores) / n_judges)
            idx += n_judges
        rewards.append(torch.tensor(row_rewards, device=device))

    return torch.stack(rewards)  # [B, K]


def engagement_juror(
    behaviors, explanations, model, tokenizer, cfg, extra
) -> torch.Tensor:
    """
    Single-judge engagement reward: for each behavior, score whether it engages
    with the request rather than unnecessarily refusing.

    behaviors:    List[List[str]] [B][K]
    explanations: List[List[str]] [B][K]  (broadcast to match behaviors)

    Returns:
      rewards: torch.Tensor [B, K]  (higher = more engaged, range [0, 1])
    """
    behavior_prompts = extra["behavior_prompts"]
    device = next(model.parameters()).device
    B = len(behavior_prompts)

    all_user_prompts = []
    k_per_examples = []
    for i in range(B):
        K = len(behaviors[i])
        k_per_examples.append(K)
        for k in range(K):
            all_user_prompts.append(
                ENGAGEMENT_TEMPLATE.format(
                    principle=explanations[i][k],
                    prompt=behavior_prompts[i],
                    response=behaviors[i][k],
                )
            )

    adapter_ctx = (
        model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    )

    torch.cuda.empty_cache()
    with torch.no_grad(), adapter_ctx:
        all_soft = soft_score_compliance(
            model,
            tokenizer,
            system_prompts=None,
            user_prompts=all_user_prompts,
            device=device,
        )

    idx = 0
    rewards = []
    for i in range(B):
        row = []
        for k in range(k_per_examples[i]):
            row.append(all_soft[idx])
            idx += 1
        rewards.append(torch.tensor(row, device=device))

    return torch.stack(rewards)  # [B, K]


def phi_moral_jury_multi_signal(behaviors, explanations, model, tokenizer, cfg, extra):
    """
    Combined multi-signal reward that auto-detects the update mode:

      Explanation update (behaviors=[B] strings, explanations=[B][K]):
        reward = (1-w) * consistency

      Behavior update (behaviors=[B][K] lists, explanations=[B][1]):
        reward = (1-w) * consistency + w * engagement

    The weight w is read from cfg.learning.auxiliary_weight (default 0 = pure consistency).

    Returns:
      rewards: torch.Tensor [B, K]
    """
    # Consistency (always computed)
    consistency = phi_moral_jury_soft(
        behaviors, explanations, model, tokenizer, cfg, extra
    )

    # The auxiliary (engagement) signal only applies to behavior updates.
    w = float(getattr(cfg.learning, "auxiliary_weight", 0.0))
    is_behavior_update = isinstance(behaviors[0], list)
    if w == 0.0 or not is_behavior_update:
        return consistency

    # Behavior update: behaviors=[B][K], explanations=[B][1] -> broadcast to K.
    norm_behaviors = [list(b) for b in behaviors]
    norm_explanations = [(e if isinstance(e, list) else [e]) for e in explanations]
    norm_explanations = [
        ex * len(b) if len(ex) == 1 else ex
        for ex, b in zip(norm_explanations, norm_behaviors)
    ]

    engagement = engagement_juror(
        norm_behaviors, norm_explanations, model, tokenizer, cfg, extra
    )
    combined = (1 - w) * consistency + w * engagement

    extra["beh_consistency_rewards"] = consistency.detach()
    extra["beh_engagement_rewards"] = engagement.detach()
    try:
        if wandb.run is not None:
            wandb.log(
                {
                    "train/consistency": consistency.mean().item(),
                    "train/engagement": engagement.mean().item(),
                    "train/combined": combined.mean().item(),
                },
                commit=False,
            )
    except Exception:
        pass

    return combined


### KL regularization ### **NOTE: Current version of this does not support full model training, only adapter training.**
def untouched_side_kl_loss(
    model, tokenizer, behavior_prompts, explanation_prompts, cfg, extra, device
):
    """
    Compute KL(P_base || P_curr) for the untouched side (behaviors or explanations)
    and return a scalar loss to be added to the main RL loss.

    Returns:
        R: torch.Tensor scalar
        parts: dict with 'kl_beh' and 'kl_expl' for logging
    """
    bw = float(cfg.learning.bw)
    behavior_prompts = list(behavior_prompts)
    explanation_prompts = list(explanation_prompts)

    # Check if the model has a method to disable adapters, and use it if available
    adapter_ctx = (
        model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    )

    base_behaviors = None
    base_explanations = None

    # Generate base behaviors and explanations using the model without adapter
    with adapter_ctx, torch.no_grad():
        if (1 - bw) > 0:
            base_behaviors, _, _ = sample_conts_gen(
                model,
                tokenizer,
                behavior_prompts,
                max_new_tokens=cfg.sampling.max_new_tokens,
                temperature=cfg.sampling.temp,
                top_p=cfg.sampling.top_p,
                greedily=False,
            )
        if bw > 0:
            base_explanations, _, _ = sample_conts_gen(
                model,
                tokenizer,
                explanation_prompts,
                max_new_tokens=cfg.sampling.max_new_tokens,
                temperature=cfg.sampling.temp,
                top_p=cfg.sampling.top_p,
                greedily=False,
                system_prompts=[MORAL_SYSTEM_PROMPT] * len(explanation_prompts),
            )

    kl_beh = torch.zeros((), device=device)
    kl_expl = torch.zeros((), device=device)

    # Compute NLLs for the base behaviors and explanations if they were generated
    if base_behaviors is not None:
        curr_beh_nll = compute_nll(
            model, tokenizer, behavior_prompts, base_behaviors, device=device
        )
        kl_beh = curr_beh_nll.mean()

    if base_explanations is not None:
        curr_expl_nll = compute_nll(
            model,
            tokenizer,
            explanation_prompts,
            base_explanations,
            device=device,
            system_prompt=MORAL_SYSTEM_PROMPT,
        )
        kl_expl = curr_expl_nll.mean()

    R = (1 - bw) * kl_beh + bw * kl_expl

    # Log the KL values for monitoring
    kls = {
        "kl_beh": float(kl_beh.detach().item()),
        "kl_expl": float(kl_expl.detach().item()),
    }
    return R, kls


def verbose_logging_moral(
    behavior_prompts,
    explanation_prompts,
    behaviors,
    explanations,
    behavior_advantages,
    explanation_advantages,
    explanation_rewards,
    extra,
    behavior_rewards,
    behavior_nlls=None,
    explanation_nlls=None,
    cont_training_prompts=None,
    cont_training_completions=None,
):
    """
    Print per-example behaviors and explanations with their rewards, advantages,
    and per-token log-probs. A side whose rewards are None (the bw=0/1 skip case)
    is reported as "not updated this step".
    """
    extra = extra or {}
    prompt_to_principle = extra.get("prompt_to_principle", {})
    beh_cons = extra.get("beh_consistency_rewards")
    beh_eng = extra.get("beh_engagement_rewards")

    def _signal(i, k):
        """Per-signal cons/engagement string for behavior k (multi-signal reward)."""
        if beh_cons is None or i >= beh_cons.shape[0] or k >= beh_cons.shape[1]:
            return ""
        s = f"  cons={beh_cons[i, k].item():.3f}"
        if beh_eng is not None and i < beh_eng.shape[0] and k < beh_eng.shape[1]:
            s += f" engage={beh_eng[i, k].item():.3f}"
        return s

    def _print_candidates(
        title, i, items, rewards, advantages, nlls, best_note, signal
    ):
        # rewards/advantages are None when this side wasn't updated (bw=0 or 1).
        if rewards is None or advantages is None:
            print(f"\n  {title}: (not updated this step)")
            return
        best_k = nlls[i].argmin().item() if nlls is not None else None
        print(f"\n  {title} ({len(items)}):")
        for k in range(len(items)):
            logp = -nlls[i, k].item() if nlls is not None else float("nan")
            sig = _signal(i, k) if signal else ""
            marker = best_note if k == best_k else ""
            print(
                f"    [{k}] r={rewards[i, k].item():.4f}  adv={advantages[i, k].item():+.3f}"
                f"  logp/tok={logp:+.3f}{sig}  {items[k]}{marker}"
            )

    k_behaviors = isinstance(behaviors[0], list)
    for i in range(len(behavior_prompts)):
        print("=" * 80)
        print(f"  Prompt: {behavior_prompts[i]}")
        print(
            f"  Request Category: {prompt_to_principle.get(behavior_prompts[i], '?')}"
        )

        if k_behaviors:
            _print_candidates(
                "Behaviors",
                i,
                behaviors[i],
                behavior_rewards,
                behavior_advantages,
                behavior_nlls,
                "  <-- best (grading explanations)",
                signal=True,
            )
        else:
            print(f"\n  Behavior: {behaviors[i]}")

        _print_candidates(
            "Explanations",
            i,
            explanations[i],
            explanation_rewards,
            explanation_advantages,
            explanation_nlls,
            "  <-- best (grading behaviors)",
            signal=False,
        )
        print()

    # ── Continued-training sample ──
    if cont_training_prompts and cont_training_completions:
        trunc = 200
        print("-" * 80)
        print("  Cont-training sample (first example, truncated):")
        print(f"    Prompt:     {str(cont_training_prompts[0])[:trunc]}")
        print(f"    Completion: {str(cont_training_completions[0])[:trunc]}")
        print()


def kl_eval(model, tokenizer, val_loader, device, cfg, extra):
    """
    Compute KL(P_base || P_curr) ≈ mean(NLL_curr - NLL_base) for both
    behaviors and explanations on the eval set.

    On the first call, generates reference texts from the base model and
    caches them (along with base NLLs) in extra. Subsequent calls reuse
    the cached references and only recompute current-model NLLs.

    Returns:
      metrics: dict with behavior_kl and explanation_kl
      extra: updated with cached reference data
    """
    model.eval()

    # Helper to compute NLLs for a batch of prompts and texts
    def _nll(prompts, texts, system_prompt=None):
        return compute_nll(
            model, tokenizer, prompts, texts, device, system_prompt=system_prompt
        ).squeeze(
            1
        )  # [N]

    # On first call, generate reference texts from base model and cache base NLLs
    if "kl_ref_behaviors" not in extra:
        print("Generating KL reference texts from base model...")
        ref_behaviors = []
        ref_explanations = []
        ref_behavior_prompts = []
        ref_explanation_prompts = []

        # Disable adapters if the model has that method, otherwise use a null context
        adapter_ctx = (
            model.disable_adapter()
            if hasattr(model, "disable_adapter")
            else nullcontext()
        )

        # Generate reference behaviors and explanations for the validation set
        with torch.no_grad(), adapter_ctx:
            for _, (explanation_batch, behavior_batch) in enumerate(val_loader):
                base_behaviors, _, _ = sample_conts_gen(
                    model,
                    tokenizer,
                    behavior_batch,
                    max_new_tokens=cfg.sampling.max_new_tokens,
                    temperature=cfg.sampling.temp,
                    top_p=cfg.sampling.top_p,
                    greedily=False,
                )
                expl_sys_prompts = (
                    [MORAL_SYSTEM_PROMPT] * len(explanation_batch)
                    if MORAL_SYSTEM_PROMPT
                    else None
                )
                base_explanations, _, _ = sample_conts_gen(
                    model,
                    tokenizer,
                    explanation_batch,
                    max_new_tokens=cfg.sampling.max_new_tokens,
                    temperature=cfg.sampling.temp,
                    top_p=cfg.sampling.top_p,
                    greedily=False,
                    system_prompts=expl_sys_prompts,
                )
                ref_behaviors.extend(base_behaviors)
                ref_explanations.extend(base_explanations)
                ref_behavior_prompts.extend(behavior_batch)
                ref_explanation_prompts.extend(explanation_batch)

            # Compute base NLLs on the reference texts
            base_behavior_nlls = _nll(ref_behavior_prompts, ref_behaviors)
            base_explanation_nlls = _nll(
                ref_explanation_prompts, ref_explanations, MORAL_SYSTEM_PROMPT
            )

        # Cache the reference texts and base NLLs in extra for future calls
        extra["kl_ref_behaviors"] = ref_behaviors
        extra["kl_ref_explanations"] = ref_explanations
        extra["kl_ref_behavior_prompts"] = ref_behavior_prompts
        extra["kl_ref_explanation_prompts"] = ref_explanation_prompts
        extra["kl_base_behavior_nlls"] = base_behavior_nlls
        extra["kl_base_explanation_nlls"] = base_explanation_nlls

    # Compute current model NLLs on cached reference texts
    with torch.no_grad():
        curr_behavior_nlls = _nll(
            extra["kl_ref_behavior_prompts"], extra["kl_ref_behaviors"]
        )
        curr_explanation_nlls = _nll(
            extra["kl_ref_explanation_prompts"],
            extra["kl_ref_explanations"],
            MORAL_SYSTEM_PROMPT,
        )

    behavior_kl = (curr_behavior_nlls - extra["kl_base_behavior_nlls"]).mean().item()
    explanation_kl = (
        (curr_explanation_nlls - extra["kl_base_explanation_nlls"]).mean().item()
    )

    return {"behavior_kl": behavior_kl, "explanation_kl": explanation_kl}, extra


def holdout_eval(model, tokenizer, val_loader, device, cfg, extra):
    """
    Evaluate on held-out principle types (unseen during training).
    Mirrors spec_eval_eval but uses the holdout split.
    The holdout DataLoader is built once and cached in extra.
    """
    model.eval()

    # Build holdout loader once, cache in extra
    if "holdout_loader" not in extra:
        holdout_ds = make_spec_eval_dataset("holdout", cfg)
        extra["holdout_loader"] = DataLoader(
            holdout_ds,
            batch_size=int(cfg.learning.batch_size),
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        extra["holdout/labels"] = [category_paraphrase(p) for p in holdout_ds.principle]
        extra["holdout/labels_principle"] = list(holdout_ds.principle)
        extra["holdout/holdout_types"] = [holdout_type(p) for p in holdout_ds.principle]

    holdout_loader = extra["holdout_loader"]
    gt_principles = extra["holdout/labels"]
    gt_principle_slugs = extra["holdout/labels_principle"]
    holdout_types = extra["holdout/holdout_types"]
    examples_seen = extra.get("examples_seen", 0)
    idx = 0
    all_scores = []
    eval_records = []

    # Generate model responses and explanations for the holdout set, then compute consistency scores
    with torch.no_grad():
        for _, (explanation_batch, behavior_batch) in enumerate(holdout_loader):

            curr_responses, _, _ = sample_conts_gen(
                model,
                tokenizer,
                behavior_batch,
                max_new_tokens=cfg.sampling.max_new_tokens,
                temperature=cfg.sampling.temp,
                top_p=cfg.sampling.top_p,
                greedily=False,
            )

            expl_sys_prompts = [MORAL_SYSTEM_PROMPT] * len(explanation_batch)

            curr_explanations, _, _ = sample_conts_gen(
                model,
                tokenizer,
                explanation_batch,
                max_new_tokens=cfg.sampling.max_new_tokens,
                temperature=cfg.sampling.temp,
                top_p=cfg.sampling.top_p,
                greedily=False,
                system_prompts=expl_sys_prompts,
            )

            for i in range(len(behavior_batch)):
                score = phi_moral_jury_soft(
                    behaviors=[curr_responses[i]],
                    explanations=[[curr_explanations[i]]],
                    model=model,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    extra={**extra, "behavior_prompts": [behavior_batch[i]]},
                ).item()

                all_scores.append(score)
                eval_records.append(
                    {
                        "idx": idx,
                        "prompt": behavior_batch[i],
                        "response": curr_responses[i],
                        "principle": curr_explanations[i],
                        "gt_principle": gt_principles[idx],
                        "gt_principle_slug": gt_principle_slugs[idx],
                        "holdout_type": holdout_types[idx],
                        "consistency_score": score,
                    }
                )
                idx += 1

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    def _subset_metrics(htype):
        subset = [r for r in eval_records if r["holdout_type"] == htype]
        n = len(subset)
        avg = sum(r["consistency_score"] for r in subset) / n if n else 0.0
        return avg, n

    broad_avg, n_broad = _subset_metrics("broad_category")
    fine_avg, n_fine = _subset_metrics("fine_principle")

    # Save holdout snapshot
    save_dir = (
        Path(getattr(cfg.learning, "results_dir", cfg.learning.save_dir))
        / "eval_snapshots"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = save_dir / f"holdout_eval_{examples_seen}.json"
    with open(snapshot_path, "w") as f:
        json.dump(
            {
                "examples_seen": examples_seen,
                "holdout_avg_consistency": avg_score,
                "holdout_broad_category_avg_consistency": broad_avg,
                "holdout_broad_category_n": n_broad,
                "holdout_fine_principle_avg_consistency": fine_avg,
                "holdout_fine_principle_n": n_fine,
                "records": eval_records,
            },
            f,
            indent=2,
        )

    print(f"Saved holdout eval snapshot to {snapshot_path}")

    # Log metrics
    return {
        "holdout/avg_consistency": avg_score,
        "holdout/broad_category/avg_consistency": broad_avg,
        "holdout/fine_principle/avg_consistency": fine_avg,
    }, extra


def spec_eval_eval(model, tokenizer, val_loader, device, cfg, extra):
    """
    Evaluation function for Spec Eval.
    Prints per-prompt: model response, model principle, ground-truth principle.
    Reports average judge score across all val samples.
    Saves per-example results as JSON snapshot to save_dir/eval_snapshots/.
    """
    model.eval()

    gt_principles = extra["val/labels"]
    gt_principle_slugs = extra["val/labels_principle"]
    examples_seen = extra.get("examples_seen", 0)
    idx = 0
    eval_records = []

    with torch.no_grad():
        for _, (explanation_batch, behavior_batch) in enumerate(val_loader):

            # Sample responses to behavior prompts
            curr_responses, _, _ = sample_conts_gen(
                model,
                tokenizer,
                behavior_batch,
                max_new_tokens=cfg.sampling.max_new_tokens,
                temperature=cfg.sampling.temp,
                top_p=cfg.sampling.top_p,
                greedily=False,
            )

            # Sample principles from explanation prompts
            expl_sys_prompts = [MORAL_SYSTEM_PROMPT] * len(explanation_batch)
            curr_explanations, _, _ = sample_conts_gen(
                model,
                tokenizer,
                explanation_batch,
                max_new_tokens=cfg.sampling.max_new_tokens,
                temperature=cfg.sampling.temp,
                top_p=cfg.sampling.top_p,
                greedily=False,
                system_prompts=expl_sys_prompts,
            )

            # Compute consistency and engagement scores for each sample
            for i in range(len(behavior_batch)):
                local_extra = {**extra, "behavior_prompts": [behavior_batch[i]]}

                cons_score = phi_moral_jury_soft(
                    behaviors=[curr_responses[i]],
                    explanations=[[curr_explanations[i]]],
                    model=model,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    extra=local_extra,
                ).item()
                eng_score = engagement_juror(
                    [[curr_responses[i]]],
                    [[curr_explanations[i]]],
                    model,
                    tokenizer,
                    cfg,
                    local_extra,
                ).item()

                eval_records.append(
                    {
                        "idx": idx,
                        "prompt": behavior_batch[i],
                        "response": curr_responses[i],
                        "principle": curr_explanations[i],
                        "gt_principle": gt_principles[idx],
                        "gt_principle_slug": gt_principle_slugs[idx],
                        "consistency_score": cons_score,
                        "engagement_score": eng_score,
                    }
                )

                print("=" * 120)
                print(f"[Sample {idx}]")
                print(f"  Prompt:              {behavior_batch[i]}")
                print(f"  Model Response:      {curr_responses[i]}")
                print(f"  Model Principle:     {curr_explanations[i]}")
                print(f"  Consistency Score:   {cons_score:.4f}")
                print(f"  Engagement Score:    {eng_score:.4f}")
                print(f"  GT Category:         {gt_principles[idx]}")
                print()

                idx += 1

    snapshot_payload = {
        "examples_seen": examples_seen,
        "records": eval_records,
    }

    avg_consistency = (
        sum(r["consistency_score"] for r in eval_records) / len(eval_records)
        if eval_records
        else 0.0
    )

    avg_engagement = (
        sum(r["engagement_score"] for r in eval_records) / len(eval_records)
        if eval_records
        else 0.0
    )

    metrics = {}
    metrics.update(
        avg_consistency=avg_consistency,
        avg_engagement=avg_engagement,
    )

    snapshot_payload.update(
        avg_consistency=avg_consistency,
        avg_engagement=avg_engagement,
    )

    save_dir = (
        Path(getattr(cfg.learning, "results_dir", cfg.learning.save_dir))
        / "eval_snapshots"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = save_dir / f"eval_{examples_seen}.json"
    with open(snapshot_path, "w") as f:
        json.dump(snapshot_payload, f, indent=2)
    print(f"Saved eval snapshot to {snapshot_path}")

    return metrics, extra


def full_eval(model, tokenizer, val_loader, device, cfg, extra):
    """Runs KL divergence eval, judge-based spec eval, and holdout eval."""
    kl_metrics, extra = kl_eval(model, tokenizer, val_loader, device, cfg, extra)
    judge_metrics, extra = spec_eval_eval(
        model, tokenizer, val_loader, device, cfg, extra
    )
    holdout_metrics, extra = holdout_eval(
        model, tokenizer, val_loader, device, cfg, extra
    )
    return {**kl_metrics, **judge_metrics, **holdout_metrics}, extra


### Main entry point for training ###
@hydra.main(
    config_path="../../configs",
    config_name="moral_expl",
    version_base=None,
)
def main(cfg: DictConfig) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    load_dir = getattr(cfg, "load_dir", None) or None
    model, tokenizer = load_model_tok(
        cfg.model_name,
        cfg.mode,
        adapter_path=load_dir,
        is_trainable=bool(load_dir),
    )

    # Ensure pad token + left pad
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"

    # Instantiate optimizer over trainable params only
    trainable_params = (p for p in model.parameters() if p.requires_grad)
    optim = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.learning.lr),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=torch.cuda.is_available(),
    )

    # Determine reward function based on auxiliary weight
    aux_w = float(getattr(cfg.learning, "auxiliary_weight", 0.0))

    if aux_w > 0:
        reward_fn = phi_moral_jury_multi_signal
    else:
        reward_fn = phi_moral_jury_soft

    # Determine whether to use a continuation training loss function
    if getattr(cfg, "cont_training", None) is not None:
        from functools import partial

        max_len = int(getattr(cfg.cont_training, "max_completion_tokens", 512))
        cont_training_loss_fn = partial(cont_training_loss, max_length=max_len)
    else:
        cont_training_loss_fn = None

    # Consistency config for the trainer
    cons_cfg = ConsistencyConfig(
        load_data=lambda c: load_data_spec_eval(c),
        collect_behaviors=collect_behaviors,
        collect_explanations=collect_explanations,
        reward_fn=reward_fn,
        cont_training_loss_fn=cont_training_loss_fn,
        eval_fn=full_eval,
        verbose_logging=verbose_logging_moral,
        untouched_kl_fn=untouched_side_kl_loss,
    )

    # Initialize the trainer
    trainer = ConsistencyTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optim,
        device=device,
        train_cfg=cfg,
        cons_cfg=cons_cfg,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
