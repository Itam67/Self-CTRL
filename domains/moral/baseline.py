"""
Introspective-baseline trainer (the "Expl. baseline" / "Beh. baseline" figure
conditions).

Each side is updated against the frozen base policy's own self-report, with NO
cross-side coupling in the reward:

  Explanation update (bw=0):  reward = P_base(Yes | prompt, principle)
                              "Does this principle describe how you'd respond?"

  Behavior update    (bw=1):  reward = P_base(Yes | prompt, response)
                              "Is this response one you'd endorse producing?"

Mirrors domains/moral.py exactly except for reward_fn — same data, collectors,
eval, untouched-side KL, continued-training loss, and ConsistencyTrainer. Which
side trains is chosen by learning.bw; the mixed setting is not meaningful here
because the two baseline rewards are independent.

Each side has its own config, so the two runs cannot write to the same save_dir:
    python domains/moral_baseline.py                                    # Expl. baseline
    python domains/moral_baseline.py --config-name moral_baseline_beh   # Beh. baseline
"""

import sys
from pathlib import Path

# Allow running this file directly (`python domains/moral_baseline.py`): Python
# only puts this file's dir on sys.path, so add the repo root to resolve
# consistency.*/domains.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import torch
from omegaconf import DictConfig
from contextlib import nullcontext

from consistency.trainer import ConsistencyConfig, ConsistencyTrainer
from consistency.model_utils import load_model_tok, cont_training_loss
from domains.moral import (
    collect_behaviors,
    collect_explanations,
    full_eval,
    load_data_spec_eval,
    untouched_side_kl_loss,
    verbose_logging_moral,
)
from domains.moral_judges import (
    BEH_BASELINE_TEMPLATE,
    EXPL_BASELINE_TEMPLATE,
    soft_score_compliance,
)


### Helper functions ###
def _score_unary(user_prompts, per_example_Ks, model, tokenizer):
    """
    Score a flat list of unary judge prompts on the FROZEN BASE policy and
    reshape back to [B, K].

    The adapter is disabled for the scoring pass (same trick phi_moral_jury_soft
    uses for the jury), so the reward always comes from the base policy's
    self-report rather than the policy being trained.
    """
    device = next(model.parameters()).device

    adapter_ctx = (
        model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    )

    torch.cuda.empty_cache()
    with torch.no_grad(), adapter_ctx:
        all_soft = soft_score_compliance(
            model,
            tokenizer,
            system_prompts=None,
            user_prompts=user_prompts,
            device=device,
        )

    idx = 0
    rewards = []
    for K in per_example_Ks:
        rewards.append(torch.tensor(all_soft[idx : idx + K], device=device))
        idx += K

    return torch.stack(rewards)  # [B, K]


### Baseline reward functions ###
def phi_expl_baseline_soft(explanations, model, tokenizer, cfg, extra):
    """Unary explanation reward — the judge sees (prompt, principle), no behavior."""
    behavior_prompts = extra["behavior_prompts"]

    user_prompts = []
    per_example_Ks = []
    for i in range(len(behavior_prompts)):
        per_example_Ks.append(len(explanations[i]))
        for expl in explanations[i]:
            user_prompts.append(
                EXPL_BASELINE_TEMPLATE.format(
                    prompt=behavior_prompts[i],
                    principle=expl,
                )
            )

    return _score_unary(user_prompts, per_example_Ks, model, tokenizer)


def phi_beh_baseline_soft(behaviors, model, tokenizer, cfg, extra):
    """Unary behavior reward — the judge sees (prompt, response), no explanation."""
    behavior_prompts = extra["behavior_prompts"]

    user_prompts = []
    per_example_Ks = []
    for i in range(len(behavior_prompts)):
        per_example_Ks.append(len(behaviors[i]))
        for beh in behaviors[i]:
            user_prompts.append(
                BEH_BASELINE_TEMPLATE.format(
                    prompt=behavior_prompts[i],
                    response=beh,
                )
            )

    return _score_unary(user_prompts, per_example_Ks, model, tokenizer)


def phi_baseline_introspective(behaviors, explanations, model, tokenizer, cfg, extra):
    """
    Dispatcher matching the trainer's reward_fn signature.

    ConsistencyTrainer._grpo_loss passes the side being SCORED as `explanations`
    (the [B][K] candidates whose nlls carry gradients) and the anchor from the
    other side as `behaviors`:

      explanation update: behaviors = [B] flat strings   -> phi_expl_baseline_soft
      behavior update:    behaviors = [B][K] lists       -> phi_beh_baseline_soft

    Whichever argument belongs to the untouched side is ignored — that is the
    entire point of the baseline (no cross-side coupling in the reward).
    """
    if isinstance(behaviors[0], str):
        return phi_expl_baseline_soft(explanations, model, tokenizer, cfg, extra)
    return phi_beh_baseline_soft(behaviors, model, tokenizer, cfg, extra)


### Main entry point for training ###
@hydra.main(
    config_path="../configs",
    config_name="moral_baseline_expl",
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

    # The auxiliary (engagement) signal is a property of the jury reward and has
    # no counterpart in the unary baseline rewards — refuse to run silently with
    # a setting that would be ignored.
    aux_w = float(getattr(cfg.learning, "auxiliary_weight", 0.0))
    if aux_w > 0:
        raise ValueError(
            "learning.auxiliary_weight > 0 has no effect on the introspective "
            "baseline (its rewards are unary). Set it to 0."
        )

    # Determine whether to use a continuation training loss function
    if getattr(cfg, "cont_training", None) is not None:
        from functools import partial

        max_len = int(getattr(cfg.cont_training, "max_completion_tokens", 512))
        cont_training_loss_fn = partial(cont_training_loss, max_length=max_len)
    else:
        cont_training_loss_fn = None

    # Consistency config for the trainer — identical to domains/moral.py except
    # for reward_fn.
    cons_cfg = ConsistencyConfig(
        load_data=lambda c: load_data_spec_eval(c),
        collect_behaviors=collect_behaviors,
        collect_explanations=collect_explanations,
        reward_fn=phi_baseline_introspective,
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
