import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Optional
from pathlib import Path
from omegaconf import OmegaConf
import torch

try:
    import wandb

    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

# Turn on/off verbose logging for debugging.
VERBOSE = True


def wandb_required(cfg) -> bool:
    """Whether losing wandb logging should abort the run.

    Training metrics must never be silently lost, so wandb is required by
    default. The two sanctioned opt-outs are explicit: WANDB_MODE=disabled (no
    logging wanted) or WANDB_MODE=offline (still records locally, no network),
    and the config flag wandb.required: false.
    """
    if os.environ.get("WANDB_MODE", "").lower() in ("disabled", "offline"):
        return False
    wandb_cfg = getattr(cfg, "wandb", None)
    return bool(getattr(wandb_cfg, "required", True))


# Conssistency configuration container
@dataclass
class ConsistencyConfig:
    """Container for all REINFORCE strategy functions."""

    load_data: Callable
    collect_behaviors: Callable
    collect_explanations: Callable
    reward_fn: Callable
    cont_training_loss_fn: Optional[Callable] = (
        None  # Continued training loss function (optional)
    )
    untouched_kl_fn: Optional[Callable] = (
        None  # Regularization function for the side not being updated (only used if lambda=1 or 0)
    )
    eval_fn: Optional[Callable] = None  # Evaluation function
    verbose_logging: Optional[Callable] = None  # Verbose logging function


# Main trainer class
class ConsistencyTrainer:
    """Flexible consistency trainer supporting multiple use cases."""

    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        device,
        train_cfg: OmegaConf,
        cons_cfg: ConsistencyConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.cfg = train_cfg
        self.cons_cfg = cons_cfg

        # Load data
        self.train_loader, self.val_loader, self.extra = self.cons_cfg.load_data(
            self.cfg
        )

    def _grpo_loss(self, explanations, actions_nlls, reward_samples, valid_mask=None):
        """Single GRPO loss step. The heart of the method."""

        # Collect Rewards
        rewards = self.cons_cfg.reward_fn(
            reward_samples,
            explanations,
            self.model,
            self.tokenizer,
            self.cfg,
            self.extra,
        ).detach()

        # Checking shapes
        assert rewards.shape == actions_nlls.shape, (rewards.shape, actions_nlls.shape)

        if valid_mask is not None:
            assert rewards.shape == valid_mask.shape, (rewards.shape, valid_mask.shape)

        # Set up mask to block invalid samples
        if valid_mask is None:
            valid_mask = torch.ones_like(rewards, dtype=torch.bool, device=self.device)
        else:
            valid_mask = valid_mask.to(self.device).bool()

        # Move to device and mask invalid rewards
        rewards = rewards.to(self.device)
        valid_mask = valid_mask.to(self.device)
        masked_rewards = rewards.masked_fill(~valid_mask, -1e9)
        m = valid_mask.to(rewards.dtype)

        # Store index of highest reward sample for each example in batch
        best_indices = masked_rewards.argmax(dim=1)

        if m.sum() == 0:
            # Every action in this microbatch is masked invalid (e.g. a coins
            # example where no candidate program parsed). Contribute nothing
            # rather than dividing by zero; multiplying through the nlls keeps
            # the returned loss attached to the graph the caller backwards.
            zero_loss = (actions_nlls * 0.0).sum()
            return zero_loss, best_indices, torch.zeros_like(rewards), masked_rewards

        # Calculate advantages
        if rewards.shape[-1] == 1:
            # If only one sample per example, use reward itself as advantage
            advantages = rewards.detach()
        else:
            # If multiple samples, calculate baseline and normalize advantages.
            # clamp_min keeps an all-masked ROW finite (mean/var are garbage
            # there, but its m-row zeroes it out of the loss below).
            denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = (rewards * m).sum(dim=1, keepdim=True) / denom
            var = ((rewards - mean) ** 2 * m).sum(dim=1, keepdim=True) / denom
            std = var.sqrt().clamp_min(0.05)
            advantages = ((rewards - mean) / std).detach()

        # Scale masked action nll by advantages
        loss_mat = (advantages * actions_nlls) * m
        loss = loss_mat.sum() / m.sum()

        return loss, best_indices, advantages, masked_rewards

    def _evaluate(self, examples_seen):
        """Run evaluation if eval_fn is provided."""
        if self.cons_cfg.eval_fn is None:
            return

        # Track number of examples seen so far for logging purposes
        self.extra["examples_seen"] = examples_seen

        # Run evaluation function
        metrics, extra = self.cons_cfg.eval_fn(
            self.model,
            self.tokenizer,
            self.val_loader,
            self.device,
            self.cfg,
            self.extra,
        )
        self.extra.update(extra) if extra is not None else None

        if _WANDB_AVAILABLE and metrics:
            # If the eval_fn already prefixed a key (contains a '/'), respect
            # that prefix so it can route metrics to a different W&B section
            # (e.g. "holdout/..."). Otherwise default to "eval/...".
            wandb.log(
                {(k if "/" in k else f"eval/{k}"): v for k, v in metrics.items()},
                step=examples_seen,
            )

    def _save_checkpoint(self, examples_seen):
        """Save model checkpoint."""

        save_dir = Path(self.cfg.learning.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = save_dir / f"ckpt_{examples_seen}"
        self.model.save_pretrained(ckpt_dir.as_posix())
        print(f"Saved checkpoint to {ckpt_dir}")

    def train(self):
        """Main training loop."""
        global _WANDB_AVAILABLE

        epochs = int(self.cfg.learning.epochs)
        examples_seen = int(getattr(self.cfg.learning, "examples_seen_start", 0) or 0)
        save_every = int(self.cfg.learning.save_every)

        # Schedule the next save strictly after the starting step.
        next_save = examples_seen + save_every if examples_seen > 0 else save_every

        self.model.to(self.device)

        # Initialize wandb (check if resuming previous run). Strict by default:
        # a run whose metrics silently vanish is worse than one that refuses to
        # start — see wandb_required for the sanctioned opt-outs.
        if not _WANDB_AVAILABLE and wandb_required(self.cfg):
            raise RuntimeError(
                "wandb is required but could not be imported. Install wandb, or "
                "opt out explicitly with WANDB_MODE=offline (still records "
                "locally), WANDB_MODE=disabled, or wandb.required: false."
            )
        if _WANDB_AVAILABLE:
            try:
                resume_id = getattr(self.cfg.wandb, "resume_id", None) or None
                wandb_kwargs = dict(
                    project=self.cfg.wandb.project,
                    name=self.cfg.wandb.run_name,
                    config=OmegaConf.to_container(self.cfg, resolve=True),
                )
                if resume_id:
                    wandb_kwargs["id"] = resume_id
                    wandb_kwargs["resume"] = "allow"
                wandb.init(**wandb_kwargs)
                wandb.define_metric("train/examples_seen")
                wandb.define_metric("eval/*", step_metric="train/examples_seen")
                wandb.define_metric("train/*", step_metric="train/examples_seen")
                wandb.define_metric("holdout/*", step_metric="train/examples_seen")
            except Exception as e:
                if wandb_required(self.cfg):
                    raise RuntimeError(
                        "wandb.init failed and wandb logging is required. Fix "
                        "the wandb setup, or opt out explicitly with "
                        "WANDB_MODE=offline / WANDB_MODE=disabled / "
                        "wandb.required: false."
                    ) from e
                # Without this the run stays "available" but has no active run,
                # and the first wandb.log below raises and kills training.
                _WANDB_AVAILABLE = False
                print(f"wandb.init failed, continuing without logging: {e}")

        # Initial evaluation: the pre-training anchor point of every
        # consistency curve (the original ran it too).
        self._evaluate(examples_seen)

        # Training loop
        self.model.train()
        for epoch in range(1, epochs + 1):
            print(f"Starting epoch {epoch}/{epochs}...")

            # Iterate over batches
            for batch in self.train_loader:
                micro_bsz = int(self.cfg.learning.micro_batch_size)

                # Unpack batch
                explanation_prompts_all = batch[0]
                behavior_prompts_all = batch[1]
                cont_training_data = batch[2] if len(batch) > 2 else None

                N = len(explanation_prompts_all)

                # Initialize accumulators for logging
                total_loss_sum = 0.0
                total_cont_training_sum = 0.0
                total_explanation_sum = 0.0
                total_behavior_sum = 0.0
                total_behavior_std = 0.0
                total_explanation_std = 0.0
                total_untouched_kl_sum = 0.0
                total_untouched_kl_beh_sum = 0.0
                total_untouched_kl_expl_sum = 0.0

                self.optimizer.zero_grad(set_to_none=True)

                # Iterate over micro-batches to manage memory usage
                for start in range(0, N, micro_bsz):
                    end = min(start + micro_bsz, N)
                    B = end - start

                    ### Collect samples ###

                    # Both sides are always collected, but a side that bw
                    # weights to zero is never backwarded: its nlls only get
                    # argmin'd to pick the other side's anchor. Building a graph
                    # for it is waste, and at bw=1.0 it is actively harmful —
                    # nothing frees that graph until `explanations_nlls` is
                    # rebound below, i.e. after the NEXT collect_explanations has
                    # already built its own, so three sets of K sequences coexist
                    # and OOM. (At bw=0.0 the stale side is `behavior_nlls`,
                    # which is rebound first, which is why only bw=1.0 died.)
                    bw = float(self.cfg.learning.bw)
                    beh_grad = nullcontext() if bw > 0.0 else torch.no_grad()
                    expl_grad = nullcontext() if bw < 1.0 else torch.no_grad()

                    # Collect Behaviors
                    with beh_grad:
                        behaviors, behavior_nlls, extra = (
                            self.cons_cfg.collect_behaviors(
                                self.model,
                                self.tokenizer,
                                behavior_prompts_all[start:end],
                                self.cfg,
                                self.extra,
                            )
                        )
                    self.extra.update(extra) if extra is not None else None

                    # Collect Explanations
                    with expl_grad:
                        explanations, explanations_nlls, valid_mask, extra = (
                            self.cons_cfg.collect_explanations(
                                self.model,
                                self.tokenizer,
                                explanation_prompts_all[start:end],
                                self.cfg,
                                self.extra,
                            )
                        )
                    self.extra.update(extra) if extra is not None else None

                    ### Calculate losses (skip the side bw weights to zero) ###
                    k_behaviors = isinstance(behaviors[0], list)  # [B][K] vs [B]
                    loss = 0.0
                    explanation_rewards = behavior_rewards = None
                    explanation_advantages = behavior_advantages = None

                    if bw < 1.0:  # explanation update (skip when bw == 1)
                        # Pick highest-likelihood behavior as anchor
                        if k_behaviors:
                            best_beh_idx = behavior_nlls.argmin(dim=1)  # [B]
                            best_behaviors = [
                                behaviors[i][best_beh_idx[i].item()] for i in range(B)
                            ]  # [B] flat strings
                        else:
                            best_behaviors = behaviors  # already [B] flat strings
                        (
                            explanation_loss,
                            _,
                            explanation_advantages,
                            explanation_rewards,
                        ) = self._grpo_loss(
                            explanations,  # [B][K]
                            explanations_nlls,  # [B, K] — gradients flow here
                            best_behaviors,  # [B]
                            valid_mask=valid_mask,
                        )
                        loss = loss + (1 - bw) * explanation_loss

                    if bw > 0.0:  # behavior update (skip when bw == 0)
                        # Pick highest-likelihood valid explanation as anchor
                        masked_expl_nlls = explanations_nlls.clone()
                        if valid_mask is not None:
                            masked_expl_nlls.masked_fill_(
                                ~valid_mask.bool(), float("inf")
                            )
                        best_expl_idx = masked_expl_nlls.argmin(dim=1)  # [B]
                        best_explanations = [
                            [explanations[i][best_expl_idx[i].item()]] for i in range(B)
                        ]  # [B][1]
                        behavior_loss, _, behavior_advantages, behavior_rewards = (
                            self._grpo_loss(
                                best_explanations,  # [B][1]
                                behavior_nlls,  # [B, K] or [B, 1] — gradients flow here
                                behaviors,  # [B][K] or [B]
                            )
                        )
                        loss = loss + bw * behavior_loss

                    total_loss_sum += float(loss.detach().item()) * B

                    # Per-example weighting
                    mb_frac = B / max(N, 1)
                    (loss * mb_frac).backward()

                    ### Calculate KL penalty ###

                    # Untouched-side KL regularizer (optional). Pulls the side
                    # we're NOT updating back toward the base model
                    untouched_kl_w = float(
                        getattr(self.cfg.learning, "untouched_side_kl_weight", 0.0)
                    )
                    if untouched_kl_w > 0 and self.cons_cfg.untouched_kl_fn is not None:
                        torch.cuda.empty_cache()
                        R, kl_parts = self.cons_cfg.untouched_kl_fn(
                            self.model,
                            self.tokenizer,
                            behavior_prompts_all[start:end],
                            explanation_prompts_all[start:end],
                            self.cfg,
                            self.extra,
                            self.device,
                        )
                        total_untouched_kl_sum += float(R.detach().item()) * B
                        total_untouched_kl_beh_sum += kl_parts["kl_beh"] * B
                        total_untouched_kl_expl_sum += kl_parts["kl_expl"] * B
                        total_loss_sum += untouched_kl_w * float(R.detach().item()) * B

                        # Update gradients for untouched-side KL regularization
                        (untouched_kl_w * R * mb_frac).backward()

                    ### Calculate continued training loss ###
                    cont_training_prompts = None
                    cont_training_completions = None
                    if (
                        cont_training_data is not None
                        and self.cons_cfg.cont_training_loss_fn is not None
                    ):
                        cont_training_prompts = cont_training_data[0][start:end]
                        cont_training_completions = cont_training_data[1][start:end]

                    # Compute cont-training loss
                    torch.cuda.empty_cache()
                    if cont_training_prompts is not None:
                        cont_training_loss = self.cons_cfg.cont_training_loss_fn(
                            self.model,
                            self.tokenizer,
                            cont_training_prompts,
                            cont_training_completions,
                            device=self.device,
                        )
                        total_cont_training_sum += (
                            float(cont_training_loss.detach().item()) * B
                        )
                        total_loss_sum += (
                            self.cfg.learning.cont_training_loss_weight
                            * float(cont_training_loss.detach().item())
                            * B
                        )
                        (
                            self.cfg.learning.cont_training_loss_weight
                            * cont_training_loss
                            * mb_frac
                        ).backward()

                    ### Logging and verbose output ###
                    if explanation_rewards is not None:
                        total_explanation_sum += explanation_rewards.mean().item() * B
                        total_explanation_std += (
                            explanation_rewards.std(dim=1).mean().item() * B
                            if explanation_rewards.shape[1] > 1
                            else 0.0
                        )

                    if behavior_rewards is not None:
                        total_behavior_sum += behavior_rewards.mean().item() * B
                        total_behavior_std += (
                            behavior_rewards.std().item() * B
                            if behavior_rewards.numel() > 1
                            else 0.0
                        )

                    if (
                        self.cons_cfg.verbose_logging is not None
                        and self.cfg.verbose_logging
                    ):
                        self.cons_cfg.verbose_logging(
                            behavior_prompts_all[start:end],
                            explanation_prompts_all[start:end],
                            behaviors,
                            explanations,
                            behavior_advantages,
                            explanation_advantages,
                            explanation_rewards,
                            self.extra,
                            behavior_rewards=behavior_rewards,
                            behavior_nlls=behavior_nlls,
                            explanation_nlls=explanations_nlls,
                            cont_training_prompts=(
                                cont_training_prompts
                                if cont_training_data is not None
                                else None
                            ),
                            cont_training_completions=(
                                cont_training_completions
                                if cont_training_data is not None
                                else None
                            ),
                        )

                # Clip the accumulated gradient and step once per batch.
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=getattr(self.cfg.learning, "max_grad_norm", 1.0),
                )
                self.optimizer.step()
                examples_seen += N

                # Logging. Domain hooks (collect_* / phi) may queue extra
                # scalars for this batch's payload under extra["log_metrics"];
                # popped unconditionally so nothing accumulates when wandb is
                # off.
                queued_metrics = self.extra.pop("log_metrics", None)
                if _WANDB_AVAILABLE:
                    log_payload = {
                        "train/avg_loss": total_loss_sum / max(N, 1),
                        "train/cont_training_loss": total_cont_training_sum / max(N, 1),
                        "train/behavior_rewards": total_behavior_sum / max(N, 1),
                        "train/behavior_rewards_std": total_behavior_std / max(N, 1),
                        "train/explanation_rewards": total_explanation_sum / max(N, 1),
                        "train/explanation_rewards_std": total_explanation_std
                        / max(N, 1),
                        "train/examples_seen": examples_seen,
                        "train/grad_norm": grad_norm,
                    }
                    if (
                        float(
                            getattr(self.cfg.learning, "untouched_side_kl_weight", 0.0)
                        )
                        > 0
                    ):
                        log_payload["train/untouched_kl"] = (
                            total_untouched_kl_sum / max(N, 1)
                        )
                        log_payload["train/untouched_kl_beh"] = (
                            total_untouched_kl_beh_sum / max(N, 1)
                        )
                        log_payload["train/untouched_kl_expl"] = (
                            total_untouched_kl_expl_sum / max(N, 1)
                        )

                    if queued_metrics:
                        log_payload.update(queued_metrics)

                    wandb.log(log_payload, step=examples_seen)

                # Evaluate and save checkpoint
                if examples_seen >= next_save:
                    self._evaluate(examples_seen)
                    self._save_checkpoint(examples_seen)
                    next_save += int(self.cfg.learning.save_every)
                    self.model.train()

        if _WANDB_AVAILABLE:
            try:
                wandb.finish()
            except Exception:
                pass
        return self.model
