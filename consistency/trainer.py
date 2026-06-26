from dataclasses import dataclass
from typing import Callable, Optional
import wandb
from pathlib import Path
from omegaconf import OmegaConf
import torch

try:
    import wandb

    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

# _WANDB_AVAILABLE = False

VERBOSE = True


# ---------------------------
# Config Container
# ---------------------------
@dataclass
class ConsistencyConfig:
    """Container for all REINFORCE strategy functions."""

    load_data: Callable
    collect_behaviors: Callable
    collect_explanations: Callable
    reward_fn: Callable
    reg_loss_fn: Optional[Callable] = None
    eval_fn: Optional[Callable] = None
    verbose_logging: Optional[Callable] = None
    # Optional regularizer that anchors the side NOT being updated (weighted by
    # bw) to the base model. Signature:
    #   fn(model, tokenizer, behavior_prompts, explanation_prompts,
    #      cfg, extra, device) -> (R_tensor, {"kl_beh": float, "kl_expl": float})
    # Only invoked when cfg.learning.untouched_side_kl_weight > 0.
    untouched_kl_fn: Optional[Callable] = None
    # Optional token-level entropy bonus on the side being updated. Signature:
    #   fn(model, tokenizer, behavior_prompts, behaviors,
    #      explanation_prompts, explanations,
    #      cfg, extra, device) -> H_scalar (with grad)
    # Only invoked when cfg.learning.entropy_beta > 0.
    entropy_fn: Optional[Callable] = None


# ---------------------------
# Main Trainer Class
# ---------------------------


class ConsistencyTrainer:
    """Flexible consistency trainer supporting multiple use cases."""

    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        device,
        cfg,
        strategies: ConsistencyConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        self.strategies = strategies

        # Load data
        self.train_loader, self.val_loader, self.extra = strategies.load_data(cfg)

    def _reinforce_loss(
        self, explanations, actions_nlls, reward_samples, valid_mask=None
    ):
        """Single REINFORCE loss step"""

        # Collect Rewards
        rewards = self.strategies.reward_fn(
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

        rewards = rewards.to(self.device)
        valid_mask = valid_mask.to(self.device)
        masked_rewards = rewards.masked_fill(~valid_mask, -1e9)

        # Store index of highest reward sample for each example in batch
        # This can be used for additional loss terms like behavioral alignment on the best explanation
        best_indices = masked_rewards.argmax(dim=1)

        m = valid_mask.to(rewards.dtype)  # Valid mask
        assert m.sum() > 0, "no valid actions in microbatch; loss divide by zero"

        # Calculate advantages
        if rewards.shape[-1] == 1:
            # If only one sample per example, use reward itself as advantage
            advantages = rewards.detach()
        else:
            # If multiple samples, calculate baseline and normalize advantages
            denom = m.sum(dim=1, keepdim=True)
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
        if self.strategies.eval_fn is None:
            return

        self.extra["examples_seen"] = examples_seen
        metrics, extra = self.strategies.eval_fn(
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
        epochs = int(self.cfg.learning.epochs)
        # Warm-start support: start the step counter past the resumed ckpt's
        # step so W&B/eval/save timelines continue rather than restart at 0.
        examples_seen = int(getattr(self.cfg.learning, "examples_seen_start", 0) or 0)
        save_every = int(self.cfg.learning.save_every)
        # Schedule the next save strictly after the starting step.
        next_save = examples_seen + save_every if examples_seen > 0 else save_every

        self.model.to(self.device)

        # Initialize wandb. When wandb.resume_id is set we attach to the
        # existing run so the curves continue as one line instead of starting
        # a new run.
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
                print(f"wandb.init failed: {e}")

        # Initial evaluation
        self._evaluate(examples_seen)

        # Training loop
        self.model.train()
        for epoch in range(1, epochs + 1):
            print(f"Starting epoch {epoch}/{epochs}...")

            # Iterate over batches
            for batch in self.train_loader:
                micro_bsz = int(self.cfg.learning.micro_batch_size)

                explanation_prompts_all = batch[0]
                behavior_prompts_all = batch[1]
                reg_data = batch[2] if len(batch) > 2 else None

                N = len(explanation_prompts_all)
                self.optimizer.zero_grad(set_to_none=True)

                total_loss_sum = 0.0
                total_reg_sum = 0.0
                total_explanation_sum = 0.0
                total_behavior_sum = 0.0
                total_behavior_std = 0.0
                total_explanation_std = 0.0
                total_untouched_kl_sum = 0.0
                total_untouched_kl_beh_sum = 0.0
                total_untouched_kl_expl_sum = 0.0
                total_entropy_sum = 0.0

                for start in range(0, N, micro_bsz):
                    end = min(start + micro_bsz, N)
                    B = end - start

                    # Collect Behaviors
                    behaviors, behavior_nlls, extra = self.strategies.collect_behaviors(
                        self.model,
                        self.tokenizer,
                        behavior_prompts_all[start:end],
                        self.cfg,
                        self.extra,
                    )
                    self.extra.update(extra) if extra is not None else None

                    # Collect Explanations
                    explanations, explanations_nlls, valid_mask, extra = (
                        self.strategies.collect_explanations(
                            self.model,
                            self.tokenizer,
                            explanation_prompts_all[start:end],
                            self.cfg,
                            self.extra,
                        )
                    )
                    self.extra.update(extra) if extra is not None else None

                    # Detect whether behaviors are K-sampled [B][K] or single [B]
                    k_behaviors = isinstance(behaviors[0], list)

                    if k_behaviors:
                        # Pick highest-likelihood behavior as anchor for explanation update
                        best_beh_idx = behavior_nlls.argmin(dim=1)  # [B]
                        best_behaviors = [
                            behaviors[i][best_beh_idx[i].item()] for i in range(B)
                        ]  # [B] flat strings
                    else:
                        best_behaviors = behaviors  # already [B] flat strings

                    # Explanation REINFORCE: grade K explanations against best behavior
                    (
                        explanation_loss,
                        best_indices,
                        explanation_advantages,
                        explanation_rewards,
                    ) = self._reinforce_loss(
                        explanations,  # [B][K]
                        explanations_nlls,  # [B, K] — gradients flow here
                        best_behaviors,  # [B]
                        valid_mask=valid_mask,
                    )

                    if k_behaviors:
                        # Pick highest-likelihood valid explanation as anchor for behavior update
                        masked_expl_nlls = explanations_nlls.clone()
                        if valid_mask is not None:
                            masked_expl_nlls.masked_fill_(
                                ~valid_mask.bool(), float("inf")
                            )
                        best_expl_idx = masked_expl_nlls.argmin(dim=1)  # [B]
                        best_explanations = [
                            [explanations[i][best_expl_idx[i].item()]] for i in range(B)
                        ]  # [B][1]

                        # Behavior REINFORCE: grade K behaviors against best explanation
                        behavior_loss, _, behavior_advantages, behavior_rewards = (
                            self._reinforce_loss(
                                best_explanations,  # [B][1] — broadcast to K
                                behavior_nlls,  # [B, K] — gradients flow here
                                behaviors,  # [B][K]
                            )
                        )
                    else:
                        # Single behavior: use highest-reward explanation (original logic)
                        highest_reward_explanations = [
                            [explanations[i][best_indices[i].item()]] for i in range(B)
                        ]  # [B][1]

                        behavior_loss, _, behavior_advantages, behavior_rewards = (
                            self._reinforce_loss(
                                highest_reward_explanations,  # [B][1]
                                behavior_nlls,  # [B, 1]
                                behaviors,  # [B]
                            )
                        )

                    # Calculate regularization loss if applicable
                    if reg_data is not None and self.strategies.reg_loss_fn is not None:
                        # Support two reg_data formats:
                        #   1) (prompts_list, completions_list) tuple — for SFT reg
                        #   2) flat list — legacy format (used as both prompts & completions)
                        if (
                            isinstance(reg_data, (tuple, list))
                            and len(reg_data) == 2
                            and isinstance(reg_data[0], (list, tuple))
                        ):
                            reg_prompts = reg_data[0][start:end]
                            reg_completions = reg_data[1][start:end]
                        else:
                            reg_prompts = reg_data[start:end]
                            reg_completions = reg_data[start:end]
                        reg_prompts_for_later = reg_prompts
                        reg_completions_for_later = reg_completions
                    else:
                        reg_prompts_for_later = None
                        reg_completions_for_later = None

                    loss = (
                        1 - self.cfg.learning.bw
                    ) * explanation_loss + self.cfg.learning.bw * behavior_loss

                    total_loss_sum += float(loss.detach().item()) * B

                    # Per-example weighting
                    mb_frac = B / max(N, 1)
                    (loss * mb_frac).backward()

                    # Untouched-side KL regularizer (optional). Pulls the side
                    # we're NOT updating back toward the base model — weighted
                    # by bw so bw=1 protects explanations, bw=0 protects
                    # behaviors. Gradients flow through the current-model NLL
                    # on sequences sampled from base (adapter disabled).
                    untouched_kl_w = float(
                        getattr(self.cfg.learning, "untouched_side_kl_weight", 0.0)
                    )
                    if (
                        untouched_kl_w > 0
                        and self.strategies.untouched_kl_fn is not None
                    ):
                        torch.cuda.empty_cache()
                        R, kl_parts = self.strategies.untouched_kl_fn(
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
                        (untouched_kl_w * R * mb_frac).backward()

                    # Token-level entropy bonus (optional). Adds  -beta * H  to
                    # the loss, where H is the mean per-token policy entropy on
                    # the sampled (prompt, continuation) pairs for the side
                    # being updated. Gradient flows directly through the
                    # current-step softmax — not via REINFORCE.
                    entropy_beta = float(
                        getattr(self.cfg.learning, "entropy_beta", 0.0)
                    )
                    if entropy_beta > 0 and self.strategies.entropy_fn is not None:
                        torch.cuda.empty_cache()
                        H = self.strategies.entropy_fn(
                            self.model,
                            self.tokenizer,
                            behavior_prompts_all[start:end],
                            behaviors,
                            explanation_prompts_all[start:end],
                            explanations,
                            self.cfg,
                            self.extra,
                            self.device,
                        )
                        total_entropy_sum += float(H.detach().item()) * B
                        total_loss_sum += -entropy_beta * float(H.detach().item()) * B
                        # H is a non-differentiable zero when the entropy term
                        # doesn't apply (e.g. behavior-side bonus at bw=0);
                        # backprop only when it actually carries a gradient.
                        if H.requires_grad:
                            (-entropy_beta * H * mb_frac).backward()

                    # Compute reg loss AFTER main backward to avoid holding
                    # both computation graphs in memory simultaneously.
                    torch.cuda.empty_cache()
                    if reg_prompts_for_later is not None:
                        reg_loss = self.strategies.reg_loss_fn(
                            self.model,
                            self.tokenizer,
                            reg_prompts_for_later,
                            reg_completions_for_later,
                            device=self.device,
                        )
                        total_reg_sum += float(reg_loss.detach().item()) * B
                        total_loss_sum += (
                            self.cfg.learning.reg_loss_weight
                            * float(reg_loss.detach().item())
                            * B
                        )
                        (
                            self.cfg.learning.reg_loss_weight * reg_loss * mb_frac
                        ).backward()

                    # Debugging
                    total_explanation_sum += explanation_rewards.mean().item() * B
                    total_explanation_std += (
                        explanation_rewards.std(dim=1).mean().item() * B
                        if explanation_rewards.shape[1] > 1
                        else 0.0
                    )

                    # Behavior rewards
                    total_behavior_sum += behavior_rewards.mean().item() * B
                    total_behavior_std += (
                        behavior_rewards.std().item() * B
                        if behavior_rewards.numel() > 1
                        else 0.0
                    )

                    if (
                        self.strategies.verbose_logging is not None
                        and self.cfg.verbose_logging
                    ):
                        self.strategies.verbose_logging(
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
                            reg_prompts=reg_prompts if reg_data is not None else None,
                            reg_completions=(
                                reg_completions if reg_data is not None else None
                            ),
                        )

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=getattr(self.cfg.learning, "max_grad_norm", 1.0),
                )
                self.optimizer.step()
                examples_seen += N

                # Logging
                if _WANDB_AVAILABLE:
                    log_payload = {
                        "train/avg_loss": total_loss_sum / max(N, 1),
                        "train/reg_loss": total_reg_sum / max(N, 1),
                        "train/behavior_rewards": total_behavior_sum / max(N, 1),
                        "train/behavior_rewards_std": total_behavior_std
                        / max(N, 1),
                        "train/explanation_rewards": total_explanation_sum
                        / max(N, 1),
                        "train/explanation_rewards_std": total_explanation_std
                        / max(N, 1),
                        "train/examples_seen": examples_seen,
                        "train/grad_norm": grad_norm,
                    }
                    if (
                        float(
                            getattr(
                                self.cfg.learning, "untouched_side_kl_weight", 0.0
                            )
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
                    if (
                        float(getattr(self.cfg.learning, "entropy_beta", 0.0)) > 0
                    ):
                        log_payload["train/token_entropy"] = (
                            total_entropy_sum / max(N, 1)
                        )
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
