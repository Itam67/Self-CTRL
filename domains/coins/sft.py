"""Supervised stage for the coins domain — the prerequisite for consistency training."""

import os
import sys
from pathlib import Path

# Allow running this file directly: Python only puts this file's dir on
# sys.path, so add the repo root to resolve consistency.*/domains.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import Dict, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from consistency.data_utils import collate_prompt_output
from consistency.model_utils import (
    derive_seed,
    get_amp_context,
    load_model_tok,
    sample_conts_gen,
)
from consistency.trainer import wandb_required
from domains.coins.data import (
    SFTDataset,
    coin_groups,
    load_opencode_sft_subset,
    read_sft_dataset,
)

try:
    import wandb

    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False


# The rollout prompt used to probe behavior during eval. Matches the phrasing
# the SFT data was built with.
ROLLOUT_PROMPT = (
    "Generate a string (list of space separated 'H' or 'T') representing the "
    "outcomes of {n} independent flips from {coin} and no other text."
)


def _set_seed(seed: int) -> torch.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


### Evaluation ###
def evaluate_loss(model, val_loader, device, step: int) -> Dict[str, float]:
    """Mean validation NLL over held-out demonstrations."""
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with get_amp_context(device):
                losses.append(float(model(**batch, use_cache=False).loss.item()))

    metrics = {"eval/total_loss": sum(losses) / max(1, len(losses))}
    print(f"  eval loss: {metrics['eval/total_loss']:.4f}")
    if _WANDB_AVAILABLE:
        wandb.log(metrics, step=step)
    return metrics


@torch.no_grad()
def evaluate_rollout_bias(
    model, tokenizer, coin_names: List[str], biases: List[float], cfg, step: int
) -> Dict[str, float]:
    """How closely the model's sampled rollouts match each coin's true bias.

    This is the behavioral half of the OOCR check: it should be small for every
    coin, including the ones that never saw a verbalization example, because all
    100 got rollout supervision. Reported split by group so the rollout-only
    coins can be seen to have learned as well as the supervised ones.
    """
    model.eval()
    prompts = [
        ROLLOUT_PROMPT.format(coin=name, n=int(cfg.num_flips)) for name in coin_names
    ]

    # Sample inside a forked RNG with a fixed derived seed: generation here
    # would otherwise consume the global stream, so how often this eval runs
    # (eval_every) would perturb the epoch>=2 shuffle order of training.
    # Deliberately step-independent seed: every eval point draws the same
    # sampling noise, so checkpoints are compared paired — mirrors the
    # cons-stage eval_fn in domains/coins/train.py. fork_rng covers all CUDA
    # devices because manual_seed_all touches every one.
    if torch.cuda.is_available():
        fork_devices = list(range(torch.cuda.device_count()))
    else:
        fork_devices = []
    with torch.random.fork_rng(devices=fork_devices):
        eval_seed = derive_seed(int(cfg.seed), "eval")
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
        continuations, _, _ = sample_conts_gen(
            model,
            tokenizer,
            prompts=prompts,
            max_new_tokens=int(cfg.num_flips) * 2,
            temperature=1.0,
            top_p=1.0,
            greedily=False,
        )

    abs_errs, kept_names = [], []
    for name, gt, cont in zip(coin_names, biases, continuations):
        h, t = cont.count("H"), cont.count("T")
        if h + t == 0:
            continue
        abs_errs.append(abs(h / (h + t) - gt))
        kept_names.append(name)

    if not abs_errs:
        print("  rollout-bias eval: no usable continuations")
        return {}

    # Errors grouped for reporting: the whole set, then each supervision group
    # (derived from the shipped data). A group with no usable rollouts is left
    # out rather than reported as zero.
    group_of, _ = coin_groups()
    errs_by_group = {"all": abs_errs}
    for group in ("sft", "cons", "holdout"):
        errs = [e for n, e in zip(kept_names, abs_errs) if group_of.get(n) == group]
        if errs:
            errs_by_group[group] = errs

    def _metric_key(group: str) -> str:
        base = "eval/rollout_bias_abs_err"
        return base if group == "all" else f"{base}_{group}"

    metrics = {
        _metric_key(group): float(np.mean(errs))
        for group, errs in errs_by_group.items()
    }

    print(
        "  rollout-bias abs err:  "
        + "  ".join(
            f"{group}={np.mean(errs):.4f}" for group, errs in errs_by_group.items()
        )
    )
    if _WANDB_AVAILABLE:
        wandb.log(metrics, step=step)
    return metrics


### Training ###
def train(model, tokenizer, train_loader, val_loader, optim, coins, biases, device, cfg):
    epochs = int(cfg.epochs)
    log_every = int(cfg.log_every)
    eval_every = int(getattr(cfg, "eval_every", 0) or 0)
    save_dir = Path(cfg.save_dir)

    step = 0
    running = 0.0

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        for batch in train_loader:
            model.train()
            batch = {k: v.to(device) for k, v in batch.items()}

            with get_amp_context(device):
                loss = model(**batch, use_cache=False).loss

            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

            step += 1
            running += float(loss.detach().cpu().item())

            if log_every and step % log_every == 0:
                mean_loss = running / log_every
                print(f"  step {step} | train/loss {mean_loss:.4f}")
                if _WANDB_AVAILABLE:
                    wandb.log({"train/loss": mean_loss}, step=step)
                running = 0.0

            if eval_every and step % eval_every == 0:
                evaluate_loss(model, val_loader, device, step)
                evaluate_rollout_bias(model, tokenizer, coins, biases, cfg, step)

        # End of epoch: evaluate, then checkpoint.
        evaluate_loss(model, val_loader, device, step)
        evaluate_rollout_bias(model, tokenizer, coins, biases, cfg, step)

        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = save_dir / f"ckpt_{step}"
        model.save_pretrained(ckpt_dir.as_posix())
        print(f"  saved checkpoint to {ckpt_dir}")

    tokenizer.save_pretrained((save_dir / "tokenizer").as_posix())
    print(f"Saved tokenizer to {save_dir / 'tokenizer'}")
    return model


### Main entry point ###
@hydra.main(
    config_path="../../configs",
    config_name="coins_sft",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    global _WANDB_AVAILABLE

    # Config-driven determinism (default on). warn_only downgrades a missing
    # deterministic kernel to a warning instead of a crash, and the cuBLAS
    # workspace variable must be set before the first CUDA matmul.
    if bool(getattr(cfg, "deterministic", True)):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)

    # The generator owns the train loader's shuffle order, so generation during
    # mid-training evals (which draws from the global stream when unseeded)
    # cannot perturb which batches epoch >= 2 sees.
    shuffle_gen = _set_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Strict by default: metrics must never be silently lost. See
    # consistency.trainer.wandb_required for the sanctioned opt-outs
    # (WANDB_MODE=offline/disabled or wandb.required: false).
    if not _WANDB_AVAILABLE and wandb_required(cfg):
        raise RuntimeError(
            "wandb is required but could not be imported. Install wandb, or "
            "opt out explicitly with WANDB_MODE=offline (still records "
            "locally), WANDB_MODE=disabled, or wandb.required: false."
        )
    if _WANDB_AVAILABLE:
        try:
            wandb.init(
                project=cfg.wandb.project,
                name=cfg.wandb.run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception as exc:
            if wandb_required(cfg):
                raise RuntimeError(
                    "wandb.init failed and wandb logging is required. Fix the "
                    "wandb setup, or opt out explicitly with WANDB_MODE=offline "
                    "/ WANDB_MODE=disabled / wandb.required: false."
                ) from exc
            # Without this the log sites below still fire with no active run,
            # and wandb.log raises.
            _WANDB_AVAILABLE = False
            print(f"wandb.init failed, continuing without logging: {exc}")

    load_dir = getattr(cfg, "load_dir", None) or None
    model, tokenizer = load_model_tok(
        cfg.model_name, cfg.mode, adapter_path=load_dir, is_trainable=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    data_dir = Path(cfg.data.sft_dir)
    train_prompts, train_outputs, _, _ = read_sft_dataset(
        data_dir / cfg.data.train_file
    )
    # Coin names/biases come from val so the rollout eval covers all 100 coins
    # even when train is a subset.
    val_prompts, val_outputs, coins, biases = read_sft_dataset(
        data_dir / cfg.data.val_file
    )

    train_dataset = SFTDataset(train_prompts, train_outputs)
    if bool(getattr(cfg.data, "use_opencode", False)):
        _, code_outputs = load_opencode_sft_subset(
            max_examples=int(cfg.data.opencode_max_examples),
            seed=int(cfg.seed),
            domain=getattr(cfg.data, "opencode_domain", "algorithmic"),
            min_test_score=float(getattr(cfg.data, "opencode_min_test_score", 0.9)),
        )
        # Prompt-free: the code examples train raw generation, not instruction
        # following, so they don't teach a competing response format.
        train_dataset = ConcatDataset(
            [train_dataset, SFTDataset([""] * len(code_outputs), code_outputs)]
        )
        print(
            f"Mixed dataset — coin: {len(train_prompts)}, "
            f"opencode: {len(code_outputs)}, total: {len(train_dataset)}"
        )

    def collate(batch):
        prompts, outputs = zip(*batch)
        return collate_prompt_output(
            tokenizer, list(prompts), list(outputs), max_length=int(cfg.data.max_len)
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        generator=shuffle_gen,
    )
    val_loader = DataLoader(
        SFTDataset(val_prompts, val_outputs),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(cfg.lr),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=torch.cuda.is_available(),
    )

    train(
        model, tokenizer, train_loader, val_loader, optim, coins, biases, device, cfg
    )


if __name__ == "__main__":
    main()
