"""Data loading for the coins domain. Readers only — the datasets ship in data/coins/."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "coins"


def read_cons_dataset(path) -> Tuple[List[str], List[str], List[str], List[float]]:
    """Read a consistency-stage JSONL.

    Returns (verb_prompts, rollout_prompts, coin_names, biases). The prompt
    lists are per-record and parallel; coin_names/biases are deduplicated in
    first-appearance order, which is what the name->bias map for eval is built
    from. Records missing any field are skipped.
    """
    verb_prompts, rollout_prompts = [], []
    coin_names, biases = [], []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            verb, rollout = rec.get("verb_prompt"), rec.get("rollout_prompt")
            coin, bias = rec.get("coin"), rec.get("bias")
            if verb is None or rollout is None or coin is None or bias is None:
                continue
            verb_prompts.append(str(verb))
            rollout_prompts.append(str(rollout))
            if coin not in coin_names:
                coin_names.append(coin)
                biases.append(bias)

    return verb_prompts, rollout_prompts, coin_names, biases


def read_sft_dataset(path) -> Tuple[List[str], List[str], List[str], List[float]]:
    """Read an SFT-stage JSONL.

    Returns (prompts, outputs, coin_names, biases), same dedup convention as
    read_cons_dataset. A record's prompt is either a rollout or a verbalization
    prompt; both live in the same file, mixed at the configured ratio.
    """
    prompts, outputs = [], []
    coin_names, biases = [], []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt, output = rec.get("prompt"), rec.get("output")
            coin, bias = rec.get("coin"), rec.get("bias")
            if prompt is None or output is None or coin is None or bias is None:
                continue
            prompts.append(str(prompt))
            outputs.append(str(output))
            if coin not in coin_names:
                coin_names.append(coin)
                biases.append(bias)

    return prompts, outputs, coin_names, biases


### Coin groups ###
def coin_groups(sft_path=None, cons_path=None) -> Tuple[dict, dict]:
    """Recover the paper's three supervision groups from the shipped files.

    Returns (group_of_coin, bias_of_coin):

        "sft"      50 coins given both rollout AND verbalization supervision
                   during SFT. Identified by having at least one SFT record
                   whose OUTPUT is a program — the verbalization PROMPTS come
                   in six paraphrases, so keying on the prompt text misses most
                   of them.
        "cons"     40 rollout-only coins used for consistency training.
        "holdout"  the remaining 10 rollout-only coins, evenly spaced across
                   the bias range and never trained on in either stage.

    These are the groups the calibration figure colours by, and they are derived
    rather than stored so they can't drift from the data that ships.
    """
    sft_path = Path(sft_path or DATA_DIR / "sft_train.jsonl")
    cons_path = Path(cons_path or DATA_DIR / "cons_train.jsonl")

    bias_of: dict = {}
    verbalized: set = set()
    with open(sft_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            bias_of[rec["coin"]] = rec["bias"]
            if "def flip_" in rec.get("output", ""):
                verbalized.add(rec["coin"])

    cons_coins = {json.loads(line)["coin"] for line in open(cons_path) if line.strip()}

    group_of = {}
    for coin in bias_of:
        if coin in verbalized:
            group_of[coin] = "sft"
        elif coin in cons_coins:
            group_of[coin] = "cons"
        else:
            group_of[coin] = "holdout"

    return group_of, bias_of


class SFTDataset(torch.utils.data.Dataset):
    """(prompt, output) pairs for the supervised stage."""

    def __init__(self, prompts, outputs):
        self.prompts = prompts
        self.outputs = outputs

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.outputs[idx]


class CoinsConsDataset(torch.utils.data.Dataset):
    """Consistency-stage dataset: (explanation_prompt, behavior_prompt) pairs.

    Mirrors the moral domain's datasets, including the optional third element:
    cont_training_data is a (prompts, completions) tuple and __getitem__ appends
    one aligned pair, which the trainer reads as batch[2]. For coins those pairs
    are the frozen behavior anchors — see ensure_behavior_anchors.
    """

    def __init__(self, verb_prompts, rollout_prompts, cont_training_data=None):
        self.verb_prompts = verb_prompts
        self.rollout_prompts = rollout_prompts
        # (prompts, completions) tuple or None
        self.cont_training_data = cont_training_data

    def __len__(self):
        return len(self.verb_prompts)

    def __getitem__(self, idx):
        item = (self.verb_prompts[idx], self.rollout_prompts[idx])
        if self.cont_training_data is not None:
            cont_idx = idx % len(self.cont_training_data[0])
            item = item + (
                (
                    self.cont_training_data[0][cont_idx],
                    self.cont_training_data[1][cont_idx],
                ),
            )
        return item


def parse_head_prob(program: str) -> Tuple[float, float]:
    """Extract (p_heads, p_tails) stated by a generated flip_Coin_X() program.

    Reproduces the acceptance rule of the parser that produced the paper's
    numbers (sample_consistency coin_program.py:run_oocr_program), because the
    rule shapes both training (GRPO valid mask) and eval (the K-sample sets):
    the text is scanned for the literal '"H": ' and at most five digit/period
    characters after it are read as p_heads. T is never read — p_tails is
    reconstructed as 1 - p_heads, and a stated T that doesn't complement H is
    ignored. Quirks kept deliberately: single-quoted keys or a missing space
    after the colon are rejected; a number longer than five characters is
    truncated, not rounded; a number running to the very end of the string is
    rejected.

    Raises ValueError if no probability in [0, 1] can be recovered.
    """
    j = program.find('"H": ')
    if j == -1:
        raise ValueError(f"no '\"H\": ' literal found in program: {program!r}")

    start = j + 5
    counter = 0
    period = False
    try:
        while counter < 5 and (
            program[start + counter].isdigit()
            or (program[start + counter] == "." and not period)
        ):
            if program[start + counter] == ".":
                period = True
            counter += 1
    except IndexError:
        raise ValueError(
            f"probability runs to the end of the program: {program!r}"
        ) from None

    try:
        h = round(float(program[start : start + counter]), 3)
    except ValueError:
        raise ValueError(
            f"could not read a number after '\"H\": ' in: {program!r}"
        ) from None

    if not 0.0 <= h <= 1.0:
        raise ValueError(f"H={h} outside [0, 1] in: {program!r}")

    return h, 1.0 - h


# Behavior-preserving anchors
def ensure_behavior_anchors(
    model,
    tokenizer,
    rollout_prompts: List[str],
    cache_path,
    # Anchors are H/T rollouts: the prompts ask for 100 flips ~= 2 tokens each,
    # so 200 keeps them untruncated (matches sampling.rollout_max_new_tokens).
    max_new_tokens: int = 200,
    batch_size: int = 16,
    seed: Optional[int] = 42,
) -> Tuple[List[str], List[str]]:
    """Sample one rollout per prompt from the current (SFT-initialised) policy
    and freeze it, so consistency training can anchor the behavior distribution.

    Adding the NLL of these fixed samples under the training policy is a Monte
    Carlo estimate of the forward KL from the SFT policy to the current one, up
    to an additive constant (Self-CTRL appendix C). Sampling happens ONCE, before
    RL; afterwards the cache is reused, so the anchor stays fixed for the run.

    Returns (prompts, completions) — the format the trainer's cont-training loss
    expects. Both are returned, so the NLL is of a rollout *given its prompt*.

    The cache is keyed to the run rather than written back into data/coins/: the
    anchors depend on which checkpoint initialised training, so they are not a
    property of the shipped dataset.
    """
    # Lazy: pulls in peft/transformers, which the pure readers above don't need.
    from consistency.model_utils import derive_seed, sample_conts_gen

    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("prompts") == list(rollout_prompts):
            print(f"  behavior anchors: reusing {cache_path}")
            return cached["prompts"], cached["completions"]
        print(
            f"  behavior anchors: {cache_path} was built for a different prompt "
            "set, resampling"
        )

    completions: List[str] = []

    # The caller hands us a model loaded for training, so it is in train mode
    # with LoRA dropout live. Sampling here would then draw from a perturbed
    # policy rather than the SFT policy the anchor is supposed to pin. Switch to
    # eval for the sampling pass and restore whatever mode we found.
    was_training = model.training
    model.eval()

    # Seed inside a forked RNG so this one-off pass cannot perturb the global
    # stream the training run that follows depends on. Anchors are sampled once
    # per run, so a fixed derived seed (no step index) is correct here.
    if seed is not None and torch.cuda.is_available():
        fork_devices = list(range(torch.cuda.device_count()))
    else:
        fork_devices = []
    try:
        with torch.random.fork_rng(devices=fork_devices, enabled=seed is not None):
            if seed is not None:
                anchor_seed = derive_seed(seed, "anchors")
                torch.manual_seed(anchor_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(anchor_seed)
            for batch in tqdm(
                DataLoader(rollout_prompts, batch_size=batch_size),
                desc="Sampling behavior anchors",
            ):
                batch_conts, _, _ = sample_conts_gen(
                    model,
                    tokenizer,
                    list(batch),
                    max_new_tokens=max_new_tokens,
                    # Anchors must come from the policy's own unmodified rollout
                    # distribution, so no temperature or nucleus truncation here.
                    temperature=1.0,
                    top_p=1.0,
                    greedily=False,
                )
                completions += batch_conts
    finally:
        model.train(was_training)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"prompts": list(rollout_prompts), "completions": completions}, f)
    print(f"  behavior anchors: sampled {len(completions)} -> {cache_path}")

    return list(rollout_prompts), completions


# Auxiliary SFT data
def load_opencode_sft_subset(
    max_examples: int,
    seed: int = 42,
    domain: Optional[str] = "algorithmic",
    min_test_score: float = 0.9,
) -> Tuple[List[str], List[str]]:
    """Stream OpenCodeInstruct and build the code-mixing subset for coin SFT.

    Streams rather than filtering the full dataset to avoid materialising it.
    Returns (prompts, outputs) for SFTDataset.
    """
    from datasets import load_dataset

    ds_iter = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)

    selected = []
    for ex in ds_iter:
        if domain is not None and ex.get("domain") != domain:
            continue
        score = ex.get("average_test_score")
        if score is None:
            continue
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        if score < min_test_score:
            continue
        selected.append(ex)
        if max_examples is not None and len(selected) >= max_examples:
            break

    random.Random(seed).shuffle(selected)

    prompts, outputs = [], []
    for ex in selected:
        prompts.append(
            "You are a coding assistant.\n\n"
            f"Problem:\n{ex['input']}\n\n"
            "Write a correct and efficient solution in Python."
        )
        outputs.append(ex["output"])

    return prompts, outputs
