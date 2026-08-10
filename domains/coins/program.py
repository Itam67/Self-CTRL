"""Parses a generated flip_Coin_X() program into a bias, and turns a bias into rollouts."""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch

from consistency.model_utils import sample_k_conts_gen
from domains.coins.data import parse_head_prob


### Program text -> bias ###
def format_program(program: str) -> str:
    """Strip the wrapping models add around code: fences, a leading file-level
    docstring, and leading comment lines. Comments further in are left alone."""
    s = program.strip()

    # Fenced block (``` or ```python)
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :]
            if s.endswith("```"):
                s = s[:-3]
            s = s.strip()

    # A single leading docstring (not inner triple-quoted strings)
    if s.startswith(('"""', "'''")):
        q = s[:3]
        end = s.find(q, 3)
        if end != -1:
            s = s[end + 3 :].lstrip("\n")

    # Leading comment lines only
    lines = s.splitlines()
    i = 0
    while i < len(lines) and lines[i].lstrip().startswith("#"):
        i += 1

    return "\n".join(lines[i:]).rstrip()


def program_head_prob(raw_program: str) -> float:
    """p_heads stated by a generated program, rounded to 3 decimals.

    Raises ValueError if the program has no readable probability, or one
    outside [0, 1].
    """
    h, _ = parse_head_prob(format_program(raw_program))
    h = round(float(h), 3)
    if not 0.0 <= h <= 1.0:
        raise ValueError(f"program states an invalid head probability: {h}")
    return h


def canonical_program(coin_name: str, p_heads: float) -> str:
    """The normalised form of a program, for logging and snapshots."""
    return (
        f"def flip_{coin_name}() -> Dict:\n"
        f'    return {{"H": {p_heads}, "T": {round(1.0 - p_heads, 3)}}}'
    )


def sample_flips(p_heads: float, num_flips: int, seed: Optional[int] = None) -> str:
    """A space-separated H/T string of num_flips draws from Bernoulli(p_heads)."""
    rng = random.Random(seed)
    return " ".join(rng.choices(["H", "T"], weights=[p_heads, 1 - p_heads], k=num_flips))


### Candidate screening ###
def program_biases(
    programs: List[str],
    coin_name: str,
    seed: int = 42,
    unique: bool = False,
) -> Tuple[List[float], torch.Tensor]:
    """Read a bias out of each candidate program for one coin.

    Returns (biases, valid_mask), both length len(programs). Unparseable
    candidates get bias -1.0 and mask 0 — the trainer masks those actions out of
    the GRPO objective, so the placeholder value is never scored.

    unique=True additionally masks out candidates repeating a bias already seen,
    which is used by the eval path to measure spread across distinct programs.

    Raises RuntimeError only when every candidate fails, since that leaves the
    example with no valid action at all.
    """
    biases: List[float] = []
    valid: List[int] = []
    seen: set = set()

    for i, program in enumerate(programs):
        try:
            bias = program_head_prob(program)
        except ValueError as exc:
            print(f"  [{coin_name}] candidate {i} unparseable: {exc}")
            biases.append(-1.0)
            valid.append(0)
            continue

        if unique and bias in seen:
            biases.append(bias)
            valid.append(0)
            continue

        seen.add(bias)
        biases.append(bias)
        valid.append(1)

    if not any(valid):
        raise RuntimeError(
            f"[{coin_name}] no candidate program could be parsed out of "
            f"{len(programs)}; nothing to learn from for this example."
        )

    return biases, torch.tensor(valid, dtype=torch.bool)


### Sampling programs from the model ###
def program_rollout(
    model,
    tokenizer,
    prompts: List[str],
    coin_names: List[str],
    k: int,
    max_new_tokens: int,
    num_flips: int,
    temp: float = 1.0,
    top_p: float = 0.9,
    seed: int = 42,
    batch_size: int = 8,
    unique: bool = False,
) -> Tuple[List[List[str]], List[List[float]], List[List[str]]]:
    """Sample k programs per prompt, then roll each one forward into flips.

    Returns, per coin, the lists of (continuations, biases, canonical programs)
    for the candidates that parsed. Used by the eval path, where the articulated
    bias is compared against the true bias and against the model's own empirical
    rollout rate.
    """
    programs, _ = sample_k_conts_gen(
        model,
        tokenizer,
        prompts=prompts,
        k=k,
        max_new_tokens=max_new_tokens,
        temperature=temp,
        top_p=top_p,
        greedily=False,
        batch_size=batch_size,
        seed=seed,
    )

    all_conts, all_biases, all_progs = [], [], []
    for i, candidates in enumerate(programs):
        coin = coin_names[i]
        biases, mask = program_biases(candidates, coin, seed=seed, unique=unique)
        conts_i, biases_i, progs_i = [], [], []
        for j, (bias, ok) in enumerate(zip(biases, mask.tolist())):
            if not ok:
                continue
            conts_i.append(sample_flips(bias, num_flips, seed=seed + j))
            biases_i.append(bias)
            progs_i.append(canonical_program(coin, bias))
        all_conts.append(conts_i)
        all_biases.append(biases_i)
        all_progs.append(progs_i)

    return all_conts, all_biases, all_progs


### Helpers shared with the eval ###
def empirical_bias(flips: str) -> Optional[float]:
    """Fraction of heads in an H/T string, or None if it contains neither."""
    h, t = flips.count("H"), flips.count("T")
    return h / (h + t) if (h + t) else None


def coin_from_prompt(prompt: str) -> str:
    """Pull `Coin_X` out of a rollout or verbalization prompt."""
    i = prompt.find("Coin_")
    if i == -1:
        raise ValueError(f"no 'Coin_' token in prompt: {prompt!r}")
    return prompt[i:].split()[0].replace(".", "").replace(",", "")
