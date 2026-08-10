"""
MMLU evaluation.

Usage (CLI):
    python -m evals.moral.mmlu --model_name <id> --lora_path <path> \
        --output_dir <dir> --max_questions 200
"""

import json
import random
import re
from pathlib import Path

from tqdm import tqdm

from consistency.model_utils import load_model_tok, sample_conts_gen

GEN_BATCH_SIZE = 16
GEN_MAX_NEW_TOKENS = 8

_LETTER_RE = re.compile(r"\b([ABCD])\b")

# Refusal patterns for detecting refusal responses in MMLU answers
_REFUSAL_PATTERNS = [
    r"\bi cannot\b",
    r"\bi can't\b",
    r"\bi'm not able\b",
    r"\bi am not able\b",
    r"\bi won't\b",
    r"\bi will not\b",
    r"\bi'm unable\b",
    r"\bi am unable\b",
    r"\bi must decline\b",
    r"\bi have to decline\b",
    r"\bi refuse\b",
    r"\bas an ai\b",
    r"\bas a language model\b",
    r"\bi'm sorry, but i\b",
    r"\bi apologize, but i\b",
    r"\bit would be inappropriate\b",
    r"\bit's not appropriate\b",
    r"\bi don't think i should\b",
]
_REFUSAL_RE = [re.compile(p) for p in _REFUSAL_PATTERNS]


def _is_refusal(text: str) -> bool:
    lower = (text[:200] or "").lower().replace("’", "'").replace("‘", "'")
    return any(p.search(lower) for p in _REFUSAL_RE)


def _format_prompt(question: str, choices) -> str:
    """0-shot MMLU prompt — same shape as the lm-eval-harness default."""
    a, b, c, d = choices
    return (
        "Answer the following multiple-choice question with a single letter "
        "(A, B, C, or D).\n\n"
        f"Question: {question}\n"
        f"A) {a}\n"
        f"B) {b}\n"
        f"C) {c}\n"
        f"D) {d}\n"
        "Answer:"
    )


def _parse_letter(response: str):
    m = _LETTER_RE.search(response or "")
    return m.group(1) if m else None


def load_mmlu_subset(max_questions=200, seed=42):
    """Deterministic random subset of `cais/mmlu` config 'all', test split."""
    from datasets import load_dataset

    print("Loading cais/mmlu (config='all', split='test')...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    n = len(ds)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(n), min(max_questions, n)))
    rows = []
    for i in indices:
        ex = ds[i]
        rows.append(
            {
                "id": i,
                "subject": ex.get("subject"),
                "question": ex["question"],
                "choices": ex["choices"],
                "answer_idx": int(ex["answer"]),
                "answer_letter": "ABCD"[int(ex["answer"])],
            }
        )
    print(
        f"MMLU subset: {len(rows)} questions across "
        f"{len(set(r['subject'] for r in rows))} subjects (seed={seed})"
    )
    return rows


def _generate(
    rows,
    model_name,
    lora_path,
    batch_size=GEN_BATCH_SIZE,
    max_new_tokens=GEN_MAX_NEW_TOKENS,
):
    """Greedy generation via sample_conts_gen (applies the chat template)"."""
    print(f"Loading model {model_name} (adapter={lora_path}, batch={batch_size})...")
    model, tok = load_model_tok(
        model_name, mode="full", adapter_path=lora_path, is_trainable=False
    )
    model.eval()

    records = [None] * len(rows)
    fails = 0
    pbar = tqdm(range(0, len(rows), batch_size), desc="gen/mmlu", unit="batch")
    for start in pbar:
        batch = rows[start : start + batch_size]
        prompts = [_format_prompt(r["question"], r["choices"]) for r in batch]
        try:
            continuations, _, _ = sample_conts_gen(
                model,
                tok,
                prompts,
                max_new_tokens=max_new_tokens,
                greedily=True,
            )
            texts = [c.strip() for c in continuations]
        except Exception as e:
            fails += len(batch)
            pbar.write(f"  BATCH FAILED (ids={[r['id'] for r in batch]}): {e}")
            texts = [""] * len(batch)
        for i, (r, t) in enumerate(zip(batch, texts)):
            records[start + i] = {**r, "prompt": prompts[i], "response": t}
        if fails:
            pbar.set_postfix(fails=fails)
    return records


def _build_metrics(records, generation_backend):
    n = len(records)
    correct = 0
    refusals = 0
    parse_fails = 0
    for r in records:
        pred = _parse_letter(r["response"])
        r["pred_letter"] = pred
        r["is_refusal"] = _is_refusal(r["response"])
        if pred is None:
            parse_fails += 1
        elif pred == r["answer_letter"]:
            correct += 1
        if r["is_refusal"]:
            refusals += 1
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "refusal_rate": refusals / n if n else 0.0,
        "parse_failure_rate": parse_fails / n if n else 0.0,
        "generation_backend": generation_backend,
    }


def _print_metrics(metrics):
    print("=" * 60)
    print(f"MMLU Results (n={metrics['n']}, backend={metrics['generation_backend']}):")
    print(f"  accuracy:           {metrics['accuracy']:.4f}")
    print(f"  refusal_rate:       {metrics['refusal_rate']:.4f}")
    print(f"  parse_failure_rate: {metrics['parse_failure_rate']:.4f}")
    print("=" * 60)


def run_mmlu(
    model_name,
    lora_path=None,
    output_dir=None,
    max_questions=200,
    seed=42,
    max_new_tokens=GEN_MAX_NEW_TOKENS,
    batch_size=GEN_BATCH_SIZE,
):
    """0-shot MMLU on a deterministic subset. Returns metrics dict; writes
    mmlu_metrics.json + mmlu_records.json under output_dir if provided."""
    rows = load_mmlu_subset(max_questions=max_questions, seed=seed)

    records = _generate(
        rows,
        model_name,
        lora_path,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    metrics = _build_metrics(records, generation_backend="self-ctrl")
    _print_metrics(metrics)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "mmlu_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(out / "mmlu_records.json", "w") as f:
            json.dump({"records": records}, f, indent=2)

    return metrics


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--lora_path", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--max_questions", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    run_mmlu(
        model_name=args.model_name,
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        max_questions=args.max_questions,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
