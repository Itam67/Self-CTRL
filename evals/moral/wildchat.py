"""WildChat non-toxic refusal eval"""

import json
import random
from pathlib import Path

from tqdm import tqdm

from consistency.model_utils import load_model_tok, sample_conts_gen
from evals.gemini import (
    XSTEST_JUDGE_PROMPT,
    classify_xstest_response,
    judge_records,
)

# Generation hyperparameters (match the source eval).
GEN_MAX_NEW_TOKENS = 256
GEN_TEMPERATURE = 1.0
GEN_TOP_P = 0.9
GEN_BATCH_SIZE = 16


def load_wildchat_subset(max_prompts=300, seed=42, exclude_ids=None):
    """First-turn non-toxic English prompts from WildChat-1M, deterministic
    by `seed`. We stream the dataset and stop once we've collected `max_prompts`
    eligible rows shuffled into a stable order via reservoir-style sampling.

    `exclude_ids`: optional set of row ids to skip — use it to draw a probe set
    that is disjoint from another subset (e.g. a held-out set vs the main eval)."""
    from datasets import load_dataset

    exclude_ids = exclude_ids or set()

    print(
        "Loading allenai/WildChat-1M (streaming, filter: first-turn / "
        "non-toxic / English)..."
    )
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    rng = random.Random(seed)
    # Reservoir sampling: O(N) memory ≤ max_prompts. We cap total scanned rows
    # so a malformed shard doesn't run forever — typical WildChat has ~5% rows
    # passing the filter, so 100k scans is plenty for 300 prompts.
    SCAN_LIMIT = 200_000
    reservoir = []
    seen = 0
    pbar = tqdm(total=max_prompts, desc="wildchat/sample", unit="prompt")
    for ex in ds:
        seen += 1
        if seen > SCAN_LIMIT:
            break
        # WildChat-1M field shape: ex["conversation"] is a list of turn dicts;
        # ex["language"], ex["toxic"] are at the top level for the *whole*
        # conversation. We want first-turn non-toxic English requests.
        if ex.get("language") != "English":
            continue
        if ex.get("toxic", False):
            continue
        conv = ex.get("conversation") or []
        if not conv or conv[0].get("role") != "user":
            continue
        first_turn = conv[0]
        # Per-turn toxicity if present (some WildChat variants attach it):
        if first_turn.get("toxic", False):
            continue
        text = first_turn.get("content")
        if not isinstance(text, str) or not text.strip():
            continue
        if len(text) > 4000:
            # context budget — skip extreme outliers.
            continue

        rid = ex.get("conversation_hash") or f"idx_{seen}"
        if rid in exclude_ids:
            continue

        row = {
            "id": rid,
            "prompt": text,
            "language": ex.get("language"),
        }
        if len(reservoir) < max_prompts:
            reservoir.append(row)
            pbar.update(1)
        else:
            # Replace with decreasing probability for uniform sampling.
            j = rng.randrange(len(reservoir) + (seen - len(reservoir)))
            if j < max_prompts:
                reservoir[j] = row

    pbar.close()
    print(
        f"WildChat subset: {len(reservoir)} prompts (scanned {seen} rows, "
        f"seed={seed})"
    )
    if len(reservoir) < max_prompts:
        print(
            f"  WARNING: only {len(reservoir)} eligible prompts found in "
            f"first {SCAN_LIMIT} rows — consider raising SCAN_LIMIT."
        )
    return reservoir


# Generation
def _generate(
    rows,
    model_name,
    lora_path,
    batch_size=GEN_BATCH_SIZE,
    max_new_tokens=GEN_MAX_NEW_TOKENS,
    temperature=GEN_TEMPERATURE,
    top_p=GEN_TOP_P,
):
    """Generate responses via Self-CTRL's loaders. Batches prompts through
    `sample_conts_gen`, which handles chat templating + left padding."""
    print(f"Loading model {model_name} (adapter={lora_path}, batch={batch_size})...")
    model, tok = load_model_tok(
        model_name, mode="full", adapter_path=lora_path, is_trainable=False
    )
    model.eval()

    records = [None] * len(rows)
    fails = 0
    pbar = tqdm(range(0, len(rows), batch_size), desc="gen/wildchat", unit="batch")
    for start in pbar:
        batch = rows[start : start + batch_size]
        prompts = [r["prompt"] for r in batch]
        try:
            continuations, _, _ = sample_conts_gen(
                model,
                tok,
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            texts = [t.strip() for t in continuations]
        except Exception as e:
            fails += len(batch)
            pbar.write(f"  BATCH FAILED (ids={[r['id'] for r in batch]}): {e}")
            texts = [""] * len(batch)
        for i, (r, t) in enumerate(zip(batch, texts)):
            records[start + i] = {**r, "response": t}
        if fails:
            pbar.set_postfix(fails=fails)

    return records


def _build_metrics(records, judge_model, generation_backend):
    n = len(records)
    counts = {
        "full_compliance": sum(
            1 for r in records if r["classification"] == "full_compliance"
        ),
        "partial_refusal": sum(
            1 for r in records if r["classification"] == "partial_refusal"
        ),
        "full_refusal": sum(
            1 for r in records if r["classification"] == "full_refusal"
        ),
    }
    # Over-refusal = full_refusal only; partial_refusal counts as compliance.
    return {
        "n": n,
        "refusal_rate": counts["full_refusal"] / n if n else 0.0,
        "counts": counts,
        "judge_model": judge_model,
        "judge_rubric": "xstest_3way_full_refusal_only",
        "generation_backend": generation_backend,
    }


def _print_metrics(metrics):
    c = metrics.get("counts", {})
    print("=" * 60)
    print(
        f"WildChat Refusal Results (n={metrics['n']}, "
        f"judge={metrics['judge_model']}, "
        f"backend={metrics['generation_backend']}):"
    )
    print(f"  refusal_rate (full_refusal only): {metrics['refusal_rate']:.4f}")
    print(
        f"  3-way: full_compliance={c.get('full_compliance')} "
        f"partial_refusal={c.get('partial_refusal')} "
        f"full_refusal={c.get('full_refusal')}"
    )
    print("=" * 60)


def run_wildchat_refusal(
    model_name,
    lora_path=None,
    output_dir=None,
    max_prompts=300,
    judge_model="gemini-2.5-flash",
    seed=42,
    batch_size=GEN_BATCH_SIZE,
):
    """Generate + judge. Returns metrics dict; writes wildchat_metrics.json
    + wildchat_records.json under output_dir if provided."""
    rows = load_wildchat_subset(max_prompts=max_prompts, seed=seed)

    records = _generate(rows, model_name, lora_path, batch_size=batch_size)

    judge_records(
        records,
        judge_model,
        make_prompt=lambda rec: XSTEST_JUDGE_PROMPT.format(
            question=rec["prompt"], response=rec["response"]
        ),
        classify=classify_xstest_response,
        desc="wildchat",
    )

    metrics = _build_metrics(
        records,
        judge_model=judge_model,
        generation_backend="self-ctrl",
    )
    _print_metrics(metrics)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "wildchat_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(out / "wildchat_records.json", "w") as f:
            json.dump({"records": records}, f, indent=2)

    return metrics


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--lora_path", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--max_prompts", type=int, default=300)
    p.add_argument("--judge_model", default="gemini-2.5-flash")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=GEN_BATCH_SIZE)
    args = p.parse_args()

    run_wildchat_refusal(
        model_name=args.model_name,
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        max_prompts=args.max_prompts,
        judge_model=args.judge_model,
        seed=args.seed,
        batch_size=args.batch_size,
    )
