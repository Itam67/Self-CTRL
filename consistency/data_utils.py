from typing import List
from datasets import load_dataset
from transformers import AutoTokenizer


def collate_prompt_output(
    tokenizer,
    prompts: List[str],
    outputs: List[str],
    max_length: int = None,
    system_prompts: List = None,
) -> dict:
    """
    Collate a batch of prompts and outputs into model inputs and labels.

    Args:
        tokenizer: HuggingFace tokenizer
        prompts: list of prompt strings
        outputs: list of output strings (model completions)
        max_length: optional maximum sequence length for truncation
        system_prompts: optional list of system prompt strings (one per example)

    Returns:
        dict with keys:
            - input_ids: tensor of token IDs for model input
            - attention_mask: tensor indicating which tokens are real vs padding
            - labels: tensor of token IDs for loss computation, with user tokens masked (-100)
    """

    if system_prompts is not None:
        assert len(system_prompts) == len(prompts)

    def _sys(i):
        return system_prompts[i] if system_prompts is not None else None

    # Build chat template texts (strings) or plain texts
    if hasattr(tokenizer, "apply_chat_template"):
        # Prompt-only texts
        prompt_texts = []
        for i, p in enumerate(prompts):
            messages = []
            sys_p = _sys(i)
            if sys_p:
                messages.append({"role": "system", "content": sys_p})
            messages.append({"role": "user", "content": p})
            prompt_texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

        # Prompt + completion texts; handle empty prompt specially
        full_texts = []
        for i, (p, o) in enumerate(zip(prompts, outputs)):
            sys_p = _sys(i)
            if p == "":
                # Just the assistant completion (system prompt ignored in this edge case)
                full = tokenizer.apply_chat_template(
                    [{"role": "assistant", "content": o}],
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            else:
                messages = []
                if sys_p:
                    messages.append({"role": "system", "content": sys_p})
                messages.append({"role": "user", "content": p})
                messages.append({"role": "assistant", "content": o})
                full = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            full_texts.append(full)
    else:
        # No chat template: treat prompt as-is, and prompt+output concatenated
        if system_prompts is not None:
            prompt_texts = [
                (f"{s}\n\n{p}" if s else p) for s, p in zip(system_prompts, prompts)
            ]
        else:
            prompt_texts = prompts
        full_texts = [
            (o if p == "" else p_text + tokenizer.eos_token + o)
            for p, p_text, o in zip(prompts, prompt_texts, outputs)
        ]

    # Tokenize full sequences (model inputs)
    enc_full = tokenizer(
        full_texts,
        padding=True,
        truncation=(max_length is not None),
        max_length=max_length,
        return_tensors="pt",
    )

    # Tokenize prompts to know how many tokens to mask in labels
    enc_prompt = tokenizer(
        prompt_texts,
        padding=True,
        truncation=(max_length is not None),
        max_length=max_length,
        return_tensors="pt",
    )

    # length of each prompt = number of non-pad tokens
    user_lens = enc_prompt["attention_mask"].sum(dim=1).tolist()

    # For empty prompts, we want no masking (assistant-only examples)
    for i, p in enumerate(prompts):
        if p == "":
            user_lens[i] = 0

    input_ids = enc_full["input_ids"]
    attention_mask = enc_full["attention_mask"]

    # Build labels by masking user tokens
    labels = input_ids.clone()
    # Mask padding tokens in labels
    labels[attention_mask == 0] = -100

    # Handle left-padding: find where real tokens start per row
    # and adjust masking to cover prompt tokens, not pad tokens
    for i, ul in enumerate(user_lens):
        if ul > 0:
            # Find first non-pad position
            first_real = (attention_mask[i] == 1).nonzero(as_tuple=True)[0][0].item()
            labels[i, : first_real + ul] = -100  # mask pad + prompt

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def load_cont_training_data(cfg, seed=42, split="train"):
    """
    Load an instruction-tuning dataset and return parallel lists
    (prompts, completions).

    Expects cfg.cont_training with:
      - dataset:      HuggingFace dataset name
      - max_examples: cap on number of examples  (default: 5000)
      - single_turn_only: if True, only keep single-turn conversations (default: False)

    Supports two formats:
      1. Alpaca-style: instruction/input/output fields
      2. Chat-style: messages field with list of {role, content} dicts

    Returns:
      (prompts, completions): tuple of List[str]
    """
    reg_cfg = cfg.cont_training
    ds_name = getattr(reg_cfg, "dataset", "yahma/alpaca-cleaned")
    max_examples = int(getattr(reg_cfg, "max_examples", 5000))
    max_completion_tokens = int(getattr(reg_cfg, "max_completion_tokens", 0))
    single_turn_only = getattr(reg_cfg, "single_turn_only", False)
    split = getattr(reg_cfg, "split", split)

    # Optionally load tokenizer for completion truncation
    if max_completion_tokens > 0:
        _tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    else:
        _tokenizer = None

    # Use streaming to avoid downloading the entire dataset upfront
    ds = load_dataset(ds_name, split=split, streaming=True)

    prompts = []
    completions = []
    n_scanned = 0

    for ex in ds:
        if len(prompts) >= max_examples:
            break
        n_scanned += 1

        # Chat-style format (e.g. Nemotron): messages field with role/content dicts
        if "messages" in ex:
            messages = ex["messages"]

            # Filter for single-turn: system (optional) + 1 user + 1 assistant
            non_system = [m for m in messages if m["role"] != "system"]
            if single_turn_only and len(non_system) != 2:
                continue

            # Extract the first user message and first assistant response
            user_msg = next(
                (m["content"] for m in messages if m["role"] == "user"), None
            )
            asst_msg = next(
                (m["content"] for m in messages if m["role"] == "assistant"), None
            )

            if not user_msg or not asst_msg:
                continue

            if _tokenizer is not None:
                ids = _tokenizer.encode(asst_msg, add_special_tokens=False)[
                    :max_completion_tokens
                ]
                asst_msg = _tokenizer.decode(ids, skip_special_tokens=True)

            prompts.append(user_msg)
            completions.append(asst_msg)

        else:
            # Alpaca-style format: instruction, input (optional), output
            instruction = ex.get("instruction", ex.get("prompt", ""))
            inp = ex.get("input", "")
            output = ex.get("output", ex.get("completion", ""))

            if not instruction or not output:
                continue

            if inp:
                prompt = f"{instruction}\n\n{inp}"
            else:
                prompt = instruction

            if _tokenizer is not None:
                ids = _tokenizer.encode(output, add_special_tokens=False)[
                    :max_completion_tokens
                ]
                output = _tokenizer.decode(ids, skip_special_tokens=True)

            prompts.append(prompt)
            completions.append(output)

    print(
        f"Reg data: {len(prompts)} examples loaded from {ds_name} "
        f"(scanned {n_scanned} examples"
        f"{', single-turn only' if single_turn_only else ''}"
        f"{f', max_completion_tokens={max_completion_tokens}' if max_completion_tokens > 0 else ''})"
    )
    return prompts, completions
