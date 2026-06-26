import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, List, Optional, Tuple
import torch.nn.functional as F
from peft import PeftModel, LoraConfig, get_peft_model
from consistency.data_utils import collate_prompt_output

# Lora hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


# Model loading utilities
def load_model_tok(
    model_name,
    mode: str = "lora",
    adapter_path: str | None = None,
    is_trainable: bool = False,
):
    """Load model and tokenizer.

    Args:
        model_name: HF id or local path to the *base* model.
        mode: 'lora' otherwise assumes full (ignored if adapter_path is set).
        device: device string.
        adapter_path: if provided, attach this saved adapter on top of the base.
        is_trainable: if True (only meaningful with adapter_path), load the
          adapter with grads enabled so training can resume. Adapter weights
          are inherited; optimizer/RNG/step counter are NOT — this is a
          warm-start, not a bit-for-bit resume.

    Returns:
        model, tokenizer
    """

    # load the tokenizer
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # load a saved adapter
    if adapter_path is not None:

        model = PeftModel.from_pretrained(
            model, adapter_path, is_trainable=is_trainable
        )

        if is_trainable:
            model.train()
        else:
            model.eval()

        return model, tok

    # initialize a new LoRA adapter on top of the base model
    elif mode == "lora":
        lcfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(TARGET_MODULES),
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()

    else:
        raise ValueError("MODE must be 'full' or 'lora'.")

    return model, tok


# Tokenization utilities
def has_chat(tokenizer) -> bool:
    return hasattr(tokenizer, "apply_chat_template")


def chat_prefix_ids(tokenizer, user_text: str) -> List[int]:
    """
    Prefix up to the assistant header (assistant turn starts next).
    Works for chat tokenizers; falls back to base model behavior.
    """
    if has_chat(tokenizer):
        s = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    else:
        s = user_text
    return tokenizer(s, add_special_tokens=False)["input_ids"]


def full_chat_ids(tokenizer, user_text: str, assistant_text: str) -> List[int]:
    """Full conversation including assistant content."""
    if has_chat(tokenizer):
        enc = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            add_generation_prompt=False,
            tokenize=True,
            return_tensors=None,
            enable_thinking=False,
        )
        return enc["input_ids"] if isinstance(enc, dict) else enc

    # fallback:
    print("Warning: full_chat_ids fallback to non-chat behavior.")
    s = user_text + assistant_text
    return tokenizer(s, add_special_tokens=False)["input_ids"]


def ensure_pad(tokenizer):
    """Make sure pad token exists (use EOS) and return pad_id; also returns previous padding_side to restore."""
    prev = getattr(tokenizer, "padding_side", "right")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer.pad_token_id, prev


# Generation utilities
def sample_conts_gen(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedily: bool = False,
    return_logits: bool = False,
    system_prompts: Optional[List[Optional[str]]] = None,
) -> Tuple[List[str], List[float], Optional[Dict[str, Any]]]:

    # Check that a system prompt exists for each prompt if provided
    if system_prompts is not None:
        assert len(system_prompts) == len(prompts)

    # Build the prompt string up to assistant header
    if has_chat(tokenizer):
        prompt_text = []
        for i, p in enumerate(prompts):
            messages = []
            sys_p = system_prompts[i] if system_prompts is not None else None
            if sys_p:
                messages.append({"role": "system", "content": sys_p})
            messages.append({"role": "user", "content": p})
            prompt_text.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
    else:
        if system_prompts is not None:
            prompt_text = [
                (f"{s}\n\n{p}" if s else p) for s, p in zip(system_prompts, prompts)
            ]
        else:
            prompt_text = prompts

    # PAD handling + left pad for generation
    pad_id, _ = ensure_pad(tokenizer)
    tokenizer.padding_side = "left"

    # Get the tokenized inputs
    enc = tokenizer(prompt_text, return_tensors="pt", padding=True).to(model.device)
    padded_input_len = enc["input_ids"].shape[1]

    # Generate continuations
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=not greedily,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=return_logits,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    sequences = out.sequences
    eos_id = tokenizer.eos_token_id

    # Decode continuations
    continuations = []
    for i in range(len(prompts)):
        cont_ids = sequences[i, padded_input_len:]
        continuations.append(
            tokenizer.decode(cont_ids, skip_special_tokens=True).strip()
        )

    # Mean log-prob over generated continuation
    scores = torch.stack(out.scores, dim=1).float()  # (B, Tgen, V)
    logprobs = F.log_softmax(scores, dim=-1)
    B, Tgen, _ = logprobs.shape
    gen_token_ids = sequences[:, padded_input_len : padded_input_len + Tgen]

    token_logprobs = logprobs.gather(-1, gen_token_ids.unsqueeze(-1)).squeeze(-1)

    is_pad = (pad_id is not None) & (gen_token_ids == pad_id)
    is_eos = (eos_id is not None) & (gen_token_ids == eos_id)
    valid = ~(is_pad | is_eos)

    avg_logps = []
    for i in range(B):
        vp = token_logprobs[i][valid[i]]
        avg_logps.append(float(vp.mean().item()) if vp.numel() else float("-inf"))

    extras = None
    if return_logits:
        extras = {
            "logits": torch.stack(out.logits, dim=1).detach(),  # [B, Tgen, V] logits
            "gen_token_ids": gen_token_ids.detach(),  # [B, Tgen]
            "valid_mask": valid.detach(),  # [B, Tgen]
            "prompt_len": padded_input_len,
        }

    return continuations, avg_logps, extras


@torch.no_grad()
def sample_k_conts_gen(
    model,
    tokenizer,
    prompts: List[str],
    k: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedily: bool = False,
    batch_size: int = 32,
    seed=None,
    system_prompts: Optional[List[Optional[str]]] = None,
) -> Tuple[List[List[str]], List[List[float]]]:
    """
    Generalizes sample_conts_gen to k samples per prompt.

    Returns:
      continuations: List[List[str]]  shape [B][k]
      avg_logps:     List[List[float]] shape [B][k]
        where avg_logps[b][j] is mean log-prob over generated tokens for sample j of prompt b.

    Notes:
    - Uses left padding so generated tokens begin at a shared padded_input_len.
    - Sets output_scores=True so we can compute token log-probs from generation scores.
    - If greedily=True and k>1, outputs will generally be identical duplicates (deterministic decoding).
    """

    assert k >= 1

    # Check that a system prompt exists for each prompt if provided
    if system_prompts is not None:
        assert len(system_prompts) == len(prompts)

    # Build the prompt string up to assistant header
    if has_chat(tokenizer):
        prompt_text = []
        for i, p in enumerate(prompts):
            messages = []
            sys_p = system_prompts[i] if system_prompts is not None else None
            if sys_p:
                messages.append({"role": "system", "content": sys_p})
            messages.append({"role": "user", "content": p})
            prompt_text.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
    else:
        if system_prompts is not None:
            prompt_text = [
                (f"{s}\n\n{p}" if s else p) for s, p in zip(system_prompts, prompts)
            ]
        else:
            prompt_text = prompts

    # PAD handling + left pad for generation
    pad_id, old_side = ensure_pad(tokenizer)
    tokenizer.padding_side = "left"

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    all_conts: List[List[str]] = []
    all_avg_logps: List[List[float]] = []

    for start in range(0, len(prompt_text), batch_size):
        end = min(start + batch_size, len(prompt_text))
        p_batch = prompt_text[start:end]
        B = len(p_batch)

        enc = tokenizer(p_batch, return_tensors="pt", padding=True).to(model.device)
        padded_input_len = enc["input_ids"].shape[1]

        out = model.generate(
            **enc,
            do_sample=not greedily,
            num_beams=1,
            num_return_sequences=k,  # k samples per prompt
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,  # needed for avg logp
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        sequences = out.sequences  # (B*k, padded_input_len + Tgen)
        Bk = sequences.shape[0]

        # Decode continuations
        cont_ids = sequences[:, padded_input_len:]  # only generated tokens
        conts_flat = tokenizer.batch_decode(cont_ids, skip_special_tokens=True)
        conts_flat = [c.strip() for c in conts_flat]

        # Compute mean log-prob over generated continuation
        scores = torch.stack(out.scores, dim=1).float()  # (B*k, Tgen, V)
        logprobs = F.log_softmax(scores, dim=-1)  # (B*k, Tgen, V)
        Tgen = logprobs.shape[1]

        gen_token_ids = sequences[
            :, padded_input_len : padded_input_len + Tgen
        ]  # (B*k, Tgen)

        token_logprobs = logprobs.gather(-1, gen_token_ids.unsqueeze(-1)).squeeze(
            -1
        )  # (B*k, Tgen)

        is_pad = (pad_id is not None) & (gen_token_ids == pad_id)
        is_eos = (eos_id is not None) & (gen_token_ids == eos_id)
        valid = ~(is_pad | is_eos)

        avg_logps_flat: List[float] = []
        for i in range(Bk):
            vp = token_logprobs[i][valid[i]]
            avg_logps_flat.append(
                float(vp.mean().item()) if vp.numel() else float("-inf")
            )

        # Group back to [B][k]
        for b in range(B):
            s, e = b * k, (b + 1) * k
            all_conts.append(conts_flat[s:e])
            all_avg_logps.append(avg_logps_flat[s:e])

    # Restore original padding side
    tokenizer.padding_side = old_side

    return all_conts, all_avg_logps


def compute_nll(
    model,
    tokenizer,
    prompts: List[str],
    continuations: List[str],
    device: torch.device,
    system_prompt: Optional[str] = None,
) -> torch.Tensor:
    """
    Flexible NLL computation. Supports three calling patterns:

      1) One prompt, K continuations (for collecting actions/behaviors):
         prompts=["p1", "p2"],  continuations=[["c1","c2"], ["c3","c4"]]

      2) K prompts, one continuation (for phi_moral):
         prompts=[["p1","p2"], ["p3","p4"]],  continuations=["c1", "c2"]

      3) K prompts, K continuations (fully paired):
         prompts=[["p1","p2"]], continuations=[["c1","c2"]]

    If `system_prompt` is provided, it's prepended as a system-role message for
    every row (same value for all prompts in this call). This must match the
    system prompt used at generation time for NLLs to be consistent.

    Returns:
      per_sample_nll: torch.Tensor [B, K]
    """
    B = len(prompts)
    assert B == len(continuations), f"Batch mismatch: {B} vs {len(continuations)}"

    # Normalize both to List[List[str]]
    if isinstance(prompts[0], str):
        prompts = [[p] for p in prompts]
    if isinstance(continuations[0], str):
        continuations = [[c] for c in continuations]

    per_group = []

    for prompt_group, cont_group in zip(prompts, continuations):
        Kp, Kc = len(prompt_group), len(cont_group)

        # Broadcast whichever side is length 1
        if Kp == 1 and Kc > 1:
            prompt_group = prompt_group * Kc
        elif Kc == 1 and Kp > 1:
            cont_group = cont_group * Kp
        elif Kp != Kc:
            raise ValueError(
                f"Cannot broadcast prompts ({Kp}) with continuations ({Kc})"
            )

        K = len(prompt_group)
        sys_list = [system_prompt] * K if system_prompt else None
        batch = collate_prompt_output(
            tokenizer, prompt_group, cont_group, system_prompts=sys_list
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]

        logits = model(**batch, use_cache=False).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        token_nll = F.cross_entropy(
            shift_logits.transpose(1, 2),
            shift_labels,
            ignore_index=-100,
            reduction="none",
        )

        mask = shift_labels.ne(-100)
        denom = mask.sum(dim=1).clamp_min(1)
        per_sample_nll = (token_nll * mask).sum(dim=1) / denom  # [K]

        per_group.append(per_sample_nll)

    if len(per_group) == 0:
        return torch.empty((0, 0), device=device)

    return torch.stack(per_group, dim=0)  # [B, K]


def cont_training_loss(model, tokenizer, prompts, completions, device, max_length=512):
    """
    Compute standard SFT cross-entropy loss on (prompt, completion) pairs.

    Signature matches the reg_loss_fn interface:
      reg_loss_fn(model, tokenizer, prompts, completions, device) -> scalar tensor

    Only assistant (completion) tokens contribute to the loss; prompt tokens
    are masked via collate_prompt_output.
    """
    batch = collate_prompt_output(
        tokenizer, prompts, completions, max_length=max_length
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    out = model(**batch, use_cache=False)
    loss = out.loss  # HF token-mean NLL over non-masked tokens
    if torch.isnan(loss):
        return torch.tensor(0.0, device=device, requires_grad=True)
    return loss
