"""Logits -> next token under an explicit sampling contract (reference). Module 01 LAB A.

Truncation rules follow the pinned Transformers warpers at commit
11c16613d93911300c38ec8c9c2460567be53281 (generation/logits_process.py):
top-k removes scores strictly below the k-th value; top-p sorts ascending and
removes entries whose cumulative probability is <= 1 - p, always keeping at
least min_tokens_to_keep.
"""
import torch

NEG_INF = float("-inf")


def apply_temperature(logits, temperature):
    """logits [..., V] -> logits / T. T must be a float > 0; T == 0 is not greedy."""
    if not isinstance(temperature, (int, float)) or not temperature > 0:
        msg = f"temperature (={temperature}) must be strictly positive"
        if temperature == 0:
            msg += "; for greedy decoding use do_sample=False instead of T=0"
        raise ValueError(msg)
    return logits / temperature


def top_k_filter(logits, k):
    """Set to -inf every score strictly below the k-th largest score (ties at the boundary survive)."""
    k = min(int(k), logits.shape[-1])
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, NEG_INF)


def top_p_filter(logits, p, min_tokens_to_keep=1):
    """Keep the high-probability set whose mass reaches p (ascending-cumulative rule)."""
    sorted_logits, idx = torch.sort(logits, dim=-1, descending=False)
    cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove_sorted = cum <= (1 - p)
    remove_sorted[..., -min_tokens_to_keep:] = False
    remove = remove_sorted.scatter(-1, idx, remove_sorted)
    return logits.masked_fill(remove, NEG_INF)


def process(logits, temperature=1.0, top_k=None, top_p=None, order=("temperature", "top_k", "top_p")):
    """Apply the enabled processors in `order`. Neutral settings (T=1, top_k None/0, top_p None/1.0) are skipped."""
    scores = logits.to(torch.float32)
    for name in order:
        if name == "temperature" and temperature != 1.0:
            scores = apply_temperature(scores, temperature)
        elif name == "top_k" and top_k:
            scores = top_k_filter(scores, top_k)
        elif name == "top_p" and top_p is not None and top_p < 1.0:
            scores = top_p_filter(scores, top_p)
    return scores


def sample_next(logits, contract, generator=None):
    """logits [B,V] (last position) -> token ids [B]. Greedy uses argmax of the RAW logits."""
    if not contract["do_sample"]:
        return torch.argmax(logits, dim=-1)
    scores = process(logits, contract["temperature"], contract["top_k"], contract["top_p"], contract["processor_order"])
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def sampling_contract(**overrides):
    """Every choice that determines the next-token distribution and the draw. Print it with each run."""
    contract = {
        "logit_position": "last",
        "logit_dtype": "float32",
        "processor_order": ["temperature", "top_k", "top_p"],
        "temperature": 1.0,
        "top_k": None,
        "top_p": None,
        "do_sample": False,
        "library_defaults": "none: hand-written sampler; unset fields are disabled (pinned Transformers would default top_k=50)",
        "seed": 0,
        "generator_device": "cpu",
        "stop_token_ids": [],
        "max_new_tokens": 8,
        "context_limit": 64,
        "truncation_policy": "raise if prompt + max_new_tokens > context_limit",
        "tokenizer": "none: synthetic token IDs",
        "use_cache": True,
    }
    unknown = set(overrides) - set(contract)
    if unknown:
        raise KeyError(f"unknown contract fields: {sorted(unknown)}")
    contract.update(overrides)
    return contract


@torch.no_grad()
def generate(model, prompt_ids, contract, generator=None):
    """prompt_ids [1,S] -> list of new token ids (int). Stops at a stop token (included) or max_new_tokens."""
    if prompt_ids.shape[1] + contract["max_new_tokens"] > contract["context_limit"]:
        raise ValueError("prompt + max_new_tokens exceeds context_limit")
    out, cache, ids = [], None, prompt_ids
    seq = prompt_ids
    for _ in range(contract["max_new_tokens"]):
        if contract["use_cache"]:
            logits, cache = model(ids, cache)
        else:
            logits, _ = model(seq)
        tok = sample_next(logits[:, -1].to(torch.float32), contract, generator)
        out.append(int(tok[0]))
        ids = tok[:, None]
        seq = torch.cat((seq, ids), dim=1)
        if out[-1] in contract["stop_token_ids"]:
            break
    return out
