"""Analytical logical-KV payload model (Lesson 3.1). Reference implementation."""
from __future__ import annotations

GIB = 1024 ** 3


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, bytes_per_element: float) -> float:
    """Uniform uncompressed MHA/GQA/MQA: 2 (K,V) x bpe x layers x kv_heads x head_dim."""
    return 2 * bytes_per_element * layers * kv_heads * head_dim


def kv_heads_for(variant: str, query_heads: int, groups: int | None = None) -> int:
    """MHA: one KV head per query head; GQA: one per group; MQA: one."""
    if variant == "MHA":
        return query_heads
    if variant == "GQA":
        if not groups or query_heads % groups:
            raise ValueError("GQA needs groups dividing query_heads")
        return groups
    if variant == "MQA":
        return 1
    raise ValueError(f"unknown variant {variant!r}")


def mla_bytes_per_token(layers: int, d_c: int, d_rope: int, bytes_per_element: float) -> float:
    """MLA retained state: one latent (d_c) + one shared RoPE key (d_rope) per layer. No KV heads."""
    return bytes_per_element * layers * (d_c + d_rope)


def concurrency_bound(device_gib: float, weights_gib: float, other_gib: float,
                      tokens_per_request: int, bytes_per_token: float) -> int:
    """Payload-only optimistic upper bound: floor(free-for-KV / per-request KV)."""
    free = (device_gib - weights_gib - other_gib) * GIB
    if free <= 0:
        return 0
    return int(free // (tokens_per_request * bytes_per_token))
