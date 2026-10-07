"""Analytical logical-KV payload model (Lesson 3.1). Fill the TODOs.

All results are LOGICAL payload bytes: no block rounding, scales, alignment,
allocator state, or sharding.
"""
from __future__ import annotations

GIB = 1024 ** 3


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, bytes_per_element: float) -> float:
    """Uniform uncompressed decoder self-attention (MHA/GQA/MQA), K and V both stored."""
    raise NotImplementedError("HINT: TODO: Lesson 3.1 formula")


def kv_heads_for(variant: str, query_heads: int, groups: int | None = None) -> int:
    """Number of stored KV heads for "MHA", "GQA" (groups KV heads), or "MQA".
    Raise ValueError for an unknown variant or a GQA group count that does not divide query_heads."""
    raise NotImplementedError("HINT: TODO")


def mla_bytes_per_token(layers: int, d_c: int, d_rope: int, bytes_per_element: float) -> float:
    """MLA retained state per token: one latent of width d_c and one shared RoPE key of
    width d_rope per layer. Which config fields matter here, and which do not?"""
    raise NotImplementedError("HINT: TODO")


def concurrency_bound(device_gib: float, weights_gib: float, other_gib: float,
                      tokens_per_request: int, bytes_per_token: float) -> int:
    """Payload-only optimistic upper bound on concurrent requests (an integer).
    Return 0 if nothing is left for KV."""
    raise NotImplementedError("HINT: TODO")
