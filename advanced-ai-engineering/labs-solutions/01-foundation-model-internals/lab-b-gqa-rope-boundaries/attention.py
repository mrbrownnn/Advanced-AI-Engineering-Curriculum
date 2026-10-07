"""MHA/MQA/GQA geometry and RoPE boundary reference. Module 01 LAB B."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def validate_geometry(d, n_q, n_kv):
    """Return head width and group size. Require d % n_q == 0 and n_q % n_kv == 0."""
    if min(d, n_q, n_kv) <= 0:
        raise ValueError("d, n_q, and n_kv must be positive")
    if d % n_q:
        raise ValueError("d must be divisible by n_q")
    if n_q % n_kv:
        raise ValueError("n_q must be divisible by n_kv")
    return d // n_q, n_q // n_kv


def geometry(d, n_q, n_kv, batch, sequence, dtype_bytes=4):
    """Logical projection/count geometry. KV payload covers one layer and full sequence."""
    dh, group = validate_geometry(d, n_q, n_kv)
    return {
        "family": "MHA" if n_kv == n_q else "MQA" if n_kv == 1 else "GQA",
        "head_dim": dh,
        "group_size": group,
        "q_elements": batch * n_q * sequence * dh,
        "k_elements": batch * n_kv * sequence * dh,
        "v_elements": batch * n_kv * sequence * dh,
        "q_parameters": d * n_q * dh,
        "k_parameters": d * n_kv * dh,
        "v_parameters": d * n_kv * dh,
        "o_parameters": n_q * dh * d,
        "kv_bytes": 2 * batch * sequence * n_kv * dh * dtype_bytes,
    }


def repeat_kv(x, group_size):
    """[B,Hkv,T,dh] -> [B,Hq,T,dh], contiguous head groups."""
    return torch.repeat_interleave(x, group_size, dim=1)


class GQAAttention(nn.Module):
    """Bias-free attention for geometry/parity studies. Input/output [B,S,d]."""
    def __init__(self, d, n_q, n_kv):
        super().__init__()
        self.d, self.n_q, self.n_kv = d, n_q, n_kv
        self.dh, self.group_size = validate_geometry(d, n_q, n_kv)
        self.q_proj = nn.Linear(d, n_q * self.dh, bias=False)
        self.k_proj = nn.Linear(d, n_kv * self.dh, bias=False)
        self.v_proj = nn.Linear(d, n_kv * self.dh, bias=False)
        self.o_proj = nn.Linear(n_q * self.dh, d, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_q, self.dh).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv, self.dh).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.o_proj(o.transpose(1, 2).reshape(b, s, self.n_q * self.dh))


def mha_reference(module, x):
    """Explicit MHA reference using materialized contiguous K/V repeats and an additive mask."""
    b, s, _ = x.shape
    q = module.q_proj(x).view(b, s, module.n_q, module.dh).transpose(1, 2)
    k = module.k_proj(x).view(b, s, module.n_kv, module.dh).transpose(1, 2)
    v = module.v_proj(x).view(b, s, module.n_kv, module.dh).transpose(1, 2)
    k, v = repeat_kv(k, module.group_size), repeat_kv(v, module.group_size)
    scores = q @ k.transpose(-1, -2) / math.sqrt(module.dh)
    scores = scores.masked_fill(torch.triu(torch.ones(s, s, dtype=torch.bool), 1), float("-inf"))
    o = torch.softmax(scores, -1) @ v
    return module.o_proj(o.transpose(1, 2).reshape(b, s, -1))


def rope_cos_sin(positions, dh, base=10000.0, dtype=torch.float32):
    """Compute angles in float64, then cos/sin cast to dtype. positions [S] -> [S,dh]."""
    if dh % 2:
        raise ValueError("RoPE head dimension must be even")
    inv = base ** (-torch.arange(0, dh, 2, dtype=torch.float64) / dh)
    angles = positions.to(torch.float64)[:, None] * inv
    emb = torch.cat((angles, angles), -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x, cos, sin):
    """Half-split RoPE: x [B,H,S,dh], cos/sin [S,dh]."""
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin


def relative_score(q, k, q_positions, k_positions, base=10000.0):
    """q,k [B,H,S,dh], positions [S]; return rotated dot scores [B,H,S]."""
    cq, sq = rope_cos_sin(q_positions, q.shape[-1], base, q.dtype)
    ck, sk = rope_cos_sin(k_positions, k.shape[-1], base, k.dtype)
    return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum(-1)
