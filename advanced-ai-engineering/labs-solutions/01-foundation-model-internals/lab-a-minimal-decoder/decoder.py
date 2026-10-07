"""Minimal decoder-only transformer (reference). Module 01 LAB A.

Axes: B batch, S query positions, T key positions, d width, Hq query heads,
Hkv K/V heads, dh head width, V vocab, m MLP width.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Config:
    vocab: int = 64
    d: int = 32
    n_layers: int = 2
    n_q: int = 4
    n_kv: int = 2
    m: int = 48
    eps: float = 1e-6
    rope_base: float = 10000.0

    @property
    def d_h(self):
        return self.d // self.n_q


def rms_norm(x, weight, eps):
    """RMSNorm over the LAST axis. x [..., d], weight [d] -> [..., d] in x.dtype.

    Mean square and rsqrt are computed in float32 (or x.dtype if wider), the
    normalized value is cast back to x.dtype, then multiplied by weight.
    """
    acc = torch.promote_types(x.dtype, torch.float32)
    h = x.to(acc)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(x.dtype)


def rope_cos_sin(positions, d_h, base):
    """positions [S] (int) -> cos, sin each [S, d_h] float32, half-split layout.

    inv_freq[i] = base ** (-2i/d_h) for i < d_h/2; angle = position * inv_freq;
    the d_h/2 angles are repeated twice: [a_0..a_{n-1}, a_0..a_{n-1}].
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, d_h, 2, dtype=torch.float32) / d_h))
    ang = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat((ang, ang), dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    """Rotate pairs (i, i + d_h/2). x [B,H,S,d_h]; cos/sin [S,d_h] -> same shape/dtype as x."""
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos.to(x.dtype) + rot * sin.to(x.dtype)


def causal_mask(s_q, s_k, dtype=torch.float32):
    """Additive mask [s_q, s_k]. Query row i sits at absolute position s_k - s_q + i
    and may attend to key j iff j <= that position (0), else -inf."""
    q_pos = torch.arange(s_q)[:, None] + (s_k - s_q)
    k_pos = torch.arange(s_k)[None, :]
    mask = torch.zeros(s_q, s_k, dtype=dtype)
    return mask.masked_fill(k_pos > q_pos, float("-inf"))


def expand_kv(x, n_rep):
    """[B,Hkv,T,dh] -> [B,Hkv*n_rep,T,dh]; output head h reads input head h // n_rep (contiguous groups)."""
    if n_rep == 1:
        return x
    b, h, t, dh = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, dh).reshape(b, h * n_rep, t, dh)


def attention(q, k, v, mask):
    """q [B,Hq,S,dh]; k, v [B,Hkv,T,dh]; mask [S,T] additive -> [B,Hq,S,dh]."""
    n_rep = q.shape[1] // k.shape[1]
    k, v = expand_kv(k, n_rep), expand_kv(v, n_rep)
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]) + mask.to(q.dtype)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return probs @ v


def swiglu(x, w_gate, w_up, w_down):
    """x [..., d]; w_gate, w_up [m, d]; w_down [d, m] -> [..., d]."""
    return F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)


class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, dh = cfg.d, cfg.d_h
        self.attn_norm = nn.Parameter(1 + 0.1 * torch.randn(d))
        self.wq = nn.Parameter(torch.randn(cfg.n_q * dh, d) / math.sqrt(d))
        self.wk = nn.Parameter(torch.randn(cfg.n_kv * dh, d) / math.sqrt(d))
        self.wv = nn.Parameter(torch.randn(cfg.n_kv * dh, d) / math.sqrt(d))
        self.wo = nn.Parameter(torch.randn(d, cfg.n_q * dh) / math.sqrt(d))
        self.mlp_norm = nn.Parameter(1 + 0.1 * torch.randn(d))
        self.w_gate = nn.Parameter(torch.randn(cfg.m, d) / math.sqrt(d))
        self.w_up = nn.Parameter(torch.randn(cfg.m, d) / math.sqrt(d))
        self.w_down = nn.Parameter(torch.randn(d, cfg.m) / math.sqrt(cfg.m))


class TinyDecoder(nn.Module):
    """Pre-norm decoder: RMSNorm -> GQA attention with RoPE -> residual -> RMSNorm -> SwiGLU -> residual."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Parameter(torch.randn(cfg.vocab, cfg.d))
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.Parameter(1 + 0.1 * torch.randn(cfg.d))
        self.lm_head = nn.Parameter(torch.randn(cfg.vocab, cfg.d) / math.sqrt(cfg.d))

    def forward(self, ids, cache=None):
        """ids [B,S] -> (logits [B,S,V], new_cache).

        cache: None or list (one per layer) of (k, v), each [B,Hkv,T_past,dh],
        holding ROTATED keys. New tokens take positions T_past .. T_past+S-1.
        """
        cfg = self.cfg
        b, s = ids.shape
        past = 0 if cache is None else cache[0][0].shape[2]
        cos, sin = rope_cos_sin(torch.arange(past, past + s), cfg.d_h, cfg.rope_base)
        mask = causal_mask(s, past + s)
        x = self.embed[ids]
        new_cache = []
        for i, L in enumerate(self.layers):
            h = rms_norm(x, L.attn_norm, cfg.eps)
            q = F.linear(h, L.wq).view(b, s, cfg.n_q, cfg.d_h).transpose(1, 2)
            k = F.linear(h, L.wk).view(b, s, cfg.n_kv, cfg.d_h).transpose(1, 2)
            v = F.linear(h, L.wv).view(b, s, cfg.n_kv, cfg.d_h).transpose(1, 2)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            if cache is not None:
                k = torch.cat((cache[i][0], k), dim=2)
                v = torch.cat((cache[i][1], v), dim=2)
            new_cache.append((k, v))
            o = attention(q, k, v, mask).transpose(1, 2).reshape(b, s, cfg.n_q * cfg.d_h)
            x = x + F.linear(o, L.wo)
            x = x + swiglu(rms_norm(x, L.mlp_norm, cfg.eps), L.w_gate, L.w_up, L.w_down)
        logits = F.linear(rms_norm(x, self.final_norm, cfg.eps), self.lm_head)
        return logits, new_cache
