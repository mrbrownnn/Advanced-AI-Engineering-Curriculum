"""LAB A starter: fill each TODO without using a high-level decoder block.

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
    """RMSNorm over LAST axis. x [...,d], weight [d] -> [...,d], preserve x.dtype; accumulate at least float32."""
    # TODO: mean square (no mean subtraction), epsilon inside sqrt, cast boundary.
    raise NotImplementedError("HINT: TODO: RMSNorm over the feature axis with float32 accumulation")


def rope_cos_sin(positions, d_h, base):
    """positions [S] -> cos,sin [S,d_h] float32 for half-split pairing (i, i+d_h/2)."""
    # TODO: inv_freq[i] = base**(-2*i/d_h); repeat each angle schedule across halves.
    raise NotImplementedError("HINT: TODO: construct half-split RoPE cos/sin")


def apply_rope(x, cos, sin):
    """x [B,H,S,dh], cos/sin [S,dh] -> rotated x, pairing (i, i+dh/2)."""
    raise NotImplementedError("HINT: TODO: rotate half-split feature pairs")


def causal_mask(s_q, s_k, dtype=torch.float32):
    """Additive [s_q,s_k]. Query i has absolute position s_k-s_q+i; allow key j <= it."""
    raise NotImplementedError("HINT: TODO: cached-prefix-aware causal mask (diagonal included)")


def expand_kv(x, n_rep):
    """[B,Hkv,T,dh] -> [B,Hkv*n_rep,T,dh], contiguous groups: out head h reads h//n_rep."""
    raise NotImplementedError("HINT: TODO: repeat each KV head as one contiguous group")


def attention(q, k, v, mask):
    """q [B,Hq,S,dh], k/v [B,Hkv,T,dh], additive mask [S,T] -> [B,Hq,S,dh]."""
    raise NotImplementedError("HINT: TODO: scaled dot-product GQA; softmax over key axis in float32")


def swiglu(x, w_gate, w_up, w_down):
    """x [...,d], gate/up [m,d], down [d,m] -> [...,d]. SiLU only gate branch."""
    raise NotImplementedError("HINT: TODO: SwiGLU")


class Layer(nn.Module):
    """Parameter container supplied so tests can reconcile shapes and gradients."""
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
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Parameter(torch.randn(cfg.vocab, cfg.d))
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.Parameter(1 + 0.1 * torch.randn(cfg.d))
        self.lm_head = nn.Parameter(torch.randn(cfg.vocab, cfg.d) / math.sqrt(cfg.d))

    def forward(self, ids, cache=None):
        """ids [B,S] -> logits [B,S,V], new_cache list[(K,V)] with K/V [B,Hkv,T,dh].

        Cache stores ROTATED keys. New-token positions start at T_past. Preserve pre-norm
        residual order: x + Attention(RMSNorm(x)); then u + SwiGLU(RMSNorm(u)).
        """
        raise NotImplementedError("HINT: TODO: compose the minimal decoder and incremental KV cache")
