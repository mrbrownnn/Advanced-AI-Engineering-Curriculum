"""Analytical accounting plus tiny real modules. Module 01 LAB C.
FMA convention: one multiply-add = 2 FLOPs. Counts are logical, not latency.
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    vocab: int = 97
    d: int = 32
    m: int = 48
    n_q: int = 4
    n_kv: int = 2
    layers: int = 2
    bias: bool = False
    tie_embeddings: bool = False
    adapter_rank: int = 0
    experts: int = 0
    experts_per_token: int = 0

    @property
    def dh(self): return self.d // self.n_q

    def validate(self):
        if self.d % self.n_q or self.n_q % self.n_kv:
            raise ValueError("require d % n_q == 0 and n_q % n_kv == 0")
        if self.tie_embeddings and self.vocab <= 0: raise ValueError("vocab must be positive")
        if bool(self.experts) != bool(self.experts_per_token): raise ValueError("experts and experts_per_token must be jointly enabled")
        if self.experts_per_token > self.experts: raise ValueError("experts_per_token cannot exceed experts")


def parameter_manifest(c):
    """Rows with name, formula, assumptions, analytical count. Sum matches TinyModel unique parameters."""
    c.validate(); d, m, dh, b = c.d, c.m, c.dh, int(c.bias)
    rows = [{"name": "token_embedding", "formula": "V*d", "assumptions": "unique input embedding", "count": c.vocab*d}]
    for i in range(c.layers):
        prefix = f"layers.{i}"
        rows += [
            {"name": f"{prefix}.q_proj", "formula": "d*(Hq*dh)+bias", "assumptions": f"bias={c.bias}", "count": d*c.n_q*dh+b*c.n_q*dh},
            {"name": f"{prefix}.k_proj", "formula": "d*(Hkv*dh)+bias", "assumptions": f"bias={c.bias}", "count": d*c.n_kv*dh+b*c.n_kv*dh},
            {"name": f"{prefix}.v_proj", "formula": "d*(Hkv*dh)+bias", "assumptions": f"bias={c.bias}", "count": d*c.n_kv*dh+b*c.n_kv*dh},
            {"name": f"{prefix}.o_proj", "formula": "(Hq*dh)*d+bias", "assumptions": f"bias={c.bias}", "count": c.n_q*dh*d+b*d},
            {"name": f"{prefix}.norms", "formula": "2*d", "assumptions": "two RMSNorm scales", "count": 2*d},
        ]
        if c.experts:
            rows += [
                {"name": f"{prefix}.router", "formula": "d*E", "assumptions": "bias-free", "count": d*c.experts},
                {"name": f"{prefix}.experts", "formula": "3*E*d*m", "assumptions": "E bias-free SwiGLU experts", "count": 3*c.experts*d*m},
            ]
        else:
            rows.append({"name": f"{prefix}.mlp", "formula": "3*d*m+3*biases", "assumptions": f"bias={c.bias}", "count": 3*d*m+b*(2*m+d)})
        if c.adapter_rank:
            rows.append({"name": f"{prefix}.adapter", "formula": "2*d*r", "assumptions": "bias-free bottleneck", "count": 2*d*c.adapter_rank})
    rows.append({"name": "final_norm", "formula": "d", "assumptions": "RMSNorm scale", "count": d})
    if not c.tie_embeddings:
        rows.append({"name": "lm_head", "formula": "V*d", "assumptions": "untied, bias-free", "count": c.vocab*d})
    return rows


def analytical_parameter_count(c): return sum(r["count"] for r in parameter_manifest(c))


def actual_parameter_count(module, unique_storage=True):
    """Count parameters, optionally deduplicating aliases by (data_ptr, storage_offset, numel)."""
    params = list(module.named_parameters(remove_duplicate=False))
    if not unique_storage: return sum(p.numel() for _, p in params)
    seen, total = set(), 0
    for _, p in params:
        key = (p.untyped_storage().data_ptr(), p.storage_offset(), p.numel())
        if key not in seen: seen.add(key); total += p.numel()
    return total


def logical_flops(c, batch, sequence):
    """Full-sequence leading FLOPs; causal dense attention is nominal full SxS. FMA=2.

    Embedding/norm/activation/softmax/router selection are excluded. Bias adds excluded.
    MoE weight term counts selected experts, not all resident experts.
    """
    c.validate(); tokens = batch*sequence; d, dh = c.d, c.dh
    attn_params = 2*d*d + 2*d*c.n_kv*dh
    if c.experts: weight_params = attn_params + 3*c.experts_per_token*d*c.m
    else: weight_params = attn_params + 3*d*c.m
    if c.adapter_rank: weight_params += 2*d*c.adapter_rank
    weight_flops = c.layers * 2*tokens*weight_params
    attention_flops = c.layers * 4*batch*sequence*sequence*d
    head_flops = 2*tokens*c.d*c.vocab
    return {"fma_convention": 2, "weight_flops": weight_flops, "attention_flops": attention_flops,
            "lm_head_flops": head_flops, "total": weight_flops+attention_flops+head_flops,
            "exclusions": ["embedding", "norm", "activation", "softmax", "bias adds", "router selection"]}


def kv_bytes(c, batch, sequence, dtype_bytes):
    """Logical K+V payload for all layers: 2*L*B*S*Hkv*dh*dtype_bytes."""
    return 2*c.layers*batch*sequence*c.n_kv*c.dh*dtype_bytes


def mla_retained_elements(layers, d_c, d_h_rope):
    """Per token across layers: L*(d_c+d_h_rope). MLA is not GQA."""
    if min(layers, d_c, d_h_rope) <= 0: raise ValueError("MLA dimensions must be positive")
    return {"per_layer_per_token": d_c+d_h_rope, "all_layers_per_token": layers*(d_c+d_h_rope),
            "cached_tensors": ["c_KV", "k_R"], "gqa_formula_applies": False}


def moe_counts(d, m, experts, selected):
    if not 0 < selected <= experts: raise ValueError("require 0 < selected <= experts")
    return {"total_expert_parameters": 3*experts*d*m, "selected_weight_elements_per_token": 3*selected*d*m}


class Block(nn.Module):
    def __init__(self, c):
        super().__init__(); d, dh, m = c.d, c.dh, c.m
        self.c = c
        self.q_proj=nn.Linear(d,c.n_q*dh,bias=c.bias); self.k_proj=nn.Linear(d,c.n_kv*dh,bias=c.bias)
        self.v_proj=nn.Linear(d,c.n_kv*dh,bias=c.bias); self.o_proj=nn.Linear(c.n_q*dh,d,bias=c.bias)
        self.norm1=nn.Parameter(torch.ones(d)); self.norm2=nn.Parameter(torch.ones(d))
        if c.experts:
            self.router=nn.Linear(d,c.experts,bias=False)
            self.experts=nn.ModuleList(nn.ModuleList([nn.Linear(d,m,bias=False),nn.Linear(d,m,bias=False),nn.Linear(m,d,bias=False)]) for _ in range(c.experts))
        else:
            self.gate=nn.Linear(d,m,bias=c.bias); self.up=nn.Linear(d,m,bias=c.bias); self.down=nn.Linear(m,d,bias=c.bias)
        if c.adapter_rank:
            self.adapter_down=nn.Linear(d,c.adapter_rank,bias=False); self.adapter_up=nn.Linear(c.adapter_rank,d,bias=False)

    def forward(self,x):
        b,s,d=x.shape; c=self.c; h=F.rms_norm(x,(d,),self.norm1,1e-6)
        q=self.q_proj(h).view(b,s,c.n_q,c.dh).transpose(1,2)
        k=self.k_proj(h).view(b,s,c.n_kv,c.dh).transpose(1,2)
        v=self.v_proj(h).view(b,s,c.n_kv,c.dh).transpose(1,2)
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
        x=x+self.o_proj(a.transpose(1,2).reshape(b,s,d)); h=F.rms_norm(x,(d,),self.norm2,1e-6)
        if c.experts:
            # Deterministic dense execution of selected top-k experts (real executed work; no capacity padding).
            idx=self.router(h).topk(c.experts_per_token,-1).indices
            y=torch.zeros_like(h)
            for e,(gate,up,down) in enumerate(self.experts):
                selected=(idx==e).any(-1)[...,None]
                y=y+torch.where(selected,down(F.silu(gate(h))*up(h)),torch.zeros_like(h))
        else: y=self.down(F.silu(self.gate(h))*self.up(h))
        x=x+y
        if c.adapter_rank: x=x+self.adapter_up(self.adapter_down(x))
        return x


class TinyModel(nn.Module):
    def __init__(self,c):
        super().__init__(); c.validate(); self.c=c
        self.embed=nn.Embedding(c.vocab,c.d); self.layers=nn.ModuleList(Block(c) for _ in range(c.layers)); self.final_norm=nn.Parameter(torch.ones(c.d)); self.lm_head=nn.Linear(c.d,c.vocab,bias=False)
        if c.tie_embeddings: self.lm_head.weight=self.embed.weight
    def forward(self,ids):
        x=self.embed(ids)
        for layer in self.layers: x=layer(x)
        return self.lm_head(F.rms_norm(x,(self.c.d,),self.final_norm,1e-6))
