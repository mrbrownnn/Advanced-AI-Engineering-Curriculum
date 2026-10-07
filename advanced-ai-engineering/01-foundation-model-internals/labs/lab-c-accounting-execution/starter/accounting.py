"""LAB C starter: reconcile formulas against real PyTorch modules. FMA=2."""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    vocab:int=97; d:int=32; m:int=48; n_q:int=4; n_kv:int=2; layers:int=2
    bias:bool=False; tie_embeddings:bool=False; adapter_rank:int=0; experts:int=0; experts_per_token:int=0
    @property
    def dh(self): return self.d//self.n_q
    def validate(self):
        """Reject invalid head geometry and invalid expert selection."""
        raise NotImplementedError("HINT: TODO: config validation")


def parameter_manifest(c):
    """One row per group: name, formula, assumptions, count. Must match TinyModel unique storage."""
    raise NotImplementedError("HINT: TODO: parameter manifest including bias/tie/adapter/MoE")

def analytical_parameter_count(c):
    raise NotImplementedError("HINT: TODO: sum manifest")

def actual_parameter_count(module, unique_storage=True):
    """Count named parameters; if unique_storage deduplicate aliases by storage identity."""
    raise NotImplementedError("HINT: TODO: measured parameter count")

def logical_flops(c,batch,sequence):
    """Return FMA convention, weight, nominal QK+AV, LM head, total, exclusions."""
    raise NotImplementedError("HINT: TODO: leading logical FLOPs")

def kv_bytes(c,batch,sequence,dtype_bytes):
    """Logical K+V bytes across all layers."""
    raise NotImplementedError("HINT: TODO: 2*L*B*S*Hkv*dh*dtype bytes")

def mla_retained_elements(layers,d_c,d_h_rope):
    """Return per-layer/all-layer retained elements, tensor names, and gqa_formula_applies=False."""
    raise NotImplementedError("HINT: TODO: c_KV + k_R; do not use num_key_value_heads")

def moe_counts(d,m,experts,selected):
    """Return separate total expert and selected weight-element counts."""
    raise NotImplementedError("HINT: TODO: distinguish total from selected")


class Block(nn.Module):
    """Build q/k/v/o, two norm scales, dense SwiGLU OR router+experts, optional adapter; forward causal GQA."""
    def __init__(self,c):
        super().__init__(); self.c=c
        raise NotImplementedError("HINT: TODO: instantiate real block matching manifest")
    def forward(self,x): raise NotImplementedError("HINT: TODO: real forward for profiler")


class TinyModel(nn.Module):
    """Embedding, c.layers Blocks, final RMSNorm, LM head; alias lm_head.weight when tied."""
    def __init__(self,c):
        super().__init__(); self.c=c
        raise NotImplementedError("HINT: TODO: instantiate model and optional tying")
    def forward(self,ids): raise NotImplementedError("HINT: TODO: model forward")
