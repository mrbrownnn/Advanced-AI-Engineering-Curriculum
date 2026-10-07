"""Transparent reference forward built from torch built-ins and a complex-number
RoPE, reading the parameters of YOUR TinyDecoder. It shares no code with
starter/decoder.py, so agreement is evidence, not tautology.
"""
import torch
import torch.nn.functional as F


def reference_logits(model, ids):
    cfg = model.cfg
    b, s = ids.shape
    dh, half = cfg.d_h, cfg.d_h // 2
    inv = cfg.rope_base ** (-torch.arange(half, dtype=torch.float64) * 2 / dh)
    rot = torch.polar(torch.ones(s, half, dtype=torch.float64), torch.arange(s, dtype=torch.float64)[:, None] * inv)

    def rope(x):  # pairs (i, i + dh/2) as one complex number
        z = torch.complex(x[..., :half].double(), x[..., half:].double()) * rot
        return torch.cat((z.real, z.imag), dim=-1).to(x.dtype)

    def norm(x, w):
        return F.rms_norm(x.float(), (cfg.d,), eps=cfg.eps).to(x.dtype) * w

    x = model.embed[ids]
    for L in model.layers:
        h = norm(x, L.attn_norm)
        q = rope((h @ L.wq.T).view(b, s, cfg.n_q, dh).transpose(1, 2))
        k = rope((h @ L.wk.T).view(b, s, cfg.n_kv, dh).transpose(1, 2))
        v = (h @ L.wv.T).view(b, s, cfg.n_kv, dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = x + o.transpose(1, 2).reshape(b, s, -1) @ L.wo.T
        h = norm(x, L.mlp_norm)
        x = x + (F.silu(h @ L.w_gate.T) * (h @ L.w_up.T)) @ L.w_down.T
    return norm(x, model.final_norm) @ model.lm_head.T
