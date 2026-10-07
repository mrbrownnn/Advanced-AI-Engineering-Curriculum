"""BREAK exercise: this looks plausible and has exactly one semantic bug.

Use causal isolation and diagonal-boundary instrumentation. Do not diff against
labs-solutions. Record first failing position in report §6.
"""
import torch


def causal_mask(s_q, s_k, dtype=torch.float32):
    q_pos = torch.arange(s_q)[:, None] + (s_k - s_q)
    k_pos = torch.arange(s_k)[None, :]
    mask = torch.zeros(s_q, s_k, dtype=dtype)
    # Planted bug is in this line. Shape remains correct.
    return mask.masked_fill(k_pos >= q_pos, float("-inf"))
