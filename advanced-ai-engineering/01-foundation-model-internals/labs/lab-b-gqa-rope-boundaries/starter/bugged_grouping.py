"""BREAK: one planted bug. Shape and aggregate counts still look right.
Instrument output heads with uniquely-valued KV heads; identify first mismatch.
"""
import torch


def repeat_kv(x, group_size):
    # Planted bug: creates an interleaved head order for Hkv > 1.
    return x.repeat(1, group_size, 1, 1)
