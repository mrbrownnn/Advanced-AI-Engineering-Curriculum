"""Seeded heterogeneous request trace for Lab A experiments. Deterministic."""
from __future__ import annotations

import random


def heterogeneous(n: int = 40, seed: int = 0, max_len: int = 2048) -> list[dict]:
    rng = random.Random(seed)
    out = []
    t = 0
    for i in range(n):
        t += rng.randint(0, 30)
        prompt = rng.choice([64, 128, 256, 512, 1024])
        output = min(max_len - prompt, int(rng.lognormvariate(5.0, 1.0)) + 1)
        out.append({"rid": f"r{i}", "arrival": t, "prompt": prompt, "output": output, "max_len": max_len})
    return out


def equal_fixed(n: int = 8, length: int = 256) -> list[dict]:
    """Break test: identical fixed-length sequences, reservation == final length."""
    return [{"rid": f"e{i}", "arrival": 0, "prompt": length // 2, "output": length - length // 2,
             "max_len": length} for i in range(n)]
