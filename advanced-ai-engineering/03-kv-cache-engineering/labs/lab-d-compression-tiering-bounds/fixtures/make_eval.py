"""Seeded synthetic eval items for the lossy-retention quality gate stub.

Each item: (baseline score in [0,1], evidence position in [0,1], 0 = oldest token).
Evidence positions are uniform, so dropping old tokens removes evidence for some items.
This is a stand-in for a real task eval, not a quality measurement.
"""
from __future__ import annotations

import random


def items(n: int = 200, seed: int = 0) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    return [(1.0 if rng.random() < 0.8 else 0.0, rng.random()) for _ in range(n)]
