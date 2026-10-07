"""LAB B reference: randomized paired benchmark under monotonic drift."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import t


@dataclass(frozen=True)
class Run:
    block: int
    position: int
    global_position: int
    treatment: str
    order: str
    value_ms: float
    seed: int


def randomized_orders(n_blocks: int, seed: int) -> list[tuple[str, str]]:
    """Balanced random orders: shuffle equal-ish AB and BA counts using seed."""
    rng = np.random.default_rng(seed)
    labels = ["AB"] * ((n_blocks + 1) // 2) + ["BA"] * (n_blocks // 2)
    rng.shuffle(labels)
    return [tuple(x) for x in labels]


def simulate(seed: int, n_blocks: int, effect_ms: float, drift_per_run_ms: float,
             noise_ms: float, randomized: bool = True) -> list[Run]:
    """Simulate fresh-process blocks with random block offsets and run drift.

    value = 100 + block_offset + global_position*drift + B_effect + run_noise.
    Treatment is assigned to run position by randomized order or fixed AB order.
    """
    rng = np.random.default_rng(seed)
    orders = randomized_orders(n_blocks, seed) if randomized else [("A", "B")] * n_blocks
    rows = []
    for block, order in enumerate(orders):
        block_offset = rng.normal(0.0, 3.0)
        label = "".join(order)
        for position, treatment in enumerate(order):
            g = 2 * block + position
            value = (100.0 + block_offset + drift_per_run_ms * g
                     + (effect_ms if treatment == "B" else 0.0)
                     + rng.normal(0.0, noise_ms))
            rows.append(Run(block, position, g, treatment, label, float(value), seed))
    return rows


def paired_differences(rows: list[Run]) -> list[float]:
    """Return one B-A difference per complete block, sorted by block ID."""
    by_block: dict[int, dict[str, float]] = {}
    for r in rows:
        if r.treatment not in ("A", "B"):
            raise ValueError("treatment must be A or B")
        if r.treatment in by_block.setdefault(r.block, {}):
            raise ValueError("duplicate treatment in block")
        by_block[r.block][r.treatment] = r.value_ms
    if any(set(pair) != {"A", "B"} for pair in by_block.values()):
        raise ValueError("every block must contain one A and one B")
    return [by_block[b]["B"] - by_block[b]["A"] for b in sorted(by_block)]


def t_interval(differences, confidence: float = 0.95) -> tuple[float, float, float]:
    """Two-sided t interval for mean block-level difference. Returns mean, lo, hi."""
    d = np.asarray(list(differences), dtype=float)
    if len(d) < 2 or not 0.0 < confidence < 1.0:
        raise ValueError("need >=2 blocks and confidence in (0,1)")
    mean = float(d.mean())
    half = float(t.ppf((1 + confidence) / 2, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d)))
    return mean, mean - half, mean + half


def as_records(rows: list[Run]) -> list[dict]:
    return [asdict(r) for r in rows]
