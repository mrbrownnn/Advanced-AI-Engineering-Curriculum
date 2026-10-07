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
    """Return a seeded, balanced shuffle of AB and BA block orders.

    Difference between counts of AB and BA must be at most one. Same seed must
    reproduce the realized list; a different seed should generally change it.
    """
    raise NotImplementedError("HINT: implement TODO: randomized balanced orders")


def simulate(seed: int, n_blocks: int, effect_ms: float, drift_per_run_ms: float,
             noise_ms: float, randomized: bool = True) -> list[Run]:
    """Simulate two runs in each independent fresh-process block.

    Use a seeded numpy Generator. Draw ONE block offset N(0,3) for each block
    and ONE run noise N(0,noise_ms) for each run. Base is 100 ms. Add
    block_offset, global_position*drift_per_run_ms, and effect_ms only for B.
    randomized=True uses randomized_orders(); False always uses AB.
    Record block, within-block position, global position, treatment, realized
    order string, value, and seed in each Run.
    """
    raise NotImplementedError("HINT: implement TODO: drift simulation")


def paired_differences(rows: list[Run]) -> list[float]:
    """Return one B-A value per complete block, sorted by block ID.

    Reject duplicate, missing, or unknown treatments with ValueError.
    """
    raise NotImplementedError("HINT: implement TODO: block-level paired differences")


def t_interval(differences, confidence: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, lower, upper) two-sided t interval.

    Experimental n is number of block-level differences. Use sample standard
    deviation (ddof=1). Require >=2 differences and confidence in (0,1).
    """
    raise NotImplementedError("HINT: implement TODO: t interval at block level")


def as_records(rows: list[Run]) -> list[dict]:
    """Convert dataclass rows to JSON-serializable dictionaries."""
    raise NotImplementedError("HINT: implement TODO: records")
