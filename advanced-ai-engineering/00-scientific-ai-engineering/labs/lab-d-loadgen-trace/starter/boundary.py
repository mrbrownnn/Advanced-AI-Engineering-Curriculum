"""LAB D reference: rescore fixed SUT event log under metric policies."""
from __future__ import annotations

import math


def nearest_rank(values, q: float) -> float:
    """Return ceil(q*n)-th sorted value, 1-based. q in (0,1]."""
    raise NotImplementedError("HINT: implement TODO: nearest-rank quantile")


def loadgen_floor(values, q: float) -> float:
    """Pinned LoadGen: sorted_values[floor(n*q)], 0-based. q in [0,1)."""
    raise NotImplementedError("HINT: implement TODO: pinned LoadGen index convention")


def rescore(events: list[dict], origin: str, population: str,
            quantile: str, q: float, timeout_ms: float) -> dict:
    """Rescore unchanged event rows.

    origin: scheduled | actual_issue
    population: success | all (timeout rows censored at timeout_ms)
    quantile: nearest_rank | loadgen_floor
    Return n_population, n_issued, n_failures, failure_rate, q, quantile_ms,
    origin, population, quantile_convention.
    """
    raise NotImplementedError("HINT: implement TODO: policy scoring")


def synthetic_log() -> list[dict]:
    """Return ten fixed events used in fixtures/sut-events.jsonl.

    IDs 0..7 succeed; 8..9 timeout. Scheduled times step by 100 ms. Issue lags
    are [0,0,0,0,0,0,300,250,200,150]; success service times are
    [40,42,43,45,47,50,55,60]. Do not alter SUT rows across scoring policies.
    """
    raise NotImplementedError("HINT: implement TODO: fixed event log")
