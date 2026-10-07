"""LAB D reference: rescore fixed SUT event log under metric policies."""
from __future__ import annotations

import math


def nearest_rank(values, q: float) -> float:
    """ceil(q*n)-th sorted value, 1-based."""
    if not values or not 0 < q <= 1:
        raise ValueError("nonempty values and q in (0,1] required")
    return sorted(values)[math.ceil(q * len(values)) - 1]


def loadgen_floor(values, q: float) -> float:
    """Pinned LoadGen convention: sorted_values[floor(n*q)], 0-based, q < 1."""
    if not values or not 0 <= q < 1:
        raise ValueError("nonempty values and q in [0,1) required")
    xs = sorted(values)
    return xs[math.floor(len(xs) * q)]


def rescore(events: list[dict], origin: str, population: str,
            quantile: str, q: float, timeout_ms: float) -> dict:
    """Score fixed rows under boundary, population, and quantile policies.

    Required row fields: scheduled_ms, actual_issue_ms, complete_ms, status.
    Timeout rows have no usable completion and are censored at timeout_ms when
    population='all'. Success-only excludes them and reports failure_rate.
    """
    if origin not in ("scheduled", "actual_issue"):
        raise ValueError("bad origin")
    if population not in ("success", "all"):
        raise ValueError("bad population")
    if quantile not in ("nearest_rank", "loadgen_floor"):
        raise ValueError("bad quantile")
    ok = [e for e in events if e["status"] == "ok"]
    failures = len(events) - len(ok)
    values = [e["complete_ms"] - e[f"{origin}_ms"] for e in ok]
    if population == "all":
        values += [timeout_ms] * failures
    fn = nearest_rank if quantile == "nearest_rank" else loadgen_floor
    return {"n_population": len(values), "n_issued": len(events), "n_failures": failures,
            "failure_rate": failures / len(events), "q": q,
            "quantile_ms": fn(values, q), "origin": origin,
            "population": population, "quantile_convention": quantile}


def synthetic_log() -> list[dict]:
    """Fixed SUT outcomes with issue delays and two timeout outcomes."""
    scheduled = list(range(0, 1000, 100))
    issue_lag = [0, 0, 0, 0, 0, 0, 300, 250, 200, 150]
    service = [40, 42, 43, 45, 47, 50, 55, 60, None, None]
    rows = []
    for i, (s, lag, work) in enumerate(zip(scheduled, issue_lag, service)):
        actual = s + lag
        rows.append({"id": i, "scheduled_ms": s, "actual_issue_ms": actual,
                     "complete_ms": None if work is None else actual + work,
                     "status": "timeout" if work is None else "ok"})
    return rows
