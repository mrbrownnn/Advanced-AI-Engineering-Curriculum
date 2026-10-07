"""LAB C reference: clustered uncertainty and tail exposure."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import t


def simulate_clusters(seed: int, n_processes: int, requests_per_process: int,
                      process_sd: float = 4.0, request_sd: float = 1.0,
                      mean_ms: float = 100.0) -> list[dict]:
    """Generate request rows sharing one random process effect per process."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_processes):
        process_mean = mean_ms + rng.normal(0.0, process_sd)
        for r in range(requests_per_process):
            rows.append({"process": p, "request": r,
                         "latency_ms": float(process_mean + rng.normal(0.0, request_sd))})
    return rows


def _interval(values, confidence: float) -> tuple[float, float, float]:
    x = np.asarray(list(values), dtype=float)
    if len(x) < 2 or not 0.0 < confidence < 1.0:
        raise ValueError("need >=2 units and confidence in (0,1)")
    mean = float(x.mean())
    half = float(t.ppf((1 + confidence) / 2, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
    return mean, mean - half, mean + half


def naive_request_interval(rows: list[dict], confidence: float = 0.95):
    """Deliberately naive interval treating requests as independent units."""
    return _interval((r["latency_ms"] for r in rows), confidence)


def process_interval(rows: list[dict], confidence: float = 0.95):
    """t interval over one mean per independent process."""
    groups: dict[int, list[float]] = {}
    for r in rows:
        groups.setdefault(r["process"], []).append(r["latency_ms"])
    return _interval((np.mean(groups[p]) for p in sorted(groups)), confidence)


def cluster_bootstrap_interval(rows: list[dict], seed: int, resamples: int = 2000,
                               confidence: float = 0.95) -> tuple[float, float, float]:
    """Percentile bootstrap that resamples whole processes, never requests."""
    if resamples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap settings")
    groups: dict[int, list[float]] = {}
    for r in rows:
        groups.setdefault(r["process"], []).append(r["latency_ms"])
    means = np.asarray([np.mean(groups[p]) for p in sorted(groups)])
    if len(means) < 2:
        raise ValueError("need >=2 processes")
    rng = np.random.default_rng(seed)
    estimates = means[rng.integers(0, len(means), size=(resamples, len(means)))].mean(axis=1)
    alpha = 1 - confidence
    lo, hi = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return float(means.mean()), float(lo), float(hi)


def tail_exposure(q: float, n: int) -> float:
    """P(at least one event) in n independent trials with event probability q."""
    if not 0.0 < q < 1.0 or n < 0:
        raise ValueError("q in (0,1), n >= 0")
    return -math.expm1(n * math.log1p(-q))


def trials_for_exposure(q: float, target: float) -> int:
    """Smallest integer n with 1-(1-q)^n >= target."""
    if not 0.0 < q < 1.0 or not 0.0 < target < 1.0:
        raise ValueError("probabilities must be in (0,1)")
    return math.ceil(math.log1p(-target) / math.log1p(-q))
