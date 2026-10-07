"""Queueing checks for the serving simulator — Module 04 LAB C (reference).

Time unit: seconds in this module (rates in req/s), matching Lesson 4.4.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- given parts
def poisson_arrivals(lam: float, horizon: float, *, seed: int) -> np.ndarray:
    """Open-loop Poisson arrival times in [0, horizon)."""
    rng = np.random.default_rng(seed)
    n = rng.poisson(lam * horizon)
    return np.sort(rng.uniform(0.0, horizon, n))


def gamma_services(n: int, mean: float, cv: float, *, seed: int) -> np.ndarray:
    """Service times with the given mean and coefficient of variation (cv=0: deterministic)."""
    if cv == 0:
        return np.full(n, float(mean))
    rng = np.random.default_rng(seed)
    shape = 1.0 / cv ** 2
    return rng.gamma(shape, mean / shape, n)


# ------------------------------------------------------------------- the lab
def fifo_single_server(arrivals, services):
    """Lindley recursion for a FIFO single-server queue.
    Returns (starts, departures): start_i = max(arrival_i, departure_{i-1})."""
    arrivals = np.asarray(arrivals, dtype=float)
    services = np.asarray(services, dtype=float)
    starts = np.empty_like(arrivals)
    deps = np.empty_like(arrivals)
    free = -np.inf
    for i in range(len(arrivals)):
        starts[i] = max(arrivals[i], free)
        free = deps[i] = starts[i] + services[i]
    return starts, deps


def pk_wq(lam: float, mean_s: float, sd_s: float) -> float:
    """Pollaczek-Khinchine mean queue wait for M/G/1 (Lesson 4.4). Requires rho < 1."""
    rho = lam * mean_s
    if rho >= 1:
        raise ValueError("P-K mean wait is undefined for rho >= 1")
    return lam * (sd_s ** 2 + mean_s ** 2) / (2.0 * (1.0 - rho))


def littles_terms(enter, leave, t0: float, t1: float) -> dict:
    """Finite-window Little's Law terms for one boundary (Little 2011, §2).

    enter/leave: per-request boundary entry and exit times (leave may be inf).
    Population: every request present in the window, i.e. it enters before t1 and
    either leaves after t0 or enters at/after t0 (zero-length visits count).
    L   = time-average count in the window
    lam = population size / (t1 - t0)
    W   = mean in-window residence (time clipped to [t0, t1))
    """
    enter = np.asarray(enter, dtype=float)
    leave = np.asarray(leave, dtype=float)
    inside = np.clip(np.minimum(leave, t1) - np.maximum(enter, t0), 0.0, None)
    present = (enter < t1) & ((leave > t0) | (enter >= t0))
    T = t1 - t0
    n = int(present.sum())
    return {"L": float(inside.sum() / T), "lam": n / T,
            "W": float(inside[present].mean()) if n else 0.0, "n": n}


def warmup_cutoff(horizon: float, warmup_frac: float) -> float:
    """Declared warm-up rule: discard the first warmup_frac of the horizon."""
    return horizon * warmup_frac


def steady_estimates(arrivals, starts, departures, horizon: float, warmup_frac: float) -> dict:
    """System (arrival->departure) and queue (arrival->start) Little terms measured
    only in [warmup_cutoff, horizon). Nothing that happens before the cutoff may
    affect the result."""
    t0 = warmup_cutoff(horizon, warmup_frac)
    sys_ = littles_terms(arrivals, departures, t0, horizon)
    q = littles_terms(arrivals, starts, t0, horizon)
    return {"L": sys_["L"], "lam": sys_["lam"], "W": sys_["W"],
            "Lq": q["L"], "lam_q": q["lam"], "Wq": q["W"], "t0": t0}


def backlog(arrivals, departures, t: float) -> int:
    """Requests in the system at time t."""
    a = np.asarray(arrivals)
    d = np.asarray(departures)
    return int(np.sum((a <= t) & (d > t)))
