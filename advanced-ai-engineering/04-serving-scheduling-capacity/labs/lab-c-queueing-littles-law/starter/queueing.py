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
    raise NotImplementedError("TODO. HINT: Follow the declared boundary and invariant in the module Feedback Contract.")


def pk_wq(lam: float, mean_s: float, sd_s: float) -> float:
    """Pollaczek-Khinchine mean queue wait for M/G/1 (Lesson 4.4). Requires rho < 1."""
    raise NotImplementedError("TODO. HINT: Follow the declared boundary and invariant in the module Feedback Contract.")


def littles_terms(enter, leave, t0: float, t1: float) -> dict:
    """Finite-window Little's Law terms for one boundary (Little 2011, §2).

    enter/leave: per-request boundary entry and exit times (leave may be inf).
    Population: every request present in the window, i.e. it enters before t1 and
    either leaves after t0 or enters at/after t0 (zero-length visits count).
    L   = time-average count in the window
    lam = population size / (t1 - t0)
    W   = mean in-window residence (time clipped to [t0, t1))
    """
    raise NotImplementedError("TODO. HINT: Follow the declared boundary and invariant in the module Feedback Contract.")


def warmup_cutoff(horizon: float, warmup_frac: float) -> float:
    """Declared warm-up rule: discard the first warmup_frac of the horizon."""
    return horizon * warmup_frac


def steady_estimates(arrivals, starts, departures, horizon: float, warmup_frac: float) -> dict:
    """System (arrival->departure) and queue (arrival->start) Little terms measured
    only in [warmup_cutoff, horizon). Nothing that happens before the cutoff may
    affect the result."""
    raise NotImplementedError("TODO. HINT: Follow the declared boundary and invariant in the module Feedback Contract.")


def backlog(arrivals, departures, t: float) -> int:
    """Requests in the system at time t."""
    raise NotImplementedError("TODO. HINT: Follow the declared boundary and invariant in the module Feedback Contract.")
