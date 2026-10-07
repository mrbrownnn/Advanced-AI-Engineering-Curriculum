"""LAB A reference: event-level harness on a virtual clock.

Units: seconds on one virtual clock. Quantile convention: nearest rank,
rank = ceil(q * n), 1-based. Timeouts stay in the denominator, sort after
every success, and are reported at the timeout value with censored=True.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Event:
    idx: int
    intended: float  # time the generator meant to issue the request
    actual: float    # time the request was really issued
    start: float     # time the server began service (server-side)
    end: float       # time the client observed the outcome
    status: str      # "ok" | "timeout"


class VirtualClock:
    """Injectable clock. Never reads wall time, so runs are deterministic."""

    def __init__(self, t0: float = 0.0):
        self.t = t0

    def now(self) -> float:
        return self.t

    def sleep_until(self, t: float) -> None:
        self.t = max(self.t, t)


class NoisyOp:
    """Single FIFO server with lognormal service time, optional cold start and stall.

    The server is frozen during [stall_at, stall_at + stall_s): work that would
    run inside the window is pushed past it. The client cannot cancel work, so a
    timed-out request still occupies the server.
    """

    def __init__(self, seed: int, base_s: float = 0.010, jitter: float = 0.3,
                 stall_at: float | None = None, stall_s: float = 2.0,
                 cold_requests: int = 0, cold_factor: float = 5.0):
        self.rng = np.random.default_rng(seed)
        self.base_s, self.jitter = base_s, jitter
        self.stall_at, self.stall_s = stall_at, stall_s
        self.cold_requests, self.cold_factor = cold_requests, cold_factor
        self.free_at = 0.0
        self.served = 0

    def submit(self, t_issue: float) -> tuple[float, float]:
        """Return (service start, service end) for a request issued at t_issue."""
        start = max(t_issue, self.free_at)
        s = self.base_s * self.rng.lognormal(0.0, self.jitter)
        if self.served < self.cold_requests:
            s *= self.cold_factor
        self.served += 1
        end = start + s
        if self.stall_at is not None:
            lo, hi = self.stall_at, self.stall_at + self.stall_s
            if lo <= start < hi:
                end = hi + s
            elif start < lo < end:
                end += self.stall_s
        self.free_at = end
        return start, end


def issue_one(op: NoisyOp, k: int, intended: float, actual: float, timeout: float) -> Event:
    """Submit one request issued at `actual`; mark it timeout if the client
    would wait longer than `timeout` after actual issue."""
    start, end = op.submit(actual)
    if end - actual > timeout:
        return Event(k, intended, actual, start, actual + timeout, "timeout")
    return Event(k, intended, actual, start, end, "ok")


def run_open_loop(op: NoisyOp, clock: VirtualClock, period: float,
                  duration: float, timeout: float) -> list[Event]:
    """Issue at intended times k*period for k*period < duration, independent of completions."""
    events = []
    k = 0
    while k * period < duration:
        intended = k * period
        clock.sleep_until(intended)
        events.append(issue_one(op, k, intended, clock.now(), timeout))
        k += 1
    return events


def run_closed_loop(op: NoisyOp, clock: VirtualClock, period: float,
                    duration: float, timeout: float) -> list[Event]:
    """One outstanding request; the next intends to go at prev actual + period
    but cannot go before the previous outcome is observed. Stops when the
    actual issue time reaches duration."""
    events = []
    k = 0
    intended = 0.0
    while True:
        clock.sleep_until(intended)
        actual = clock.now()
        if actual >= duration:
            break
        ev = issue_one(op, k, intended, actual, timeout)
        events.append(ev)
        clock.sleep_until(ev.end)
        intended = actual + period
        k += 1
    return events


def nearest_rank(values, q: float) -> float:
    """Nearest-rank quantile: the ceil(q*n)-th smallest value (1-based)."""
    if not 0.0 < q <= 1.0:
        raise ValueError("q must be in (0, 1]")
    xs = sorted(values)
    if not xs:
        raise ValueError("empty sample")
    return xs[max(1, math.ceil(q * len(xs))) - 1]


def summarize(events: list[Event], origin: str, timeout: float,
              quantiles=(0.5, 0.95, 0.99)) -> dict:
    """Aggregate over ALL issued events, measuring latency from `origin`."""
    if origin not in ("intended", "actual"):
        raise ValueError("origin must be 'intended' or 'actual'")
    ok = [e.end - getattr(e, origin) for e in events if e.status == "ok"]
    n_timeout = sum(e.status == "timeout" for e in events)
    values = ok + [math.inf] * n_timeout
    out = {
        "n_issued": len(events),
        "n_ok": len(ok),
        "n_timeout": n_timeout,
        "failure_rate": n_timeout / len(events),
        "mean_ok_s": float(np.mean(ok)) if ok else math.nan,
        "quantiles": {},
        "censored": {},
    }
    for q in quantiles:
        v = nearest_rank(values, q)
        out["censored"][q] = math.isinf(v)
        out["quantiles"][q] = timeout if math.isinf(v) else v
    return out
