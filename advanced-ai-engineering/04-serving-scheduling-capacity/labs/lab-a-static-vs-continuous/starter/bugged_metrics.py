"""Iteration-level serving simulator — Module 04 LAB A (reference).

Time unit: milliseconds. Every number this module produces is a consequence of
the injected, declared step-cost model; nothing here is a hardware fact.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------- given parts
@dataclass(frozen=True)
class Request:
    rid: int
    arrival: float  # ms, the declared arrival boundary t0
    n_in: int
    n_out: int


@dataclass
class Record:
    req: Request
    admit: float | None = None                   # t1: start of the iteration that first schedules it
    tokens: list = field(default_factory=list)   # o_1..o_N: emission time of each output token
    chunks: list = field(default_factory=list)   # prefill chunk sizes (LAB B)

    @property
    def finish(self) -> float:
        return self.tokens[-1]


@dataclass(frozen=True)
class LinearCost:
    """Declared synthetic step-cost model (ms).

    T_step = base + per_decode*n_decode + per_prefill*n_prefill + per_chunk*n_chunks
    sched_ms is the scheduler-loop cost of one batch-composition decision.
    """
    base: float = 10.0
    per_decode: float = 0.1
    per_prefill: float = 0.02
    per_chunk: float = 0.0
    sched_ms: float = 0.0

    def __call__(self, n_decode: int, n_prefill: int = 0, n_chunks: int = 0) -> float:
        return (self.base + self.per_decode * n_decode
                + self.per_prefill * n_prefill + self.per_chunk * n_chunks)


@dataclass
class Run:
    records: list   # list[Record], one per request
    steps: list     # list[dict] per iteration: t, dt, n_decode, n_prefill, slots, useful


def _gamma(rng, mean, sd, n):
    if sd == 0:
        return np.full(n, float(mean))
    shape = (mean / sd) ** 2
    return rng.gamma(shape, sd * sd / mean, n)


def make_workload(n, *, seed, rate_per_s=None, n_in_mean=512, n_in_sd=256,
                  n_out_mean=128, n_out_sd=50):
    """Seeded workload. rate_per_s=None means sustained backlog (all arrive at t=0);
    otherwise open-loop Poisson arrivals. Lengths are gamma-distributed, rounded, >= 1."""
    rng = np.random.default_rng(seed)
    if rate_per_s is None:
        arrivals = np.zeros(n)
    else:
        arrivals = np.cumsum(rng.exponential(1000.0 / rate_per_s, n))
    n_in = np.maximum(1, np.rint(_gamma(rng, n_in_mean, n_in_sd, n))).astype(int)
    n_out = np.maximum(1, np.rint(_gamma(rng, n_out_mean, n_out_sd, n))).astype(int)
    return [Request(i, float(arrivals[i]), int(n_in[i]), int(n_out[i])) for i in range(n)]


def quantile(xs, q):
    """Declared empirical quantile convention: nearest-rank (inverted CDF)."""
    return float(np.quantile(np.asarray(xs, dtype=float), q, method="inverted_cdf"))


# ------------------------------------------------------------------- metrics
def ttft(rec: Record) -> float:
    # PLANTED BUG: coordinated-omission cousin; excludes ingress/queue residence.
    return rec.tokens[0] - rec.admit


def queue_time(rec: Record) -> float:
    return rec.admit - rec.req.arrival


def e2e(rec: Record) -> float:
    return rec.tokens[-1] - rec.req.arrival


def itls(rec: Record) -> np.ndarray:
    return np.diff(np.asarray(rec.tokens, dtype=float))


def tpot(rec: Record):
    n = len(rec.tokens)
    if n < 2:
        return None  # N_out = 1 has no TPOT; report as its own class
    return (rec.tokens[-1] - rec.tokens[0]) / (n - 1)


def mean_tpot(records, weighting: str) -> float:
    """Aggregate TPOT over records with N_out >= 2 (Lesson 4.1 / 4.7 Step 2).
    weighting="request": mean of per-request TPOT; "gap": total post-first-output
    time divided by total inter-output gaps."""
    recs = [r for r in records if len(r.tokens) >= 2]
    if weighting == "request":
        return float(np.mean([tpot(r) for r in recs]))
    if weighting == "gap":
        return sum(r.tokens[-1] - r.tokens[0] for r in recs) / sum(len(r.tokens) - 1 for r in recs)
    raise ValueError(weighting)


def bubble_fraction(run: Run) -> float:
    slots = sum(s["slots"] for s in run.steps)
    useful = sum(s["useful"] for s in run.steps)
    return 1.0 - useful / slots


def summarize(run: Run, *, slo_ttft_ms: float, slo_tpot_ms: float) -> dict:
    recs = run.records
    t_start = min(r.req.arrival for r in recs)
    makespan_s = (max(r.finish for r in recs) - t_start) / 1000.0
    tok = sum(len(r.tokens) for r in recs)
    ttfts = [ttft(r) for r in recs]
    tpots = [x for x in (tpot(r) for r in recs) if x is not None]
    gaps = np.concatenate([itls(r) for r in recs]) if recs else np.array([])
    good = sum(1 for r in recs
               if ttft(r) <= slo_ttft_ms and (tpot(r) is None or tpot(r) <= slo_tpot_ms))
    return {
        "n": len(recs),
        "makespan_s": makespan_s,
        "throughput_tok_s": tok / makespan_s,
        "throughput_req_s": len(recs) / makespan_s,
        "goodput_req_s": good / makespan_s,
        "bubble": bubble_fraction(run),
        "ttft_p50_ms": quantile(ttfts, 0.5),
        "ttft_p99_ms": quantile(ttfts, 0.99),
        "tpot_p50_ms": quantile(tpots, 0.5) if tpots else None,
        "tpot_p99_ms": quantile(tpots, 0.99) if tpots else None,
        "itl_p99_ms": quantile(gaps, 0.99) if len(gaps) else None,
    }


# ---------------------------------------------------------------- schedulers
def run_static(reqs, B: int, cost: LinearCost) -> Run:
    """Fixed-membership batches. When the engine is idle it takes up to B arrived
    requests FIFO (it does not wait to fill). One prefill iteration emits token 1
    for every member; then padded decode iterations run until the longest member
    finishes. No request joins a batch already in flight."""
    pending = deque(sorted(reqs, key=lambda r: (r.arrival, r.rid)))
    recs = {r.rid: Record(r) for r in reqs}
    steps, t = [], 0.0
    while pending:
        t = max(t, pending[0].arrival)
        batch = []
        while pending and len(batch) < B and pending[0].arrival <= t:
            batch.append(pending.popleft())
        for r in batch:
            recs[r.rid].admit = t
        t += cost.sched_ms
        dt = cost(0, sum(r.n_in for r in batch))
        steps.append({"t": t, "dt": dt, "n_decode": 0,
                      "n_prefill": sum(r.n_in for r in batch), "slots": B, "useful": len(batch)})
        t += dt
        for r in batch:
            recs[r.rid].tokens.append(t)
        for k in range(1, max(r.n_out for r in batch)):
            dt = cost(len(batch))  # padded: finished slots still occupy the batch
            live = [r for r in batch if k < r.n_out]
            steps.append({"t": t, "dt": dt, "n_decode": len(batch), "n_prefill": 0,
                          "slots": B, "useful": len(live)})
            t += dt
            for r in live:
                recs[r.rid].tokens.append(t)
    return Run([recs[r.rid] for r in reqs], steps)


def run_continuous(reqs, B: int, cost: LinearCost) -> Run:
    """Iteration-level scheduling (Lesson 4.2 steps 1-4). Each iteration pays
    cost.sched_ms, admits waiting requests into free slots (their whole prompt is
    prefilled this iteration and emits token 1), and decodes every running request."""
    pending = deque(sorted(reqs, key=lambda r: (r.arrival, r.rid)))
    recs = {r.rid: Record(r) for r in reqs}
    waiting, running, steps, t = deque(), [], [], 0.0
    while pending or waiting or running:
        while pending and pending[0].arrival <= t:
            waiting.append(pending.popleft())
        if not waiting and not running:
            t = pending[0].arrival
            continue
        new = []
        while waiting and len(running) + len(new) < B:
            new.append(waiting.popleft())
        for r in new:
            recs[r.rid].admit = t
        n_pre = sum(r.n_in for r in new)
        dt = cost.sched_ms + cost(len(running), n_pre)
        steps.append({"t": t, "dt": dt, "n_decode": len(running), "n_prefill": n_pre,
                      "slots": B, "useful": len(running) + len(new)})
        t += dt
        for r in running + new:
            recs[r.rid].tokens.append(t)
        running = [r for r in running + new if len(recs[r.rid].tokens) < r.n_out]
    return Run([recs[r.rid] for r in reqs], steps)
