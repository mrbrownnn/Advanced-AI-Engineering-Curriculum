"""Chunked prefill scheduler — Module 04 LAB B (reference).

Reuses Request / Record / Run from LAB A's sim.py. The cost model is any
callable cost(n_decode, n_prefill, n_chunks) -> ms; an optional attribute
`sched_ms` is charged once per iteration.
"""
from __future__ import annotations

import math
from collections import deque

from sim import Record, Run

INF = math.inf


def prefill_cap(n_decode: int, chunk: float, budget: float) -> float:
    """Prefill tokens allowed this iteration (Lesson 4.3 step 3):
    min(C_max, T_budget - B_dec), never negative."""
    return max(0, min(chunk, budget - n_decode))


def run_chunked(reqs, B: int, cost, chunk: float = INF, budget: float = INF) -> Run:
    """Sarathi-style iteration: decodes first, then prefill chunks FIFO from
    admitted-but-unfinished prompts until the token budget or the chunk limit is hit.
    chunk=INF and budget=INF is monolithic (unchunked) prefill.

    A request occupies one of B slots from admission until its last token.
    Requires budget >= B so decodes always fit (decodes are never starved).
    Token 1 is emitted at the end of the iteration that finishes its last chunk.
    A decoding request emits one token per iteration it is scheduled in.
    """
    if budget < B:
        raise ValueError("budget must be >= B so every running decode fits")
    sched_ms = getattr(cost, "sched_ms", 0.0)
    pending = deque(sorted(reqs, key=lambda r: (r.arrival, r.rid)))
    recs = {r.rid: Record(r) for r in reqs}
    left = {}                                  # rid -> prompt tokens not yet prefilled
    waiting, prefilling, decoding = deque(), [], []
    steps, t = [], 0.0
    while pending or waiting or prefilling or decoding:
        while pending and pending[0].arrival <= t:
            waiting.append(pending.popleft())
        if not (waiting or prefilling or decoding):
            t = pending[0].arrival
            continue
        while waiting and len(prefilling) + len(decoding) < B:
            r = waiting.popleft()
            recs[r.rid].admit = t
            left[r.rid] = r.n_in
            prefilling.append(r)
        n_dec = len(decoding)
        cap, sched, n_pre = prefill_cap(n_dec, chunk, budget), [], 0
        for r in prefilling:  # FIFO over admitted prompts
            c = min(left[r.rid], cap - n_pre)
            if c <= 0:
                break
            sched.append((r, c))
            n_pre += c
        dt = sched_ms + cost(n_dec, n_pre, len(sched))
        steps.append({"t": t, "dt": dt, "n_decode": n_dec, "n_prefill": n_pre,
                      "n_chunks": len(sched), "slots": B,
                      "useful": n_dec + sum(1 for r, c in sched if c == left[r.rid])})
        t += dt
        for r in decoding:
            recs[r.rid].tokens.append(t)
        done = []
        for r, c in sched:
            recs[r.rid].chunks.append(c)
            left[r.rid] -= c
            if left[r.rid] == 0:
                recs[r.rid].tokens.append(t)
                done.append(r)
        prefilling = [r for r in prefilling if left[r.rid] > 0]
        decoding = [r for r in decoding + done if len(recs[r.rid].tokens) < r.n_out]
    return Run([recs[r.rid] for r in reqs], steps)
