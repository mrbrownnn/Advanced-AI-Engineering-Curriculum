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
    raise NotImplementedError("TODO. HINT: Follow the declared boundary and invariant in the module Feedback Contract.")


def run_chunked(reqs, B: int, cost, chunk: float = INF, budget: float = INF) -> Run:
    """Sarathi-style iteration: decodes first, then prefill chunks FIFO from
    admitted-but-unfinished prompts until the token budget or the chunk limit is hit.
    chunk=INF and budget=INF is monolithic (unchunked) prefill.

    A request occupies one of B slots from admission until its last token.
    Requires budget >= B so decodes always fit (decodes are never starved).
    Token 1 is emitted at the end of the iteration that finishes its last chunk.
    A decoding request emits one token per iteration it is scheduled in.
    """
    raise NotImplementedError("TODO. HINT: Follow the declared boundary and invariant in the module Feedback Contract.")
