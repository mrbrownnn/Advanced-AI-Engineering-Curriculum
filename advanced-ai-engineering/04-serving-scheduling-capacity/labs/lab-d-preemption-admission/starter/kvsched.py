"""KV-block-aware continuous batching with preemption and admission control —
Module 04 LAB D (reference). Reuses Request / Record / LinearCost from LAB A sim.py.

Declared model (all synthetic):
- KV is a counter of fixed-size blocks; a request holding k computed tokens holds
  ceil(k / block) blocks. Module 03 LAB B's block manager can replace the counter.
- A prefill of p tokens computes p KV tokens and emits one output token.
  Each decode step computes the KV of the previous token (+1) and emits one token.
- RECOMPUTE preemption frees all blocks; on resume the request re-prefills
  n_in + tokens already emitted, charged to the step cost model.
- SWAP preemption moves blocks to an unbounded host pool; out and in each cost
  swap_ms_per_block per block, added to that step's duration.
- Preempted requests are prepended to the waiting queue (vLLM V1 style).
- A client cancels a request that has not finished `timeout_ms` after arrival.
- The gateway rejects an arrival when the waiting queue already holds
  `queue_limit` requests, or when the request could never fit in the pool.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from sim import Record


# ---------------------------------------------------------------- given parts
@dataclass(frozen=True)
class Policy:
    n_blocks: int                  # KV pool size in blocks
    block: int = 16                # tokens per block
    B: int = 32                    # max running requests
    admission: str = "greedy"      # "greedy": prompt blocks only; "reserve": n_in + n_out up front
    victim: str = "newest"         # "newest" (last admitted) | "largest" (most blocks held)
    mode: str = "recompute"        # "recompute" | "swap"
    swap_ms_per_block: float = 0.5
    queue_limit: float = math.inf  # gateway waiting-queue bound
    timeout_ms: float = math.inf   # client cancellation deadline after arrival


@dataclass
class ReqState:
    rec: Record
    status: str = "waiting"   # waiting | running | done | rejected | cancelled
    kv: int = 0               # KV tokens computed and resident (or on host if swapped)
    held: int = 0             # GPU blocks held
    swapped: bool = False
    order: int = -1           # admission order of current residency
    preemptions: int = 0
    recomputed: int = 0       # prefill tokens spent re-building discarded KV


def blocks_for(tokens: int, block: int) -> int:
    raise NotImplementedError("TODO. HINT: Follow the declared boundary and invariant in the module Feedback Contract.")


# ------------------------------------------- Lesson 4.5 lower-bound arithmetic
def swap_roundtrip_ms(payload_bytes: float, bw_bytes_per_s: float) -> float:
    """Payload-only out-and-back transfer lower bound (no DMA/sync overhead)."""
    raise NotImplementedError("TODO. HINT: Follow the declared boundary and invariant in the module Feedback Contract.")


def recompute_bound_ms(params: float, tokens: int, peak_flops: float) -> float:
    """Dense-matmul recompute lower bound 2*P*tokens/peak (excludes attention work)."""
    raise NotImplementedError("TODO. HINT: Follow the declared boundary and invariant in the module Feedback Contract.")


def breakeven_bw(payload_bytes: float, recompute_ms: float) -> float:
    """Effective bandwidth (bytes/s) at which payload-only round trip equals recompute time."""
    raise NotImplementedError("TODO. HINT: Follow the declared boundary and invariant in the module Feedback Contract.")


def raise_if_nonpositive(x: float) -> None:
    if not x > 0:
        raise ValueError("denominator must be > 0")


# ------------------------------------------------------------------- the lab
def run_kv(reqs, cost, pol: Policy):
    """Returns (states, steps). steps[i] keys: t, dt, n_decode, n_prefill, used,
    waiting, preempted, alloc_fail, progress, swap_blocks."""
    raise NotImplementedError("TODO. HINT: Follow the declared boundary and invariant in the module Feedback Contract.")


def outcomes(states, steps, *, slo_ttft_ms: float) -> dict:
    """Completions, SLO goodput, rejections, cancellations, wasted work, fairness."""
    done = [s for s in states if s.status == "done"]
    span_s = max(1e-9, (steps[-1]["t"] + steps[-1]["dt"] - steps[0]["t"]) / 1000.0) if steps else 1e-9
    good = [s for s in done if s.rec.tokens[0] - s.rec.req.arrival <= slo_ttft_ms]
    wasted = sum(s.recomputed for s in states) + sum(
        s.rec.req.n_in + len(s.rec.tokens) for s in states if s.status == "cancelled" and s.rec.admit is not None)
    slow = np.array([(s.rec.tokens[-1] - s.rec.req.arrival) / s.rec.req.n_out for s in done])
    jain = float(slow.sum() ** 2 / (len(slow) * (slow ** 2).sum())) if len(slow) else None
    return {
        "n": len(states), "completed": len(done), "good": len(good),
        "goodput_req_s": len(good) / span_s,
        "rejected": sum(s.status == "rejected" for s in states),
        "cancelled": sum(s.status == "cancelled" for s in states),
        "preemptions": sum(s.preemptions for s in states),
        "wasted_tokens": int(wasted),
        "swap_blocks": int(sum(s["swap_blocks"] for s in steps)),
        "jain_slowdown": jain,
    }


def detect_thrashing(steps, *, window_ms: float, baseline_rate: float,
                     drop: float = 0.5, min_preempt: int = 1) -> list:
    """Flag time windows that show thrashing (Lesson 4.5, CLM-009b): preemption churn
    that is (a) aligned with KV allocation failures and (b) accompanied by useful
    progress (new output tokens per ms) below `drop` x the declared baseline rate.
    Churn alone is not thrashing. Returns list of window start times (ms)."""
    raise NotImplementedError("TODO. HINT: Follow the declared boundary and invariant in the module Feedback Contract.")
