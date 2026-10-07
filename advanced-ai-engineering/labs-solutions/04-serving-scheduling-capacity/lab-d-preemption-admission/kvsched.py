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
    return -(-tokens // block)


# ------------------------------------------- Lesson 4.5 lower-bound arithmetic
def swap_roundtrip_ms(payload_bytes: float, bw_bytes_per_s: float) -> float:
    """Payload-only out-and-back transfer lower bound (no DMA/sync overhead)."""
    raise_if_nonpositive(bw_bytes_per_s)
    return 2.0 * payload_bytes / bw_bytes_per_s * 1000.0


def recompute_bound_ms(params: float, tokens: int, peak_flops: float) -> float:
    """Dense-matmul recompute lower bound 2*P*tokens/peak (excludes attention work)."""
    raise_if_nonpositive(peak_flops)
    return 2.0 * params * tokens / peak_flops * 1000.0


def breakeven_bw(payload_bytes: float, recompute_ms: float) -> float:
    """Effective bandwidth (bytes/s) at which payload-only round trip equals recompute time."""
    raise_if_nonpositive(recompute_ms)
    return 2.0 * payload_bytes / (recompute_ms / 1000.0)


def raise_if_nonpositive(x: float) -> None:
    if not x > 0:
        raise ValueError("denominator must be > 0")


# ------------------------------------------------------------------- the lab
def run_kv(reqs, cost, pol: Policy):
    """Returns (states, steps). steps[i] keys: t, dt, n_decode, n_prefill, used,
    waiting, preempted, alloc_fail, progress, swap_blocks."""
    sched_ms = getattr(cost, "sched_ms", 0.0)
    pending = deque(sorted(reqs, key=lambda r: (r.arrival, r.rid)))
    st = {r.rid: ReqState(Record(r)) for r in reqs}
    waiting, running, steps = deque(), [], []
    free, t, order = pol.n_blocks, 0.0, 0

    def release(s):
        nonlocal free
        free += s.held
        s.held = 0

    def preempt(s):
        nonlocal free
        running.remove(s)
        release(s)
        s.preemptions += 1
        s.status = "waiting"
        if pol.mode == "recompute":
            s.kv = 0
        else:
            s.swapped = True
        waiting.appendleft(s)
        return blocks_for(s.kv, pol.block) if pol.mode == "swap" else 0

    def pick_victim():
        if pol.victim == "newest":
            return max(running, key=lambda s: s.order)
        if pol.victim == "largest":
            return max(running, key=lambda s: (s.held, s.order))
        raise ValueError(pol.victim)

    while pending or waiting or running:
        # 1. arrivals through the gateway
        while pending and pending[0].arrival <= t:
            r = pending.popleft()
            s = st[r.rid]
            if len(waiting) >= pol.queue_limit or blocks_for(r.n_in + r.n_out, pol.block) > pol.n_blocks:
                s.status = "rejected"
            else:
                waiting.append(s)
        # 2. client cancellations
        for s in list(waiting) + list(running):
            if t >= s.rec.req.arrival + pol.timeout_ms:
                if s in running:
                    running.remove(s)
                else:
                    waiting.remove(s)
                release(s)
                s.status = "cancelled"
        if not (waiting or running):
            if pending:
                t = pending[0].arrival
                continue
            break
        preempted = alloc_fail = swap_blocks = 0
        # 3. grow running requests by one KV token, preempting on failure
        decode = []
        for s in sorted(running, key=lambda s: s.order):
            if s not in running:
                continue  # already preempted this step
            need = blocks_for(s.kv + 1, pol.block) - s.held
            while need > free:
                alloc_fail += 1
                v = pick_victim()
                swap_blocks += preempt(v)
                preempted += 1
                if v in decode:
                    decode.remove(v)
                if v is s:
                    break
            if s in running:
                free -= max(0, need)
                s.held += max(0, need)
                decode.append(s)
        # 4. admit from the waiting queue (preempted requests are at its front)
        new, n_pre = [], 0
        while waiting and len(running) < pol.B:
            s = waiting[0]
            r = s.rec.req
            if s.swapped:
                target = blocks_for(s.kv + 1, pol.block)
            elif pol.admission == "reserve":
                target = blocks_for(r.n_in + r.n_out, pol.block)
            else:
                target = blocks_for(r.n_in + len(s.rec.tokens), pol.block)
            if target > free:
                break  # waiting-path allocation failure stops admission this step
            waiting.popleft()
            free -= target
            s.held = target
            s.status = "running"
            s.order = order
            order += 1
            running.append(s)
            if s.rec.admit is None:
                s.rec.admit = t
            if s.swapped:
                s.swapped = False
                swap_blocks += target
                decode.append(s)
            else:
                p = r.n_in + len(s.rec.tokens)
                if s.rec.tokens:
                    s.recomputed += p
                s.kv = p
                n_pre += p
                new.append(s)
        for s in decode:
            s.kv += 1
        dt = sched_ms + cost(len(decode), n_pre) + pol.swap_ms_per_block * swap_blocks
        t += dt
        for s in decode + new:
            s.rec.tokens.append(t)
        for s in list(running):
            if len(s.rec.tokens) >= s.rec.req.n_out:
                running.remove(s)
                release(s)
                s.status = "done"
        steps.append({"t": t - dt, "dt": dt, "n_decode": len(decode), "n_prefill": n_pre,
                      "used": pol.n_blocks - free, "waiting": len(waiting),
                      "preempted": preempted, "alloc_fail": alloc_fail,
                      "progress": len(decode) + len(new), "swap_blocks": swap_blocks})
    return [st[r.rid] for r in reqs], steps


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
    if not steps:
        return []
    t0 = steps[0]["t"]
    agg = {}
    for s in steps:
        k = int((s["t"] - t0) // window_ms)
        a = agg.setdefault(k, {"pre": 0, "fail": 0, "prog": 0, "dt": 0.0})
        a["pre"] += s["preempted"]
        a["fail"] += s["alloc_fail"]
        a["prog"] += s["progress"]
        a["dt"] += s["dt"]
    flagged = []
    for k, a in sorted(agg.items()):
        rate = a["prog"] / a["dt"] if a["dt"] else 0.0
        if a["pre"] >= min_preempt and a["fail"] >= a["pre"] and rate < drop * baseline_rate:
            flagged.append(t0 + k * window_ms)
    return flagged
