"""Lesson 3.5 timeline replayer driving the Lab B block manager. Reference implementation.

Rules, as stated in the Lesson 3.5 worked example:
1. Growth: each iteration, every running request (oldest first) writes one KV
   position; it needs a new block when that position is a multiple of block_size.
2. Shortfall: preempt the youngest running request with recomputation (free all
   its blocks; it re-queues at the head and later re-prefills prompt + emitted).
3. Admission: after growth, admit waiting requests FIFO while their prefill
   blocks fit; under the delay policy only once the gate request has completed.
4. Cost: iteration time = cost(resident batch, prefill tokens this iteration).
5. Emit: every running request emits one token at the end of the iteration; a
   request that emitted all its tokens frees its blocks.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from block_manager import BlockManager, blocks_for


@dataclass
class Req:
    name: str
    prompt_tokens: int
    output_tokens: int
    resident: bool = False   # KV for the prompt already cached when the replay starts
    emitted: int = 0
    gaps: list = field(default_factory=list)
    ttft_ms: float | None = None
    done_ms: float | None = None


def k1(batch: int, prefill_tokens: int) -> float:
    if not 1 <= batch <= 3:
        raise ValueError("K1 is only defined for batch 1-3")
    return 20.0 + 0.5 * prefill_tokens


def k2(batch: int, prefill_tokens: int) -> float:
    return 14.0 + 6.0 * batch + 0.5 * prefill_tokens


def my_cost(batch: int, prefill_tokens: int) -> float:
    """Example third cost model: per-iteration fixed cost, mild batch term, and a
    prefill term with a chunk overhead every 64 tokens. Learners choose their own."""
    return 18.0 + 2.0 * batch + 0.5 * prefill_tokens + 4.0 * blocks_for(prefill_tokens, 64)


def lesson35(timeline: str) -> tuple[list[Req], list[Req]]:
    """(running, waiting) for timeline T1 or T2. A holds 6 blocks, B 8, C needs 6."""
    a_out = {"T1": 64, "T2": 12}[timeline]
    return ([Req("A", 96, a_out, resident=True), Req("B", 128, 128, resident=True)],
            [Req("C", 96, 32)])


def replay(running: list[Req], waiting: list[Req], cost, *, delay_until: str | None = None,
           free_blocks: int = 10, block_size: int = 16, max_iters: int = 100_000) -> dict:
    B = block_size
    bm = BlockManager(free_blocks + sum(blocks_for(r.prompt_tokens, B) for r in running), B)
    for r in running:
        bm.allocate(r.name, [0] * r.prompt_tokens)
    running, waiting = list(running), deque(waiting)
    reqs = {r.name: r for r in [*running, *waiting]}
    finished: set[str] = set()
    now, preemptions, trace = 0.0, 0, []
    it = 0
    while running or waiting:
        it += 1
        if it > max_iters:
            raise RuntimeError("max_iters exceeded")
        events = []
        for r in list(running):                      # 1. growth, oldest first
            if r not in running:
                continue
            while bm.append_cost(r.name) > bm.num_free():
                victim = running.pop()               # 2. youngest
                events.append({"preempt": victim.name, "discarded_positions": bm.lengths[victim.name]})
                bm.free(victim.name)
                waiting.appendleft(victim)
                preemptions += 1
                if victim is r:
                    break
            if r in running:
                bm.append(r.name, 0)
        prefill = 0                                  # 3. admission
        if delay_until is None or delay_until in finished:
            while waiting and blocks_for(_context(waiting[0]), B) <= bm.num_free():
                r = waiting.popleft()
                bm.allocate(r.name, [0] * _context(r))
                prefill += _context(r)
                running.append(r)
                events.append({"admit": r.name, "prefill_tokens": _context(r)})
        if not running:
            raise RuntimeError(f"iteration {it}: nothing can run")
        dt = cost(len(running), prefill)             # 4. cost
        now += dt
        for r in list(running):                      # 5. emit
            r.emitted += 1
            r.gaps.append(dt)
            if r.ttft_ms is None:
                r.ttft_ms = now
            if r.emitted == r.output_tokens:
                r.done_ms = now
                bm.free(r.name)
                running.remove(r)
                finished.add(r.name)
        bm.check_invariants()
        trace.append({"iteration": it, "ms": dt, "t_ms": now, "running": [r.name for r in running],
                      "free_blocks": bm.num_free(), "events": events})
    return {
        "preemptions": preemptions,
        "requests": {n: {"mean_gap_ms": sum(r.gaps) / len(r.gaps), "ttft_ms": r.ttft_ms,
                         "done_ms": r.done_ms, "tokens": r.emitted} for n, r in reqs.items()},
        "trace": trace,
    }


def _context(r: Req) -> int:
    """Tokens to (re)prefill on admission: the prompt plus anything emitted before preemption."""
    return r.prompt_tokens + r.emitted
