"""Lesson 3.5 timeline replayer driving your block manager. Fill the TODOs.

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
Mean gap = mean duration of the iterations in which the request emitted a token.
"""
from __future__ import annotations

from collections import deque  # noqa: F401
from dataclasses import dataclass, field

from block_manager import BlockManager, blocks_for  # noqa: F401


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
    """Your own step-cost model (Lab B action). Must be positive and non-decreasing
    in prefill_tokens. Record its justification in report.md."""
    raise NotImplementedError("HINT: TODO: define a third cost model")


def lesson35(timeline: str) -> tuple[list[Req], list[Req]]:
    """(running, waiting) for timeline T1 or T2. A holds 6 blocks, B 8, C needs 6."""
    a_out = {"T1": 64, "T2": 12}[timeline]
    return ([Req("A", 96, a_out, resident=True), Req("B", 128, 128, resident=True)],
            [Req("C", 96, 32)])


def replay(running: list[Req], waiting: list[Req], cost, *, delay_until: str | None = None,
           free_blocks: int = 10, block_size: int = 16, max_iters: int = 100_000) -> dict:
    """Replay iterations until every request is done.

    Build a BlockManager with `free_blocks` free after the running requests'
    resident prompts are allocated. Drive it with append/allocate/free; call
    check_invariants() every iteration.

    Return {"preemptions": int,
            "requests": {name: {"mean_gap_ms", "ttft_ms", "done_ms", "tokens"}},
            "trace": [{"iteration", "ms", "t_ms", "running", "free_blocks", "events"}]}
    """
    raise NotImplementedError("HINT: TODO")
