"""Contiguous KV reservation allocator + waste classifier (Lesson 3.2). Reference implementation.

Address space is measured in token slots. Each request gets ONE contiguous
extent of `reserve` slots (rounded up to block_size) at admission.

Accounting identity at every instant:
    allocated = logical + internal_fragmentation + reservation_waste
- logical: tokens actually held
- internal_fragmentation: block rounding of what is held, B*ceil(L/B) - L
- reservation_waste: reserved but beyond the held blocks
Admission failures are classified as
- "capacity": total free < request
- "contiguous": total free >= request but no single hole fits (external fragmentation)
"""
from __future__ import annotations

from dataclasses import dataclass


def round_up(n: int, block: int) -> int:
    return -(-n // block) * block


class ContiguousAllocator:
    def __init__(self, capacity: int, block_size: int = 1):
        self.capacity = capacity
        self.block_size = block_size
        self.extents: dict[object, tuple[int, int]] = {}  # id -> (start, size)

    def holes(self) -> list[tuple[int, int]]:
        """Free (start, size) gaps in address order."""
        out, cur = [], 0
        for start, size in sorted(self.extents.values()):
            if start > cur:
                out.append((cur, start - cur))
            cur = start + size
        if cur < self.capacity:
            out.append((cur, self.capacity - cur))
        return out

    def free_total(self) -> int:
        return self.capacity - sum(s for _, s in self.extents.values())

    def reserve(self, rid, tokens: int) -> int | str:
        """First-fit. Return start offset, or the failure class 'capacity' / 'contiguous'."""
        if rid in self.extents:
            raise ValueError(f"{rid!r} already reserved")
        size = round_up(tokens, self.block_size)
        for start, hole in self.holes():
            if hole >= size:
                self.extents[rid] = (start, size)
                return start
        return "capacity" if self.free_total() < size else "contiguous"

    def release(self, rid) -> None:
        del self.extents[rid]


def classify(alloc: ContiguousAllocator, lengths: dict) -> dict:
    """Waste breakdown for the requests currently reserved, given their held lengths."""
    B = alloc.block_size
    logical = sum(lengths[r] for r in alloc.extents)
    held_blocks = sum(round_up(lengths[r], B) for r in alloc.extents)
    allocated = sum(s for _, s in alloc.extents.values())
    holes = alloc.holes()
    return {
        "logical": logical,
        "allocated": allocated,
        "internal_fragmentation": held_blocks - logical,
        "reservation_waste": allocated - held_blocks,
        "free": alloc.free_total(),
        "largest_hole": max((h for _, h in holes), default=0),
        "num_holes": len(holes),
    }


@dataclass
class Request:
    rid: str
    arrival: int      # step at which it joins the queue
    prompt: int
    output: int
    max_len: int      # what a max-length policy reserves

    @property
    def final(self) -> int:
        return self.prompt + self.output


def simulate(requests: list[Request], capacity: int, policy: str = "max_len", block_size: int = 1,
             max_steps: int = 100_000) -> dict:
    """Step simulation. FIFO admission (head-of-line blocks); each running request
    grows one token per step from its prompt; completion releases its extent.

    policy: "max_len" reserves max_len; "exact" reserves the final length (oracle).
    """
    alloc = ContiguousAllocator(capacity, block_size)
    pending = sorted(requests, key=lambda r: (r.arrival, r.rid))
    queue: list[Request] = []
    lengths: dict[str, int] = {}
    running: dict[str, Request] = {}
    failures = {"capacity": 0, "contiguous": 0}
    snapshots, completed = [], []
    for step in range(max_steps):
        while pending and pending[0].arrival <= step:
            queue.append(pending.pop(0))
        while queue:
            r = queue[0]
            want = r.max_len if policy == "max_len" else r.final
            if want < r.final:
                raise ValueError(f"{r.rid}: reservation {want} < final length {r.final}")
            res = alloc.reserve(r.rid, want)
            if isinstance(res, str):
                failures[res] += 1
                break
            queue.pop(0)
            running[r.rid] = r
            lengths[r.rid] = r.prompt
        snap = classify(alloc, lengths)
        snap.update(step=step, running=len(running), queued=len(queue))
        snapshots.append(snap)
        for rid in list(running):
            r = running[rid]
            if lengths[rid] >= r.final:
                alloc.release(rid)
                del running[rid]
                completed.append(rid)
            else:
                lengths[rid] += 1
        if not pending and not queue and not running:
            break
    else:
        raise RuntimeError("max_steps exceeded")
    return {"snapshots": snapshots, "failures": failures, "completed": completed}
