"""Minimal paged KV block manager (Lessons 3.3-3.4). Fill the TODOs.

Reused by Lab C (prefix caching builds on it) and by Module 04 labs, so keep
this public API stable: num_free, append_cost, allocate, append, fork, free,
touch, tokens, locate, check_invariants. Subclasses may override
_take_free_block (Lab C uses it as the eviction hook).

Each block stores the token ids written into it, so isolation is observable:
if sharing goes wrong, another sequence's tokens() changes.
"""
from __future__ import annotations

from collections import Counter, deque  # noqa: F401  (Counter is handy in check_invariants)


class InvariantError(Exception):
    """A lifecycle invariant would be (or was) violated. Raise it; never corrupt silently."""


class OutOfBlocks(Exception):
    """Not enough free blocks. Raise BEFORE changing any state."""


def blocks_for(num_tokens: int, block_size: int) -> int:
    """Number of blocks needed to hold num_tokens."""
    return -(-num_tokens // block_size)


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be >= 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_queue: deque[int] = deque(range(num_blocks))  # front = next to reuse
        self.refcount = [0] * num_blocks
        self.data: list[list[int]] = [[] for _ in range(num_blocks)]  # token ids per block
        self.tables: dict[object, list[int]] = {}   # seq -> [physical block per logical block]
        self.lengths: dict[object, int] = {}        # seq -> tokens held
        self.cow_events = 0
        self.allocations = 0

    # ---- queries -------------------------------------------------------
    def num_free(self) -> int:
        return len(self.free_queue)

    def _table(self, seq) -> list[int]:
        if seq not in self.tables:
            raise InvariantError(f"unknown sequence {seq!r} (freed or never allocated)")
        return self.tables[seq]

    def tokens(self, seq) -> list[int]:
        return [t for b in self._table(seq) for t in self.data[b]]

    def locate(self, seq, token_index: int) -> tuple[int, int]:
        """(physical_block, offset) of token_index in seq; IndexError if out of range."""
        raise NotImplementedError("HINT: TODO: logical_block = index // B, offset = index % B, look up the table")

    def append_cost(self, seq) -> int:
        """Free blocks that appending one token to seq would consume: 0 or 1.

        A new block is needed when the tail is full; a copy is needed when the
        partial tail is shared. Otherwise the token fits in place.
        """
        raise NotImplementedError("HINT: TODO")

    # ---- block lifecycle -------------------------------------------------
    def _take_free_block(self) -> int:
        """Pop the front of the free pool, clear it, set refcount 1."""
        if not self.free_queue:
            raise OutOfBlocks("free pool empty")
        b = self.free_queue.popleft()
        self.data[b] = []
        self.refcount[b] = 1
        self.allocations += 1
        return b

    def touch(self, block: int) -> None:
        """Add one active reference; a zero-ref block must leave the free pool."""
        raise NotImplementedError("HINT: TODO")

    def _release(self, block: int) -> None:
        """Drop one reference. Releasing a block with refcount 0 is a double free:
        raise InvariantError. A block reaching 0 goes to the back of the free pool."""
        raise NotImplementedError("HINT: TODO")

    def _write(self, block: int, offset: int, token: int) -> None:
        """Guarded write: only a sole owner may write, and only at the next free slot."""
        if self.refcount[block] != 1:
            raise InvariantError(f"write into block {block} with refcount {self.refcount[block]} (needs CoW)")
        if offset != len(self.data[block]) or offset >= self.block_size:
            raise InvariantError(f"write at offset {offset} of block {block} holding {len(self.data[block])}")
        self.data[block].append(token)

    # ---- sequence operations ---------------------------------------------
    def allocate(self, seq, tokens: list[int], prefix_blocks=()) -> None:
        """Create seq holding tokens.

        prefix_blocks: physical blocks (full, already holding tokens[0:k*B]) to
        share instead of recomputing; each gains a reference via touch(). Lab C
        uses this for prefix-cache hits. Check capacity first (OutOfBlocks with
        no state change), then append the remaining tokens one by one.
        """
        if seq in self.tables:
            raise InvariantError(f"sequence {seq!r} already allocated")
        prefix_blocks = list(prefix_blocks)
        B = self.block_size
        covered = len(prefix_blocks) * B
        if covered > len(tokens):
            raise ValueError("prefix longer than tokens")
        for i, b in enumerate(prefix_blocks):
            if self.data[b] != list(tokens[i * B:(i + 1) * B]):
                raise InvariantError(f"prefix block {b} does not hold tokens of logical block {i}")
        revived = sum(1 for b in prefix_blocks if self.refcount[b] == 0)
        if blocks_for(len(tokens), B) - len(prefix_blocks) + revived > self.num_free():
            raise OutOfBlocks(f"{seq!r} needs more blocks than are free")
        for b in prefix_blocks:
            self.touch(b)
        self.tables[seq] = prefix_blocks
        self.lengths[seq] = covered
        for t in tokens[covered:]:
            self.append(seq, t)

    def append(self, seq, token: int) -> None:
        """Append one token to seq.

        - tail full (or no blocks): take a new block, map it at the next logical index
        - partial tail with refcount > 1: copy-on-write — copy the tail into a new
          block, move THIS sequence's reference to the copy (refcount of the
          original drops by one, the table entry is redirected), count cow_events
        - partial tail you own alone: write in place (no copy)
        Raise OutOfBlocks before mutating anything if a block is needed but none is free.
        Use self._write for the actual write.
        """
        raise NotImplementedError("HINT: TODO")

    def fork(self, parent, child) -> None:
        """child shares every block of parent (parallel sampling): refcounts +1, no copy."""
        raise NotImplementedError("HINT: TODO")

    def free(self, seq) -> None:
        """Release every block of seq (tail first) and forget seq."""
        raise NotImplementedError("HINT: TODO")

    # ---- invariants ------------------------------------------------------
    def check_invariants(self) -> None:
        """Raise InvariantError if any of these fail:
        - no block appears twice in the free pool
        - no block is both free and referenced by a block table
        - refcount[b] == number of table entries referencing b (catches stale mappings)
        - every block is either free or referenced (no leaks)
        - len(table) == blocks_for(length); each block holds exactly the tokens the table implies
        """
        raise NotImplementedError("HINT: TODO")
