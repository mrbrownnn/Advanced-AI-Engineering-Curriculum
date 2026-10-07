"""Minimal paged KV block manager (Lessons 3.3-3.4). Reference implementation.

Reused by Lab C (prefix caching subclasses it) and Module 04 labs. Public API:
num_free, append_cost, allocate, append, fork, free, touch, tokens, locate,
check_invariants. Subclasses may override _take_free_block (eviction hook).

Each block stores the token ids written into it, so isolation is observable:
if sharing goes wrong, another sequence's tokens() changes.
"""
from __future__ import annotations

from collections import Counter, deque


class InvariantError(Exception):
    """A lifecycle invariant would be (or was) violated."""


class OutOfBlocks(Exception):
    """Not enough free blocks; no state was changed."""


def blocks_for(num_tokens: int, block_size: int) -> int:
    return -(-num_tokens // block_size)


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be >= 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_queue: deque[int] = deque(range(num_blocks))  # front = next to reuse
        self.refcount = [0] * num_blocks
        self.data: list[list[int]] = [[] for _ in range(num_blocks)]
        self.tables: dict[object, list[int]] = {}
        self.lengths: dict[object, int] = {}
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
        if not 0 <= token_index < self.lengths.get(seq, 0):
            raise IndexError(token_index)
        return self._table(seq)[token_index // self.block_size], token_index % self.block_size

    def append_cost(self, seq) -> int:
        """Free blocks that appending one token to `seq` would consume (new block or CoW copy)."""
        table, n = self._table(seq), self.lengths[seq]
        if n % self.block_size == 0:
            return 1
        return 1 if self.refcount[table[-1]] > 1 else 0

    # ---- block lifecycle -------------------------------------------------
    def _take_free_block(self) -> int:
        if not self.free_queue:
            raise OutOfBlocks("free pool empty")
        b = self.free_queue.popleft()
        self.data[b] = []
        self.refcount[b] = 1
        self.allocations += 1
        return b

    def touch(self, block: int) -> None:
        """Add one active reference; a zero-ref block leaves the free pool."""
        if self.refcount[block] == 0:
            self.free_queue.remove(block)  # ponytail: O(n); use a linked list for large pools
        self.refcount[block] += 1

    def _release(self, block: int) -> None:
        if self.refcount[block] <= 0:
            raise InvariantError(f"double free of block {block}")
        self.refcount[block] -= 1
        if self.refcount[block] == 0:
            self.free_queue.append(block)

    def _write(self, block: int, offset: int, token: int) -> None:
        if self.refcount[block] != 1:
            raise InvariantError(f"write into block {block} with refcount {self.refcount[block]} (needs CoW)")
        if offset != len(self.data[block]) or offset >= self.block_size:
            raise InvariantError(f"write at offset {offset} of block {block} holding {len(self.data[block])}")
        self.data[block].append(token)

    # ---- sequence operations ---------------------------------------------
    def allocate(self, seq, tokens: list[int], prefix_blocks=()) -> None:
        """Create `seq` holding `tokens`; optionally start from shared full `prefix_blocks`."""
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
        table, n = self._table(seq), self.lengths[seq]
        if self.append_cost(seq) > self.num_free():
            raise OutOfBlocks(f"append to {seq!r}")
        offset = n % self.block_size
        if offset == 0:
            b = self._take_free_block()
            table.append(b)
        else:
            b = table[-1]
            if self.refcount[b] > 1:  # shared partial tail: copy, then redirect this table only
                new = self._take_free_block()
                self.data[new] = self.data[b][:offset]
                self.refcount[b] -= 1  # stays >= 1: another table still references b
                table[-1] = new
                self.cow_events += 1
                b = new
        self._write(b, offset, token)
        self.lengths[seq] = n + 1

    def fork(self, parent, child) -> None:
        if child in self.tables:
            raise InvariantError(f"sequence {child!r} already allocated")
        table = self._table(parent)
        for b in table:
            self.refcount[b] += 1
        self.tables[child] = list(table)
        self.lengths[child] = self.lengths[parent]

    def free(self, seq) -> None:
        table = self._table(seq)
        for b in reversed(table):  # tail first: tail blocks become reuse candidates first
            self._release(b)
        del self.tables[seq]
        del self.lengths[seq]

    # ---- invariants ------------------------------------------------------
    def check_invariants(self) -> None:
        B = self.block_size
        free = list(self.free_queue)
        free_set = set(free)
        if len(free) != len(free_set):
            raise InvariantError("a block appears twice in the free pool")
        if set(self.tables) != set(self.lengths):
            raise InvariantError("tables and lengths track different sequences")
        refs = Counter(b for t in self.tables.values() for b in t)
        for b in range(self.num_blocks):
            if b in free_set and refs[b]:
                raise InvariantError(f"block {b} is both free and mapped by a block table")
            if self.refcount[b] != refs[b]:
                raise InvariantError(f"block {b}: refcount {self.refcount[b]} != {refs[b]} table references (stale mapping?)")
            if b not in free_set and refs[b] == 0:
                raise InvariantError(f"block {b} leaked: not free and not referenced")
        for seq, table in self.tables.items():
            n = self.lengths[seq]
            if len(table) != blocks_for(n, B):
                raise InvariantError(f"{seq!r}: {len(table)} blocks for {n} tokens")
            for i, b in enumerate(table):
                want = min(B, n - i * B)
                if len(self.data[b]) != want:
                    raise InvariantError(f"{seq!r}: block {b} holds {len(self.data[b])} tokens, table implies {want}")
