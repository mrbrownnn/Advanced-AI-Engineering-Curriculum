"""A complete, working-looking paged block manager with ONE real lifecycle bug (BREAK).

All invariant checks pass and no sequence is corrupted. Find the bug with
instrumentation (counters, refcount traces), not by diffing against your own
block_manager.py. Fix it in place; tests/test_break.py tells you when it is gone.
"""
from __future__ import annotations

from collections import Counter, deque


class InvariantError(Exception):
    pass


class OutOfBlocks(Exception):
    pass


def blocks_for(num_tokens: int, block_size: int) -> int:
    return -(-num_tokens // block_size)


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_queue: deque[int] = deque(range(num_blocks))
        self.refcount = [0] * num_blocks
        self.data: list[list[int]] = [[] for _ in range(num_blocks)]
        self.tables: dict[object, list[int]] = {}
        self.lengths: dict[object, int] = {}
        self.shared: set[int] = set()  # blocks that have been mapped by more than one table
        self.cow_events = 0
        self.allocations = 0

    def num_free(self) -> int:
        return len(self.free_queue)

    def _table(self, seq) -> list[int]:
        if seq not in self.tables:
            raise InvariantError(f"unknown sequence {seq!r}")
        return self.tables[seq]

    def tokens(self, seq) -> list[int]:
        return [t for b in self._table(seq) for t in self.data[b]]

    def append_cost(self, seq) -> int:
        table, n = self._table(seq), self.lengths[seq]
        if n % self.block_size == 0:
            return 1
        return 1 if table[-1] in self.shared else 0

    def _take_free_block(self) -> int:
        if not self.free_queue:
            raise OutOfBlocks("free pool empty")
        b = self.free_queue.popleft()
        self.data[b] = []
        self.refcount[b] = 1
        self.shared.discard(b)
        self.allocations += 1
        return b

    def _release(self, block: int) -> None:
        if self.refcount[block] <= 0:
            raise InvariantError(f"double free of block {block}")
        self.refcount[block] -= 1
        if self.refcount[block] == 0:
            self.free_queue.append(block)

    def _write(self, block: int, offset: int, token: int) -> None:
        if self.refcount[block] != 1:
            raise InvariantError(f"write into block {block} with refcount {self.refcount[block]}")
        if offset != len(self.data[block]) or offset >= self.block_size:
            raise InvariantError(f"bad write offset {offset} in block {block}")
        self.data[block].append(token)

    def allocate(self, seq, tokens: list[int]) -> None:
        if seq in self.tables:
            raise InvariantError(f"sequence {seq!r} already allocated")
        if blocks_for(len(tokens), self.block_size) > self.num_free():
            raise OutOfBlocks(f"{seq!r}")
        self.tables[seq] = []
        self.lengths[seq] = 0
        for t in tokens:
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
            if b in self.shared:
                new = self._take_free_block()
                self.data[new] = self.data[b][:offset]
                self._release(b)
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
            self.shared.add(b)
        self.tables[child] = list(table)
        self.lengths[child] = self.lengths[parent]

    def free(self, seq) -> None:
        for b in reversed(self._table(seq)):
            self._release(b)
        del self.tables[seq]
        del self.lengths[seq]

    def check_invariants(self) -> None:
        free_set = set(self.free_queue)
        if len(free_set) != len(self.free_queue):
            raise InvariantError("duplicate in free pool")
        refs = Counter(b for t in self.tables.values() for b in t)
        for b in range(self.num_blocks):
            if b in free_set and refs[b]:
                raise InvariantError(f"block {b} free and mapped")
            if self.refcount[b] != refs[b]:
                raise InvariantError(f"block {b} refcount mismatch")
            if b not in free_set and refs[b] == 0:
                raise InvariantError(f"block {b} leaked")
