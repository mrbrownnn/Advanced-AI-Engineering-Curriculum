"""Prefix cache on top of the Lab B block manager (Lesson 3.6). Reference implementation.

- Identity: block hash = H(parent_hash, block tokens, extra_keys). extra_keys carry
  execution identity (model, adapter) for every block; the tenant salt enters only
  the first block and propagates through the parent chain.
- Only FULL blocks are hashed and shared.
- Residency vs ownership: a cached block with refcount 0 sits in the free queue and
  is still reusable; it is evicted (hash dropped) only when popped for reuse.
  The free queue is LRU: released blocks go to the back, the front is reused first.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from block_manager import BlockManager, InvariantError, blocks_for  # Lab B artifact


@dataclass(frozen=True)
class Identity:
    model: str
    adapter: str | None = None
    salt: str | None = None


def block_hashes(tokens, block_size: int, ident: Identity) -> list[str]:
    """Chained hashes of the full blocks of tokens."""
    out, parent = [], "ROOT"
    for i in range(len(tokens) // block_size):
        extra = (ident.model, ident.adapter, ident.salt if i == 0 else None)
        blk = tuple(tokens[i * block_size:(i + 1) * block_size])
        parent = hashlib.sha256(repr((parent, blk, extra)).encode()).hexdigest()
        out.append(parent)
    return out


class PrefixCachingBlockManager(BlockManager):
    def __init__(self, num_blocks: int, block_size: int):
        super().__init__(num_blocks, block_size)
        self.hash_to_block: dict[str, int] = {}
        self.block_to_hash: dict[int, str] = {}
        self.evictions = 0
        self.hit_tokens = 0
        self.query_tokens = 0

    def _take_free_block(self) -> int:
        b = self.free_queue[0] if self.free_queue else None
        if b is not None and b in self.block_to_hash:  # evict the LRU cached block
            del self.hash_to_block[self.block_to_hash.pop(b)]
            self.evictions += 1
        return super()._take_free_block()

    def longest_hit(self, tokens, ident: Identity) -> list[int]:
        """Cached blocks for the longest full-block prefix, leaving >= 1 token to compute."""
        hit = []
        for h in block_hashes(tokens[:len(tokens) - 1], self.block_size, ident):
            b = self.hash_to_block.get(h)
            if b is None:
                break
            hit.append(b)
        return hit

    def allocate_cached(self, seq, tokens, ident: Identity) -> int:
        """Allocate seq reusing cached prefix blocks; register its full blocks. Return hit tokens."""
        hit = self.longest_hit(tokens, ident)
        self.allocate(seq, tokens, prefix_blocks=hit)
        for h, b in zip(block_hashes(tokens, self.block_size, ident), self.tables[seq]):
            if h not in self.hash_to_block and b not in self.block_to_hash:
                self.hash_to_block[h] = b
                self.block_to_hash[b] = h
        n = len(hit) * self.block_size
        self.hit_tokens += n
        self.query_tokens += len(tokens)
        return n

    def num_cached_free(self) -> int:
        return sum(1 for b in self.free_queue if b in self.block_to_hash)

    def check_invariants(self) -> None:
        super().check_invariants()
        if len(self.hash_to_block) != len(self.block_to_hash):
            raise InvariantError("hash maps out of sync")
        for h, b in self.hash_to_block.items():
            if self.block_to_hash.get(b) != h:
                raise InvariantError(f"hash map disagrees for block {b}")
            if len(self.data[b]) != self.block_size:
                raise InvariantError(f"partial block {b} is indexed")


__all__ = ["Identity", "block_hashes", "PrefixCachingBlockManager", "blocks_for"]
