"""Prefix cache on top of YOUR Lab B block manager (Lesson 3.6). Fill the TODOs.

Imports block_manager from lab-b-paged-block-manager (tests/conftest.py puts it
on sys.path), so Lab B must pass first.

- Identity: block hash = H(parent_hash, block tokens, extra_keys). extra_keys carry
  execution identity (model, adapter) for every block; the tenant salt enters only
  the first block and propagates through the parent chain.
- Only FULL blocks are hashed and shared.
- Residency vs ownership: a cached block with refcount 0 sits in the free queue and
  is still reusable; it is evicted (hash dropped) only when popped for reuse.
  The free queue is LRU: released blocks go to the back, the front is reused first.
"""
from __future__ import annotations

import hashlib  # noqa: F401
from dataclasses import dataclass

from block_manager import BlockManager, InvariantError, blocks_for  # noqa: F401  (Lab B artifact)


@dataclass(frozen=True)
class Identity:
    model: str
    adapter: str | None = None
    salt: str | None = None


def block_hashes(tokens, block_size: int, ident: Identity) -> list[str]:
    """Chained hashes (hex strings) of the FULL blocks of tokens, one per full block.
    Use a stable hash (hashlib), not Python's randomized hash()."""
    raise NotImplementedError("HINT: TODO")


class PrefixCachingBlockManager(BlockManager):
    def __init__(self, num_blocks: int, block_size: int):
        super().__init__(num_blocks, block_size)
        self.hash_to_block: dict[str, int] = {}
        self.block_to_hash: dict[int, str] = {}
        self.evictions = 0
        self.hit_tokens = 0
        self.query_tokens = 0

    def _take_free_block(self) -> int:
        """Eviction hook: if the block about to be popped from the front of the free
        queue is still indexed, drop its hash (count an eviction), then delegate."""
        raise NotImplementedError("HINT: TODO")

    def longest_hit(self, tokens, ident: Identity) -> list[int]:
        """Physical blocks for the longest cached full-block prefix of tokens.
        At least one token must remain to compute (the model needs logits)."""
        raise NotImplementedError("HINT: TODO")

    def allocate_cached(self, seq, tokens, ident: Identity) -> int:
        """Allocate seq reusing the longest hit (BlockManager.allocate(prefix_blocks=...)),
        then index seq's full blocks that are not indexed yet. Update hit_tokens and
        query_tokens. Return hit tokens."""
        raise NotImplementedError("HINT: TODO")

    def num_cached_free(self) -> int:
        """Blocks that are free (refcount 0) but still indexed: reusable residency."""
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
