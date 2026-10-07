# LAB E — Verified Source-Trace Answer Key

Open only after submitting the trace. Static verification date: 2026-10-08.
Repository: `vllm-project/vllm`; commit
`25b0add7b8a1c944d5c4e364f2de6aa82497a2ad`. All named symbols below were read in
raw source at that commit. Dynamic workload branches were not run: `TODO_VERIFY`.

## 1 Scheduler prefix lookup

File `vllm/v1/core/sched/scheduler.py`, symbol `Scheduler.schedule`: prefix lookup
for a new request goes through `_get_local_prefix_cache_hit`, not unconditionally
direct to `get_computed_blocks`. The wrapper can call connector-specific lookup
when divergent local hybrid hits are supported. Lookup runs when
`request.num_computed_tokens == 0`.

File `vllm/v1/core/kv_cache_manager.py`, `prefix_cache_lookup_enabled`: lookup is
enabled only when caching is on and `request.skip_reading_prefix_cache` is false.
The skip flag covers prompt-logprob and all-token pooling paths.

## 2 Allocation refusal and preemption

For a RUNNING request, `schedule` retries `allocate_slots` in a loop. Under FCFS,
the victim is `self.running[-1]`; under PRIORITY it selects the max
`(priority, arrival_time)`. `_preempt_request` frees blocks, resets
`num_computed_tokens = 0`, sets status `PREEMPTED`, and prepends the request to the
waiting queue: recomputation, not swap. Connector-pending frees and blocks that
cannot yet be freed add guards; if the victim is the request itself, scheduling it
stops.

For a WAITING request, `allocate_slots(... full_sequence_must_fit=...,
reserved_blocks=..., has_scheduled_reqs=...)` returning `None` stops the waiting
loop (and releases encoder cache if present). It does not preempt a running request.

## 3 Prefix-hit cap

`KVCacheManager.get_computed_blocks` sets
`max_cache_hit_length = request.num_tokens - 1`: the final token must be recomputed
to obtain logits. Block alignment can cause a whole final block to be recomputed.
It returns `(blocks, num_new_computed_tokens, shared_prefix_boundary)`.

## 4 `allocate_slots` order and feasibility

Order: compute watermark; optional full-sequence admission gate;
`coordinator.remove_skipped_blocks`; `get_num_blocks_to_allocate`; feasibility;
`allocate_new_computed_blocks`; `allocate_new_blocks`; then, unless delayed,
`coordinator.cache_blocks`.

Exact core comparison:

```python
available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
required_blocks = num_blocks_to_allocate + watermark_blocks
if required_blocks > available_blocks:
    return None
```

Lookahead enters the slot token count, capped by `max_model_len`. The configured
watermark applies when there are already scheduled requests and the request status
is WAITING or PREEMPTED. Default/configured values are deployment inputs, not
universal constants.

## 5 New blocks

`BlockPool.get_new_blocks` rejects a request larger than
`get_num_free_blocks()`, then takes `free_block_queue.popleft_n(num_blocks)`. For
each popped block it calls `_maybe_evict_cached_block`, asserts refcount 0, and
increments to 1. A popped cached eviction candidate loses its hash entries before
reuse.

## 6 Touch

`BlockPool.touch`: if refcount is 0 and the block is not null, remove it from the
free queue; then increment refcount. A prefix hit therefore revives a cached
zero-owner block rather than allocating new storage.

## 7 Release and queue order

`BlockPool.free_blocks` decrements each refcount; only blocks reaching 0 rejoin the
queue. Unhashed/non-cached blocks are prepended for LIFO/locality; hashed cached
blocks are appended for FIFO/LRU eviction. `SingleTypeKVCacheManager.free` reverses
a request's blocks so tail blocks are freed first. This preserves the documented
eviction priority: older blocks first, and among equal age, deeper chain tails
first.

## 8 Cached-hash eviction

`BlockPool._maybe_evict_cached_block` calls `_remove_cached_block_hashes`, removing
the primary block hash and any additional hashes registered for that block; it
updates metrics/events and returns whether anything was evicted. Callers include
`get_new_blocks` and `evict_blocks`.

## 9 Hashing and extra keys

`get_request_block_hasher` stops when a block would be incomplete: only full hash
blocks are walked. `hash_block_tokens` hashes parent hash, current token tuple and
extra keys; each result becomes the next parent.

`generate_block_hash_extra_keys` combines:

- LoRA name;
- overlapping multimodal feature identifier and relative offset;
- cache salt only for the first block:
  `[request.cache_salt] if (start_token_idx == 0 and request.cache_salt) else []`;
- SHA-256 of prompt-embedding tensor data for the block.

Salt propagates to later hashes through parent chaining. Model identity remains a
deployment responsibility in this revision.

## 10 Coordinators

`vllm/v1/core/kv_cache_coordinator.py` defines abstract `KVCacheCoordinator`,
`KVCacheCoordinatorNoPrefixCache`, `UnitaryKVCacheCoordinator`, and
`HybridKVCacheCoordinator`. Factory `get_kv_cache_coordinator` chooses: no-prefix
when caching is disabled; unitary for one KV cache group; hybrid otherwise. Which a
run selects depends on its cache configuration: `TODO_VERIFY` without that run.

## 11 Partial CoW: narrow runtime path

Contrary to a full-block-only simplification, partial CoW exists in
`single_type_kv_cache_manager.py`: `_partial_hit_reqs`, `_pending_cow_copies`,
`_apply_cow`, and the `allocate_new_blocks` partial-hit redirect. It is enabled by
`HybridKVCacheCoordinator` only for partial hash hits involving an aligned Mamba
group (`allow_partial_hash_hits` plus the relevant Mamba/block-size conditions).
Plain unitary full-attention prefix caching shares full blocks; Lab B's arbitrary
shared-partial-tail mechanism must not be generalized to it.

## 12 Object mapping

- `KVCacheBlock.ref_cnt`: active logical ownership.
- `FreeKVCacheBlockQueue`: allocatable blocks, including zero-ref cached eviction
  candidates; it is not a set of semantically empty blocks.
- `BlockPool.cached_block_hash_to_block`: reusable-residency index.
- coordinator: policy/layout bridge across cache group managers.
- cache groups: physical/layout compatibility grouping, not request count.

## 13 Invalid simulator behavior

Lab B clears every block popped from the free queue and has no cached residency;
Lab C models all zero-ref cached blocks with one simple LRU order. The runtime's
free queue includes cached eviction candidates, and `free_blocks` gives unhashed
blocks front/LIFO reuse while cached blocks go to the back/FIFO. Therefore "free
queue means empty" and uniform FIFO reuse are both invalid runtime claims.

## 14 Dynamic comparison

`TODO_VERIFY`: no vLLM runtime workload was executed. Cache groups/coordinator,
block IDs, reusable-token counters, allocatable vs cached-evictable counts, and
allocation refusals require an instrumented run under a declared config.
