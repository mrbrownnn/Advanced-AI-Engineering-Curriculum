# Lab Report — Module 03, LAB E: Pinned vLLM V1 Lifecycle Trace

> Fill sections 1–2 and **commit before running any experiment**. The git log is
> your evidence that the prediction came first. Do not edit sections 1–2 after
> running; record disagreements in section 5.

## Source-trace questions (answer in sections 4-7)

Pin: `vllm-project/vllm` at `25b0add7b8a1c944d5c4e364f2de6aa82497a2ad`. For every
answer give file, symbol, line range, and a short quote. Mark anything you did not
read or execute `TODO_VERIFY`. Do not use the removed `vllm/core/block_manager.py`.

1. `Scheduler.schedule` (`vllm/v1/core/sched/scheduler.py`): where does the prefix lookup
   happen for a waiting request, and is `KVCacheManager.get_computed_blocks` called directly
   or through another method? Under which condition is the lookup skipped?
2. When `allocate_slots` returns `None` for a RUNNING request, what does the scheduler do?
   Which request is chosen as victim, does the choice depend on the scheduling policy, and is
   the victim's KV swapped or discarded for recomputation? What happens instead for a WAITING request?
3. `KVCacheManager.get_computed_blocks`: what is the maximum number of tokens a cache hit may
   cover, and why can it not cover the whole prompt?
4. `KVCacheManager.allocate_slots`: list the calls in order. Write the exact feasibility
   comparison. Which headroom terms (watermark, reserved blocks, lookahead) enter it, and when
   does the watermark apply?
5. `BlockPool.get_new_blocks`: where do blocks come from, what happens to a popped block that is
   still indexed by hash, and what refcount does it leave with?
6. `BlockPool.touch`: what happens to a block whose refcount is 0?
7. `BlockPool.free_blocks` and the caller's release order: which freed blocks go to the front of
   the free queue and which to the back? Why are a request's blocks freed in reverse order?
8. `BlockPool._maybe_evict_cached_block`: what does it remove, and where is it called?
9. `get_request_block_hasher`, `hash_block_tokens`, `generate_block_hash_extra_keys`
   (`vllm/v1/core/kv_cache_utils.py`): are partial blocks hashed? How is the parent hash chained?
   List every extra-key source and the condition under which the cache salt is included.
10. `vllm/v1/core/kv_cache_coordinator.py`: which coordinator classes exist and which factory
    selects among them under which conditions? Which one would your run select, and why?
11. Does this revision ever copy-on-write a shared partial block? Under which configuration, if
    any? What does that imply for your Lab B CoW mechanism as a model of this runtime?
12. Map each implementation object (`KVCacheBlock.ref_cnt`, `FreeKVCacheBlockQueue`,
    `cached_block_hash_to_block`, coordinator, cache groups) to a Module 03 concept.
13. BREAK / FALSIFY: name one behavior of your Lab B/C simulator that is NOT valid for this
    runtime (for example, treating every free-queue block as semantically empty, or the free
    order/eviction order), with the source line that shows the difference.
14. Optional instrumented run (only if you have a runtime available; no lab requires a GPU):
    prefix caching on vs off on a small workload. Record cache groups/coordinator, block IDs,
    allocatable vs cached-evictable blocks, reusable-token hits and any allocation refusal.
    Every unexecuted branch stays `TODO_VERIFY`.

## 1 PREDICTION (commit before running)

- **Claim under test** (atomic; population, boundary, estimand; label O/D/H):
- **Predicted result** (quantitative, with units and direction):
- **Why** (mechanism you expect):
- **Falsifier** (the observation that would prove the prediction wrong):
- **Confidence** (0–100%):

## 2 DESIGN

- Independent variables:
- Dependent variables:
- Controlled variables:
- Experimental unit and `n` (what was independently assigned?):
- Confounders and mitigation:
- Practical threshold (smallest effect that changes a decision):

## 3 ENVIRONMENT

```yaml
hardware: CPU | COLAB(<gpu>) | TRACE
os: _
python: _
packages: _          # pinned versions
commit: _            # git rev-parse HEAD
dirty_tree: _        # git status --porcelain empty?
seeds: _
```

## 4 RESULTS

Raw data path(s), checksums, and the aggregation command. Tables with
uncertainty at the correct unit. State quantile convention and failure
semantics.

## 5 PREDICTION vs OBSERVATION

| Prediction | Observed | Gap | Explanation of gap |
|---|---|---|---|

Was the falsifier triggered? What did you learn that you did not expect?

## 6 BREAK

What you broke, what the system did, which invariant/test caught it (or why
nothing did).

## 7 EXPLAIN / IMPROVE / FALSIFY

Competing explanations, the discriminating measurement, the change you made,
and the same-boundary remeasurement.

## 8 LIMITATIONS & TRANSFER

What this result does **not** support. Under which workload would the
conclusion change?
