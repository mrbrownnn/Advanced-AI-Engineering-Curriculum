# LAB E — Pinned vLLM V1 Lifecycle Trace

**Hardware**: TRACE
**Effort**: 2h source trace (counted under `source_trace`) + 1h comparison

Spec: [README §06 LAB E](../../README.md#06-engineering-labs) (Lesson 3.10) and the
Required Artifact in README §09.

## Pin

`vllm-project/vllm` at commit `25b0add7b8a1c944d5c4e364f2de6aa82497a2ad`. Read raw files at
that commit, for example
`https://raw.githubusercontent.com/vllm-project/vllm/25b0add7b8a1c944d5c4e364f2de6aa82497a2ad/vllm/v1/core/block_pool.py`.

Files: `vllm/v1/core/sched/scheduler.py`, `vllm/v1/core/kv_cache_manager.py`,
`vllm/v1/core/kv_cache_coordinator.py`, `vllm/v1/core/single_type_kv_cache_manager.py`,
`vllm/v1/core/block_pool.py`, `vllm/v1/core/kv_cache_utils.py`.

## PREDICT gate

Commit `report.md` §1-2 first: predict at least one lifecycle or cache-residency state in
which the runtime differs from your Lab B/C simulator.

## Do

Answer the source-trace questions in `report.md` with file, symbol, line range and a
quote. Static reading is enough; an instrumented run is optional and needs a runtime you
have access to. Anything not read or executed is `TODO_VERIFY`. Do not cite the removed
`vllm/core/block_manager.py`.

## Submit

`report.md`. The answer key is in `labs-solutions/` (open only after submitting).
