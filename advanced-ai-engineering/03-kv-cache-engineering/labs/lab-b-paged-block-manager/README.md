# LAB B — Minimal Paged KV Block Manager

**Hardware**: CPU
**Effort**: 4h

Spec: [README §06 LAB B](../../README.md#06-engineering-labs) (Lessons 3.3, 3.4, 3.5).

**Artifact reuse:** `starter/block_manager.py` is imported by Lab C (prefix caching
subclasses it) and by later Module 04 labs. Keep its public API stable:
`num_free, append_cost, allocate, append, fork, free, touch, tokens, locate,
check_invariants`, plus the `_take_free_block` eviction hook.

## PREDICT gate

Commit `report.md` §1-2 before running `experiments/run_timeline.py`.

## What to build (`starter/`)

- `block_manager.py`: block pool, block tables, refcounts, copy-on-write for a shared
  partial tail, guarded writes, and `check_invariants` (no block both free and mapped;
  refcount == table references; no leaks; table length and block contents match the
  sequence length). CoW is the PagedAttention reference mechanism, not a claim about
  every current runtime.
- `timeline.py`: `replay` drives your block manager through the Lesson 3.5 worked
  example (T1/T2 x K1/K2 x admit-now with youngest-preempt-recompute vs delay), plus
  your own `my_cost`. `fixtures/lesson35_table.json` is the README's published table.

Tests include a hypothesis property test: random allocate/append/fork/free scripts must
keep every invariant and never change one sequence's tokens through another.

## BREAK

1. Faults that must raise `InvariantError`, never corrupt silently: double-free, a write
   through a shared mutable tail, a stale block-table entry, a block both free and
   mapped, a leaked block.
2. Planted bug: `starter/bugged_block_manager.py` passes every invariant check and keeps
   sequences isolated, yet has one real lifecycle bug. Find it with instrumentation
   (counters, refcount traces), not by diffing; fix it in place. `tests/test_break.py`
   passes once it is gone.

## Run

```bash
cd 03-kv-cache-engineering/labs/lab-b-paged-block-manager
python -m pytest tests -q                     # your starter/
python experiments/run_timeline.py            # results/<stamp>-timeline/{raw.jsonl,result.yaml}
```

Instructor: `LAB_IMPL=solution python -m pytest tests -q`.

## Submit

`report.md` (sections 1-2 committed before any experiment), your `starter/`, and the
`results/` run directories you cite. Results are append-only.
