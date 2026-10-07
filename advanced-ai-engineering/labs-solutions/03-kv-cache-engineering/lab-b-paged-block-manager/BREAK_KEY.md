# LAB B — Instructor / BREAK Key

Open only after submitting the lab report.

## Planted bug

`starter/bugged_block_manager.py` remembers every block that has *ever* been shared
in `shared` and uses that history, rather than current `refcount > 1`, to decide
copy-on-write. With N writers sharing one 14-token tail, it makes N copies; only
N-1 are required because the final writer is sole owner. All safety invariants pass,
but allocation/COW counters expose wasted work. Fix: base `append_cost` and the CoW
decision on the current refcount. (`labs-solutions/.../bugged_block_manager.py`
imports the corrected manager.)

## Injected lifecycle faults

- Double release: `_release` rejects refcount <= 0.
- Shared-tail direct write: `_write` rejects refcount != 1.
- Stale table: invariant recomputes table references and compares every refcount.
- Free-and-mapped: invariant checks intersection between free pool and references.
- Leak: invariant rejects a block neither free nor referenced.

## Lesson 3.5 replay

All eight README rows reproduce exactly after rounding mean gaps to two decimals;
no mismatched cells. The key ordering is growth oldest-first, shortfall preemption,
re-admission/prefill, cost, emit/completion. T1/K1 admit preempts C at iteration 17,
discards 111 positions, and re-prefills 112 after A completes.
