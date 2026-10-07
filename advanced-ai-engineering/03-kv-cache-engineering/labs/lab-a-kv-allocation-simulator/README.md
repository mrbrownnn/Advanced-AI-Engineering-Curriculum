# LAB A — KV Analytical Model + Allocation Simulator

**Hardware**: CPU
**Effort**: 3h

Spec: [README §06 LAB A](../../README.md#06-engineering-labs) (Lessons 3.1, 3.2). This
folder adds code, fixtures and tests; it does not change the spec.

## PREDICT gate

Commit `report.md` §1-2 before running `experiments/`. Predict the dominant waste
category per reservation policy and whether contiguous-allocation failures appear.

## What to build (`starter/`)

- `kv_model.py`: logical bytes/token for MHA/GQA/MQA and MLA; payload-only concurrency bound.
- `allocator.py`: `ContiguousAllocator.holes` / `reserve` (first-fit; failure classes
  `capacity` vs `contiguous`) and `classify` (logical, internal fragmentation,
  reservation waste). `simulate` is provided.

Tests check invariants (accounting identity, disjoint extents, holes sum to free space)
plus the README's own worked numbers. Fixtures: `fixtures/make_trace.py` (seeded).

## BREAK

Equal fixed-length sequences with enough contiguous space must show zero fragmentation
(`test_break_equal_fixed_lengths_have_zero_fragmentation`). If not, inspect your
invariant or definition.

## Run

```bash
cd 03-kv-cache-engineering/labs/lab-a-kv-allocation-simulator
python -m pytest tests -q                     # your starter/
python experiments/run_waste_sweep.py         # results/<stamp>-waste-sweep/{raw.jsonl,result.yaml}
```

Instructor: `LAB_IMPL=solution python -m pytest tests -q`.

## Submit

`report.md` (sections 1-2 committed before any experiment), your `starter/`, and the
`results/` run directories you cite. Results are append-only.
