# LAB D — Compression, Tiering, & Disaggregation Bounds

**Hardware**: CPU (analytical; the README's optional single-GPU latency measurement is out of scope)
**Effort**: 3h

Spec: [README §06 LAB D](../../README.md#06-engineering-labs) (Lessons 3.8, 3.9, 3.10).
Every output is a screening bound, not a measurement.

## PREDICT gate

Commit `report.md` §1-2 before running `experiments/run_bounds.py`.

## What to build (`starter/bounds.py`)

Payload + quantization-metadata model, concurrency screening bound, transfer lower
bound with an explicit `Bandwidth(value, unit)` (GiB/s vs GB/s guard), critical path
with setup and overlap, simplified prefill-to-decode TTFT, and a paired quality gate
whose failure withdraws the capacity claim. `fixtures/make_eval.py` is a seeded
synthetic eval stub for the lossy-retention policy.

## BREAK

A 16-token transfer dominated by setup, and a lossy retention policy that fails the
quality gate. Neither may keep its capacity or latency claim.

## Run

```bash
cd 03-kv-cache-engineering/labs/lab-d-compression-tiering-bounds
python -m pytest tests -q                     # your starter/
python experiments/run_bounds.py              # results/<stamp>-bounds/{raw.jsonl,result.yaml}
```

Instructor: `LAB_IMPL=solution python -m pytest tests -q`.

## Submit

`report.md` (sections 1-2 committed before any experiment), your `starter/`, and the
`results/` run directories you cite. Results are append-only.
