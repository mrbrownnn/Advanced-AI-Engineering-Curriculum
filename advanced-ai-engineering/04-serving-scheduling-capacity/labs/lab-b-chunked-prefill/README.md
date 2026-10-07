# LAB B — Chunked Prefill & ITL Variance

**Hardware**: CPU

**Specification**: [Module README §06, LAB B](../../README.md#lab-b-chunked-prefill--itl-variance-under-mixed-workloads)  
**Estimated effort**: 3.5h

## PREDICT gate

Fill and commit `report.md` §§1–2 before running an experiment. Declare timestamp/resource boundaries, quantile convention, practical threshold, SLO predicate, and falsifier. Do not edit those sections afterward.

## Build

Complete every TODO in `starter/chunked.py`. TODOs deliberately raise `NotImplementedError`; tests must collect before implementation. Lab B reuses Lab A `sim.py`. Run `python experiments/run_sweep.py`. Plot or tabulate ITL tail, long-prompt TTFT, throughput, step count, and cost components across chunk sizes including 32 and unchunked.

## Run

```bash
python -m pytest tests -q
python experiments/run_sweep.py
```

PowerShell instructor check: `$env:LAB_IMPL='solution'; python -m pytest tests -q`. Results append under `results/<UTC>-<run>/` as `raw.jsonl` and `result.yaml`; never overwrite a run.

## Submit

Submit completed starter code, `report.md`, raw JSONL, result YAML, checksums, and aggregation command. Open instructor solution and `BREAK_KEY.md` only after submission.
