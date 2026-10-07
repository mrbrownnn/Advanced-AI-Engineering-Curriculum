# LAB D — Preemption & Admission Control

**Hardware**: CPU

**Specification**: [Module README §06, LAB D](../../README.md#lab-d-preemption-under-memory-pressure--admission-control)  
**Estimated effort**: 3.5h

## PREDICT gate

Fill and commit `report.md` §§1–2 before running an experiment. Declare timestamp/resource boundaries, quantile convention, practical threshold, SLO predicate, and falsifier. Do not edit those sections afterward.

## Build

Complete every TODO in `starter/kvsched.py`. TODOs deliberately raise `NotImplementedError`; tests must collect before implementation. Lab D reuses Lab A `sim.py`; Module 03 Lab B block manager can replace the simple counter. Run `python experiments/run_sweep.py`. Compare victim, preemption, and admission policies. Diagnose thrashing only from aligned allocation failures, churn, and lost useful progress.

## Run

```bash
python -m pytest tests -q
python experiments/run_sweep.py
```

PowerShell instructor check: `$env:LAB_IMPL='solution'; python -m pytest tests -q`. Results append under `results/<UTC>-<run>/` as `raw.jsonl` and `result.yaml`; never overwrite a run.

## Submit

Submit completed starter code, `report.md`, raw JSONL, result YAML, checksums, and aggregation command. Open instructor solution and `BREAK_KEY.md` only after submission.
