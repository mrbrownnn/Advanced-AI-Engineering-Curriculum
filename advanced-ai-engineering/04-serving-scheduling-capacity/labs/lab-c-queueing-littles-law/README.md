# LAB C — Queueing Dynamics & Little's Law

**Hardware**: CPU

**Specification**: [Module README §06, LAB C](../../README.md#lab-c-queueing-dynamics-littles-law--saturation-breakdown)  
**Estimated effort**: 3h

## PREDICT gate

Fill and commit `report.md` §§1–2 before running an experiment. Declare timestamp/resource boundaries, quantile convention, practical threshold, SLO predicate, and falsifier. Do not edit those sections afterward.

## Build

Complete every TODO in `starter/queueing.py`. TODOs deliberately raise `NotImplementedError`; tests must collect before implementation. Run `python experiments/run_sweep.py`. Check finite-window Little identities, P-K mean predictions for stable M/G/1 cases, confidence intervals across independent seeds, and failure of steady-state interpretation above capacity.

## Run

```bash
python -m pytest tests -q
python experiments/run_sweep.py
```

PowerShell instructor check: `$env:LAB_IMPL='solution'; python -m pytest tests -q`. Results append under `results/<UTC>-<run>/` as `raw.jsonl` and `result.yaml`; never overwrite a run.

## Submit

Submit completed starter code, `report.md`, raw JSONL, result YAML, checksums, and aggregation command. Open instructor solution and `BREAK_KEY.md` only after submission.
