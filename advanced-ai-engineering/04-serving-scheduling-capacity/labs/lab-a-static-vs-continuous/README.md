# LAB A — Static vs. Continuous Batching

**Hardware**: CPU

**Specification**: [Module README §06, LAB A](../../README.md#lab-a-serving-scheduler-simulator-static-vs-continuous-batching)  
**Estimated effort**: 4h

## PREDICT gate

Fill and commit `report.md` §§1–2 before running an experiment. Declare timestamp/resource boundaries, quantile convention, practical threshold, SLO predicate, and falsifier. Do not edit those sections afterward.

## Build

Complete every TODO in `starter/sim.py`. TODOs deliberately raise `NotImplementedError`; tests must collect before implementation. Run `python experiments/run_sweep.py`. Compare throughput, bubble percentage, TTFT/TPOT quantiles, and SLO goodput across policies, batch sizes, and generation variance. Then diagnose `starter/bugged_metrics.py` without diffing it.

## Run

```bash
python -m pytest tests -q
python experiments/run_sweep.py
```

PowerShell instructor check: `$env:LAB_IMPL='solution'; python -m pytest tests -q`. Results append under `results/<UTC>-<run>/` as `raw.jsonl` and `result.yaml`; never overwrite a run.

## Submit

Submit completed starter code, `report.md`, raw JSONL, result YAML, checksums, and aggregation command. Open instructor solution and `BREAK_KEY.md` only after submission.
