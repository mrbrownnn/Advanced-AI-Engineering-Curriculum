# LAB B — Randomized Paired Benchmark

[Module specification: §06 LAB B](../../README.md#lab-b--randomized-paired-benchmark)

**Hardware**: CPU

**Effort**: 3 hours (2h implementation, 1h analysis)

## PREDICT gate

Complete and commit report sections 1–2 first. Predict B−A under randomized
within-block order and fixed A-then-B order. State drift mechanism, practical
threshold, assignment unit, expected telemetry, and falsifier.

## Steps

1. Fill TODOs in `starter/benchmark.py`; preserve seed, realized order, block,
   run position, treatment, and raw measurement.
2. Run `python -m pytest tests -q`.
3. Run `python experiments/run.py`. It appends `runs.jsonl` and `result.yaml`
   under `results/<UTC timestamp>/`.
4. Compare paired block-level estimates and 95% t intervals for randomized and
   fixed-order designs. Explain why request/run count is not inferential `n`.
5. BREAK: select `LAB_IMPL=bugged`, diagnose the failed assignment invariant
   without source diffing, and record the discriminating observation in §6.
6. Change drift direction or magnitude and state which explanation this weakens.

Instructor: `LAB_IMPL=solution python -m pytest tests -q` and
`python experiments/run.py --solution`.

Optional extension: replace the simulated run with a fresh Python subprocess;
keep the same raw schema and block-level analysis.

## Submit

Completed report, implementation, passing output, raw JSONL/result YAML pair,
realized randomization record, paired differences, interval, and BREAK diagnosis.
