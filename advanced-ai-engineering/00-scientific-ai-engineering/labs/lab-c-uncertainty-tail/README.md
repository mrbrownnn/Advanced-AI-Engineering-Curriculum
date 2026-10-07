# LAB C — Uncertainty and Tail Audit

[Module specification: §06 LAB C](../../README.md#lab-c--uncertainty-and-tail-audit)

**Hardware**: CPU

**Effort**: 3 hours (2h analysis, 1h report)

## PREDICT gate

Complete and commit report sections 1–2. Predict interval-width changes when
adding requests within five processes versus increasing process starts at
similar total requests. State inferential unit, assumptions, and why tail
exposure is not percentile precision.

## Steps

1. Fill TODOs in `starter/audit.py`: clustered generator, deliberately naive
   request interval, process interval, cluster bootstrap, and tail exposure.
2. Run `python -m pytest tests -q`.
3. Run `python experiments/run.py`; inspect appended request JSONL and YAML.
4. Compare methods at similar total request count. Name experimental `n` for
   every interval and audit sensitivity to within-process replication.
5. Verify 1% event exposure at 95% reaches minimum `n=299`; explicitly reject
   any P99 confidence/precision guarantee from this calculation.
6. Falsify the favored model by changing the process/request variance ratio.

Instructor: `LAB_IMPL=solution python -m pytest tests -q` and
`python experiments/run.py --solution`.

## Submit

Completed report and implementation, passing tests, raw JSONL/result YAML,
interval table with assumptions and unit, sensitivity analysis, and tail audit.
