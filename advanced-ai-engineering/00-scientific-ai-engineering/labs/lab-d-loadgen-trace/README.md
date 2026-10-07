# LAB D — Reproducible LoadGen Source Trace

[Module specification: §06 LAB D](../../README.md#lab-d--reproducible-loadgen-source-trace)

**Hardware**: TRACE + CPU boundary experiment

**Effort**: 3 hours (2h source trace, 1h boundary experiment)

Pinned upstream revision: `mlcommons/inference@3fbc329939999c13d0a7b5e67fb2092287e06047`.

## PREDICT gate

Complete and commit report sections 1–2. Predict which policy changes score while
fixed SUT event log remains unchanged: scheduled versus actual-issue origin,
success-only versus all-issued population, and LoadGen floor index versus
nearest rank. State falsifier and expected direction.

## Steps

1. Trace pinned source locally or at raw GitHub URLs. Answer report questions
   with exact file, symbol, line(s), revision, and `STATIC VERIFIED` or
   `TODO_VERIFY`. Do not imply execution.
2. Fill TODOs in `starter/boundary.py`; do not change fixture SUT events.
3. Run `python -m pytest tests -q`.
4. Run `python experiments/run.py`; inspect appended `scores.jsonl` and YAML.
5. Explain every score change using only boundary, population, or estimator.
6. Record exact trace commands and output artifacts. Any branch not inspected or
   run stays `TODO_VERIFY`.

Instructor: `LAB_IMPL=solution python -m pytest tests -q` and
`python experiments/run.py --solution`.

## Submit

Completed report, boundary implementation, tests, raw/YAML result pair, pinned
trace with source paths/symbols, exact commands, static-versus-executed labels,
and unresolved `TODO_VERIFY` items.
