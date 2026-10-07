# LAB A — Event-Level Benchmark Harness

[Module specification: §06 LAB A](../../README.md#lab-a--event-level-benchmark-harness)

**Hardware**: CPU

**Effort**: 3 hours (2h implementation, 1h analysis)

## PREDICT gate

Before code or experiments, complete and commit report sections 1–2. Predict sample-count and P95 direction under a 2 s stall. Name population, boundary, nearest-rank convention, timeout treatment, and falsifier.

## Steps

1. Fill all TODO functions in `starter/harness.py`. Preserve intended issue, actual issue, start, end, and status on every event. Use only injected time.
2. Run `python -m pytest tests -q` until all invariants pass.
3. Run `python experiments/run.py` to append raw `events.jsonl` and `result.yaml` under `results/<UTC timestamp>/`.
4. Explain open-loop versus response-coupled observations. Then inspect `starter/bugged_harness.py` as a black box: run with `LAB_IMPL=bugged`, find the violated measurement invariant, and document evidence in §6.
5. Improve or falsify the explanation with one changed control while keeping the measurement boundary fixed.

Instructor check: `LAB_IMPL=solution python -m pytest tests -q` and `python experiments/run.py --solution`.

## Submit

- Completed `report.md` with prediction history intact.
- Completed `starter/harness.py` and passing test output.
- One `results/<timestamp>/events.jsonl` and matching `result.yaml`.
- BREAK diagnosis naming the invariant and observation that exposed it.
