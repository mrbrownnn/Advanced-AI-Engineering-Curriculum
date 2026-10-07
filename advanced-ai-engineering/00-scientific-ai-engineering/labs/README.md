# Module 00 Runnable Labs

Module source of truth: [§06 Engineering Labs](../README.md#06-engineering-labs).
All commands run from the named lab directory. Results append beneath each lab's
`results/<UTC timestamp>/`; never overwrite a run.

| Lab | Hardware | Effort | Test command | Experiment command |
|---|---|---:|---|---|
| [A — Event-Level Harness](lab-a-event-harness/) | CPU | 3h | `python -m pytest tests -q` | `python experiments/run.py` |
| [B — Randomized Paired Benchmark](lab-b-randomized-pairs/) | CPU | 3h | `python -m pytest tests -q` | `python experiments/run.py` |
| [C — Uncertainty and Tail Audit](lab-c-uncertainty-tail/) | CPU | 3h | `python -m pytest tests -q` | `python experiments/run.py` |
| [D — LoadGen Source Trace](lab-d-loadgen-trace/) | TRACE + CPU | 3h | `python -m pytest tests -q` | `python experiments/run.py` |

Prediction gate: complete and commit each `report.md` §§1–2 before running.
Instructor reference: set `LAB_IMPL=solution`; add `--solution` to experiment.
