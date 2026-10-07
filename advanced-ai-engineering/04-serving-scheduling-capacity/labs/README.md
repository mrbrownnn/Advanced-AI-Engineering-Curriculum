# Module 04 Runnable Labs

CPU-only deterministic serving simulators. Complete each `report.md` prediction and design before running. Reference solutions are instructor material and should remain private until submission.

| Lab | Hardware | Effort | Run |
|---|---|---:|---|
| [A — Static vs continuous batching](lab-a-static-vs-continuous/) | CPU | 4h | `cd lab-a-static-vs-continuous && python -m pytest tests -q` |
| [B — Chunked prefill](lab-b-chunked-prefill/) | CPU | 3.5h | `cd lab-b-chunked-prefill && python -m pytest tests -q` |
| [C — Queueing and Little's Law](lab-c-queueing-littles-law/) | CPU | 3h | `cd lab-c-queueing-littles-law && python -m pytest tests -q` |
| [D — Preemption and admission](lab-d-preemption-admission/) | CPU | 3.5h | `cd lab-d-preemption-admission && python -m pytest tests -q` |

Instructor check: set `LAB_IMPL=solution` (PowerShell: `$env:LAB_IMPL='solution'`) before the same command.
