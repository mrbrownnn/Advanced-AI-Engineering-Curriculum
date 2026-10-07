# Module 01 runnable labs

All runs are offline, deterministic, and CPU-only. Python 3.10+ required. Module dependency added per [LABS.md](../../LABS.md): CPU-only `torch` (installed from `https://download.pytorch.org/whl/cpu`). Existing dependencies: `pytest`, `hypothesis`, `pyyaml`.

| Lab | Hardware | Effort | Run |
|---|---|---:|---|
| [A — Minimal Decoder](lab-a-minimal-decoder/) | CPU | 5h | `cd lab-a-minimal-decoder && python -m pytest tests -q` |
| [B — GQA/RoPE Boundaries](lab-b-gqa-rope-boundaries/) | CPU | 3h | `cd lab-b-gqa-rope-boundaries && python -m pytest tests -q` |
| [C — Accounting/Execution](lab-c-accounting-execution/) | CPU | 3h | `cd lab-c-accounting-execution && python -m pytest tests -q` |
| [D — Pinned Source Trace](lab-d-pinned-transformers-trace/) | TRACE | 3h | `cd lab-d-pinned-transformers-trace && python -m pytest tests -q` |

Default tests target `starter/`. Instructor check from each lab: PowerShell `$env:LAB_IMPL='solution'; python -m pytest tests -q`; POSIX `LAB_IMPL=solution python -m pytest tests -q`. Lab D is trace-only; its instructor answer is `../../labs-solutions/01-foundation-model-internals/lab-d-pinned-transformers-trace/BREAK_KEY.md`.

Experiments use `python experiments/run.py` after selecting implementation with `LAB_IMPL=solution` when needed. Every run creates a new timestamped directory under `results/` containing raw `raw.jsonl` and `result.yaml`; previous runs are never overwritten.
