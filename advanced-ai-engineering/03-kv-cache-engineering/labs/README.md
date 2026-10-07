# Module 03 Labs

Scaffolding for README §06. Conventions: [LABS.md](../../LABS.md). Shared plumbing
(implementation selection, append-only result writer) lives in `_shared/labkit.py`.

| Lab | Hardware | Effort | Run |
|---|---|---|---|
| [A — KV analytical model + allocation simulator](lab-a-kv-allocation-simulator/) | CPU | 3h | `python -m pytest tests -q`; `python experiments/run_waste_sweep.py` |
| [B — Minimal paged KV block manager](lab-b-paged-block-manager/) (BREAK: `starter/bugged_block_manager.py`) | CPU | 4h | `python -m pytest tests -q`; `python experiments/run_timeline.py` |
| [C — Prefix sharing & multi-worker routing](lab-c-prefix-routing/) (reuses Lab B) | CPU | 3h | `python -m pytest tests -q`; `python experiments/run_routing.py` |
| [D — Compression, tiering & disaggregation bounds](lab-d-compression-tiering-bounds/) | CPU | 3h | `python -m pytest tests -q`; `python experiments/run_bounds.py` |
| [E — Pinned vLLM V1 lifecycle trace](lab-e-vllm-lifecycle-trace/) | TRACE | 2h + 1h | answer `report.md` |

Run each command from the lab directory. Instructor: prefix with `LAB_IMPL=solution`
(PowerShell: `$env:LAB_IMPL='solution'`). Lab B's block manager is reused by Lab C and
by Module 04 labs, so keep its public API stable.
