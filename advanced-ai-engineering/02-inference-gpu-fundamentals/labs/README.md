# Module 02 Labs — Inference GPU Fundamentals

No lab uses the local GPU. CPU cores run locally; GPU measurements run only in Google Colab. Colab notebooks target free-tier T4 (sm_75: no FP8, no bf16 tensor cores, no FlashAttention-2), save append-only results to Drive, and are shipped **not executed**.

| Lab | Hardware | Effort | Local / trace command | GPU part |
|---|---|---:|---|---|
| [A — Transactions, Layout, Useful Bandwidth](lab-a-transactions-bandwidth/) | COLAB + CPU core | 4h | `cd lab-a-transactions-bandwidth && python -m pytest tests -q` | `colab.ipynb` |
| [B — Asynchronous Timing Trap](lab-b-async-timing/) | COLAB + CPU simulator | 3h | `cd lab-b-async-timing && python -m pytest tests -q` | `colab.ipynb` |
| [C — Qualified Roofline Matrix](lab-c-roofline-matrix/) | COLAB + CPU analytical core | 4h | `cd lab-c-roofline-matrix && python -m pytest tests -q` | `colab.ipynb` |
| [D — Competing Bottleneck Diagnosis](lab-d-bottleneck-diagnosis/) | COLAB + CPU synthetic traces | 4h | `cd lab-d-bottleneck-diagnosis && python -m pytest tests -q` | `colab.ipynb` |
| [E — Pinned Benchmark Source Trace](lab-e-benchmark-source-trace/) | TRACE | 3h (2h trace + 1h comparison) | read pinned source; answer `report.md` | reuses Lab B results |

## Workflow

1. Open the lab README.
2. Fill `report.md` §1–2 and commit **before** any experiment.
3. Implement TODOs in `starter/`; run `python -m pytest tests -q`.
4. Run CPU experiment(s), then the notebook for COLAB labs.
5. Finish the report, including falsifier, BREAK, and limitations.
6. Only after submission, compare with `labs-solutions/`.

Instructor check from a lab directory:

```bash
LAB_IMPL=solution python -m pytest tests -q
```

On PowerShell use `$env:LAB_IMPL='solution'` before the pytest command and `Remove-Item Env:LAB_IMPL` afterward.

## Profiler constraint

Nsight Compute device counters may be unavailable on Colab due to permissions. Every affected notebook probes availability. Until a pilot run exists, counter availability is `TODO_VERIFY`; fallbacks use CUDA-event timing plus explicitly named analytical bytes/FLOPs, and conclusions remain at that boundary.
