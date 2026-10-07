# LAB C — Qualified Roofline Matrix

**Hardware**: COLAB (analytical core runs locally; measurement in `colab.ipynb`)

Spec: [Module 02 README §06 LAB C](../../README.md#lab-c--qualified-roofline-matrix). Lessons 2.4–2.5.

## PREDICT gate

Fill and commit `report.md` §1–2 before running anything in `experiments/` or `colab.ipynb`.

## What you build

| Part | Where | Runs on |
|---|---|---|
| `gemm`, `gemv`, `elementwise`, `reduction` workloads (W, one-pass Q, path, boundary) | `starter/roofline.py` | CPU |
| `ridge`, `attainable`, `bound` (refuses mismatched path/boundary), `qualify`, `classification_flips` | `starter/roofline.py` | CPU |
| Declared ceiling sets: exercise, `t4_nameplate` (**TODO_VERIFY**), `t4_measured` (fill from Colab) | `fixtures/ceilings.yaml` | — |
| Analytical matrix | `experiments/analytic_matrix.py` | CPU |
| Sustained-ceiling probes, timed shapes, fp64 error, flips nameplate→measured | `experiments/gpu_roofline.py` via `colab.ipynb` | Colab T4 |

## Assumptions you must keep visible

FMA = 2 FLOPs. Q = one-pass algorithmic bytes at DRAM. fp16 → tensor path and fp32 → CUDA-core path are **assumptions** about what cuBLAS picks; confirm with the kernel name in a profiler trace. T4 nameplate numbers in `ceilings.yaml` were not re-read from the datasheet in this revision: verify them before use. T4 (sm_75) has no bf16 tensor cores or FP8; those paths are out of scope here.

## Run

```bash
cd 02-inference-gpu-fundamentals/labs/lab-c-roofline-matrix
python -m pytest tests -q
python experiments/analytic_matrix.py                         # exercise ceilings
python experiments/analytic_matrix.py --ceilings t4_nameplate
```

After the Colab run, copy measured ceilings into `t4_measured` and rerun with `--ceilings t4_measured`.

## Counter availability (TODO_VERIFY)

Measured DRAM/L2 traffic and executed FLOPs need Nsight Compute counters, which may be blocked on Colab. **Fallback:** CUDA-event time + algorithmic Q/W; label the boundary "algorithmic one-pass" in every table row and do not call it measured intensity.

## Submit

`report.md` with the analytical-versus-measured table, run directories, starter code.
