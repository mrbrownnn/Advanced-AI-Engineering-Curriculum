# LAB A — Transactions, Layout, and Useful Bandwidth

**Hardware**: COLAB (CPU core runs locally; GPU measurement in `colab.ipynb`)

Spec: [Module 02 README §06 LAB A](../../README.md#lab-a--transactions-layout-and-useful-bandwidth). Lessons 2.1–2.2. The spec is the source of truth; this folder only adds scaffolding.

## PREDICT gate

Fill and commit `report.md` §1–2 before running anything in `experiments/` or `colab.ipynb`.

## What you build

| Part | Where | Runs on |
|---|---|---|
| Sector-transaction model: useful vs moved bytes for stride, offset, AoS/SoA (`sectors_touched`, `warp_traffic`, `pattern_traffic`) | `starter/transactions.py` | CPU |
| Useful bandwidth with an exact unit divisor (`useful_bandwidth`) | `starter/transactions.py` | CPU |
| Tiled matmul with shape/bounds assertions — the correctness oracle | `starter/transactions.py` | CPU |
| Measured useful bandwidth for the same patterns (`Tensor.copy_`, CUDA events) | `experiments/gpu_bandwidth.py` via `colab.ipynb` | Colab T4 |

The model is a declared derivation (D): whole aligned sectors, parameter `sector_bytes` (default 32), no cache reuse between warp instructions. Fixed transaction rules are architecture-specific; do not universalize them.

## Run

```bash
cd 02-inference-gpu-fundamentals/labs/lab-a-transactions-bandwidth
python -m pytest tests -q                 # your starter; fails with HINTs until done
python experiments/sweep_patterns.py      # model table -> results/<stamp>-model/
```

GPU part: open `colab.ipynb` in Colab (T4 runtime). It clones the repo, records `nvidia-smi` and versions, runs the CPU tests, probes Nsight Compute, measures, and writes `raw.jsonl` + `result.yaml` to Drive.

## BREAK

`starter/bugged_transactions.py` is a complete counter with one real bug. Find it by feeding it inputs whose answer you can derive by hand, not by diffing. `test_break_module_counts_boundary_crossing` stays red until you fix it in place. Record the finding in `report.md` §6.

## Counter availability (TODO_VERIFY)

L1/L2/DRAM sector counts need Nsight Compute device counters. On Colab these may be blocked (`ERR_NVGPUCTRPERM`); no Colab pilot has run yet. The notebook records the probe result in `result.yaml` (`environment.ncu`). **Fallback:** CUDA-event time + the analytical byte model from this lab. Then your conclusion is limited to useful bandwidth vs modeled traffic; you cannot claim measured hierarchy traffic.

## Submit

`report.md`, the `results/` (CPU) and Drive (Colab) run directories, and your starter code. Then, and only then, open `labs-solutions/`.

