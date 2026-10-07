# LAB D — Competing Bottleneck Diagnosis

**Hardware**: COLAB (synthetic-trace core runs locally; partial fault injection in `colab.ipynb`)

Spec: [Module 02 README §06 LAB D](../../README.md#lab-d--competing-bottleneck-diagnosis). Lessons 2.1, 2.2, 2.5, 2.6.

## PREDICT gate

Fill and commit `report.md` §1–2 before running anything in `experiments/` or `colab.ipynb`.

## What you build

| Part | Where | Runs on |
|---|---|---|
| Trace features, `direction`, `compare`, `score`, `diagnose` (scoring rule: ≥2 ruled down), `discriminating_features`, `attribute_regression`, `recovered_fraction` | `starter/diagnose.py` | CPU |
| Seeded synthetic traces, one planted fault each, with/without kernel counters | `fixtures/gen_traces.py` | CPU |
| Starting hypothesis matrix (H — edit it when a falsifier fires) | `fixtures/signatures.yaml` | — |
| Blind diagnosis over all faults, counters vs timeline-only | `experiments/blind_diagnosis.py` | CPU |
| Inject 4 faults on a T4, unprofiled baseline + `torch.profiler` timeline, same `diagnose.py` | `experiments/gpu_inject.py` via `colab.ipynb` | Colab T4 |

## Fault coverage

| Fault | Synthetic | Colab injection |
|---|---|---|
| host launch gap, synchronization, small grid, strided loads | yes | yes |
| divergence, register pressure | yes | TODO: needs a custom Triton/CuPy kernel |
| background GPU activity | yes | TODO: second process |
| constrained clock/power | yes | no — clocks are not controllable on Colab; do not try |

Synthetic magnitudes are invented to be plausible and to make two pairs look alike in utilization. They are not GPU measurements.

## Run

```bash
cd 02-inference-gpu-fundamentals/labs/lab-d-bottleneck-diagnosis
python -m pytest tests -q
python experiments/blind_diagnosis.py
python fixtures/gen_traces.py            # optional: dump traces as JSON to inspect
```

## Counter availability (TODO_VERIFY)

Nsight Compute counters (spills, sectors/request, branch efficiency) may be blocked on Colab. Then those features are `unknown`, and `diagnose()` must abstain where only counters discriminate (e.g. strided loads vs register pressure). **Fallback:** a one-variable intervention measured unprofiled, such as restoring contiguous layout or lowering register use, then `recovered_fraction`. `experiments/chrome_trace.py` field names are also TODO_VERIFY against a real Colab export.

## Submit

`report.md` with the full chain per fault: symptom → hypotheses → missing evidence → discrimination → ranked cause → intervention → remeasurement.
