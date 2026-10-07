# LAB B — Asynchronous Timing Trap

**Hardware**: COLAB (CPU simulator core runs locally; GPU measurement in `colab.ipynb`)

Spec: [Module 02 README §06 LAB B](../../README.md#lab-b--asynchronous-timing-trap). Lesson 2.3.

## PREDICT gate

Fill and commit `report.md` §1–2 before running anything in `experiments/` or `colab.ipynb`.

## What you build

| Part | Where | Runs on |
|---|---|---|
| `time_synced`, `time_events`, `time_application` (incl. per-op sync), `summarize` (cold vs warm) | `starter/timing.py` | CPU |
| Deterministic async device: host clock + per-stream queues + events (**provided**, read it) | `starter/simdevice.py` | CPU |
| Adapter so the same `timing.py` drives `torch.cuda` streams/events | `experiments/torch_device.py` | Colab T4 |
| Unsynced vs synced vs events vs application vs `torch.utils.benchmark`, one and two streams | `experiments/gpu_timing.py` via `colab.ipynb` | Colab T4 |

The simulator's semantics are a simplification (D): no stream contention, fixed launch/queue/sync costs. Its job is to make the boundary logic testable with an injectable clock, not to predict GPU numbers.

## Run

```bash
cd 02-inference-gpu-fundamentals/labs/lab-b-async-timing
python -m pytest tests -q
python experiments/sim_timing_sweep.py    # -> results/<stamp>-sim/
```

## BREAK

`starter/bugged_timing.py` has a synchronized timer with one real bug. Build the simulated scenario where it is wrong. `test_break_module_excludes_earlier_work` stays red until fixed in place. Record it in `report.md` §6.

## Notes

- LAB E traces `torch.utils.benchmark.Timer.blocked_autorange` at a pinned commit; the Colab run here gives you the utility's numbers to compare.
- Nsight Systems is not required. Timeline evidence comes from `torch.profiler` if you want it (LAB D does this).

## Submit

`report.md` (with a timing contract per method), run directories, starter code.
