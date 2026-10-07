# LAB C — Accounting Versus Real Execution

Spec: [Module README §06, LAB C](../../README.md#lab-c--accounting-versus-real-execution). Alignment: Lessons 1.6–1.7.

**Hardware**: CPU

**Effort**: 3h. **Dependencies**: Python 3.10+, CPU-only `torch` (profiler + `FlopCounterMode`), `pytest`, `pyyaml`. No downloads or network.

## PREDICT gate

Before running, fill and commit `report.md` §§1–2. Predict parameter deltas from bias, tying, adapter, and MoE; state FMA convention and exclusions; predict where profiler FLOPs, memory events, and latency will differ from analytical work.

## Steps

1. Fill TODOs in `starter/accounting.py`; manifest rows must map to instantiated tensors.
2. Run `python -m pytest tests -q`; reconcile counts, aliases, logical FLOPs, KV tensor bytes, MoE categories, and pinned MLA fields.
3. Run `python experiments/run.py`; inspect raw profiler shapes, CPU memory events, and repeated timings.
4. Sweep/toggle bias, tying, adapter, Hkv, and (within tiny sizes) MoE. Explain each count delta by name.
5. Record why `num_key_value_heads` is rejected for pinned DeepSeek-V2 MLA retained state.

Instructor check: `$env:LAB_IMPL='solution'; python -m pytest tests -q`.

## What to submit

Completed `starter/accounting.py`, `report.md`, timestamped result artifacts, manifest with analytical/actual/delta columns, FMA/exclusion statement, profiler evidence, MLA retained-state row, and limitations distinguishing logical work from measured execution.
