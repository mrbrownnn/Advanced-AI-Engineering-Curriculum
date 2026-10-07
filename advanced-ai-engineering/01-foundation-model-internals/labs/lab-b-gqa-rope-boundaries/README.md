# LAB B — MHA/MQA/GQA and RoPE Boundary Study

Spec: [Module README §06, LAB B](../../README.md#lab-b--mhamqagqa-and-rope-boundary-study). Alignment: Lessons 1.3–1.4.

**Hardware**: CPU

**Effort**: 3h. **Dependencies**: Python 3.10+, CPU-only `torch`, `pytest`, `pyyaml`. No downloads or network.

## PREDICT gate

Before running, fill and commit `report.md` §§1–2. Quantify expected K/V parameter and byte ratios for Hkv = Hq, intermediate Hkv, and Hkv = 1. Predict whether CPU latency follows those ratios and a position/dtype boundary for RoPE error.

## Steps

1. Fill TODOs in `starter/attention.py`; preserve axis and contiguous head-group contracts.
2. Run `python -m pytest tests -q` across MHA/GQA/MQA endpoints and RoPE boundaries.
3. BREAK: diagnose `starter/bugged_grouping.py` using uniquely-valued KV heads; do not diff instructor code.
4. Run `python experiments/run.py`; compare declared geometry against raw CPU timings. Do not claim fewer KV heads always lower latency.
5. Complete report with backend identity (`PyTorch CPU` plus effective operator observations), boundaries, uncertainty, and limits.

Instructor check: `$env:LAB_IMPL='solution'; python -m pytest tests -q`.

## What to submit

Completed `starter/attention.py`, `report.md`, timestamped raw/result artifacts, head-to-group map, MHA endpoint parity, RoPE shift-error table, and one workload where logical savings and latency are not proportional (or a precise statement that the tested sweep did not find one).
