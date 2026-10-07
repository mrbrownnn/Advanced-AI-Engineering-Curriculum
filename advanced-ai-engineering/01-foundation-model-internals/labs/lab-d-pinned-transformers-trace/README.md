# LAB D — Pinned Transformers Source Trace

Spec: [Module README §06, LAB D](../../README.md#lab-d--pinned-transformers-source-trace). Alignment: Lesson 1.8 and Lesson 1.2 sampling contract.

**Hardware**: TRACE

**Effort**: 3h (2h source trace + 1h execution). **Dependencies for supplied offline checks**: Python 3.10+, `pytest`, `pyyaml`, CPU-only `torch` for result metadata. No model download. `transformers` is deliberately not installed by scaffold.

## PREDICT gate

Before opening instructor answers or executing a pinned environment, fill and commit `report.md` §§1–2. Predict default backend, cast boundaries, processor list with omitted versus explicit top-k, and branch changed by forced backend.

## Steps

1. Inspect `fixtures/pinned-trace.yaml`; follow its raw URLs at commit `11c16613d93911300c38ec8c9c2460567be53281`.
2. Answer every trace prompt in `report.md` with file, symbol, source observation, equation/tensor contract, and static/executed status.
3. Run `python -m pytest tests -q` and `python experiments/run.py` offline to validate/emit supplied static evidence.
4. Separately execute pinned Transformers if your environment permits it: tiny config only, random weights, CPU only, no checkpoint download. Log `config._attn_implementation`, hooks, cache boundary, and `type(p).__name__` for every effective processor.
5. Force one supported alternative attention backend and compare processor lists with omitted `top_k` versus explicit `top_k=0`. Mark every unrun branch `TODO_VERIFY`.

## What to submit

`report.md`; repository/commit/verification date; exact paths/symbols/commands; dynamic configuration; selected backend; printed processor list; source-to-equation map; static-versus-executed labels; `TODO_VERIFY` markers; timestamped static trace artifacts.
