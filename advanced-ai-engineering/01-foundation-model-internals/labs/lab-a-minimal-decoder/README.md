# LAB A — Minimal Decoder, Numerical Parity, and Sampling Contract

Spec: [Module README §06, LAB A](../../README.md#lab-a--minimal-decoder-numerical-parity-and-sampling-contract). Alignment: Lessons 1.1–1.5.

**Hardware**: CPU

**Effort**: 5h. **Dependencies**: Python 3.10+, CPU-only `torch`, `pytest`, `hypothesis`, `pyyaml`. No model downloads or network.

## PREDICT gate

Before tests or experiments, fill and commit `report.md` §§1–2. Predict numerical tolerance, first failure under each named defect, and how temperature/truncation changes the toy distribution. Do not edit those sections after running.

## Steps

1. Read signatures, docstrings, axis names, and `fixtures/tiny.yaml`.
2. Fill every TODO in `starter/decoder.py` and `starter/sampling.py`. Do not use a high-level decoder block or `generate()`.
3. Run `python -m pytest tests -q`. Assertions give diagnostic hints, not implementations.
4. BREAK: instrument `starter/bugged_mask.py`; record first failed invariant without diffing against the instructor tree.
5. Run `python experiments/run.py`; inspect new `results/<timestamp>-decoder/{raw.jsonl,result.yaml}`.
6. Complete report §§3–8. Only after submission, compare reference implementation.

Instructor check: `$env:LAB_IMPL='solution'; python -m pytest tests -q`.

## What to submit

- Completed `starter/decoder.py`, `starter/sampling.py`, and `report.md`.
- Timestamped result directory, raw-data checksum, exact pytest/experiment commands.
- Named-axis trace, tolerance/dtype justification, full printed sampling contract, BREAK localization, limitations.
