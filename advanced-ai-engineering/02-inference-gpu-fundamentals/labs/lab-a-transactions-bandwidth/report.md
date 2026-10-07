# Lab Report — Module 02, LAB A: Transactions, Layout, and Useful Bandwidth

> Fill sections 1–2 and **commit before running any experiment**. The git log is
> your evidence that the prediction came first. Do not edit sections 1–2 after
> running; record disagreements in section 5.

## 1 PREDICTION (commit before running)

- **Claim under test** (atomic; population, boundary, estimand; label O/D/H):
- **Predicted result** (quantitative, with units and direction):
- **Why** (mechanism you expect):
- **Falsifier** (the observation that would prove the prediction wrong):
- **Confidence** (0–100%):

## 2 DESIGN

- Independent variables:
- Dependent variables:
- Controlled variables:
- Experimental unit and `n` (what was independently assigned?):
- Confounders and mitigation:
- Practical threshold (smallest effect that changes a decision):

## 3 ENVIRONMENT

```yaml
hardware: CPU | COLAB(<gpu>) | TRACE
os: _
python: _
packages: _          # pinned versions
commit: _            # git rev-parse HEAD
dirty_tree: _        # git status --porcelain empty?
seeds: _
```

## 4 RESULTS

Raw data path(s), checksums, and the aggregation command. Tables with
uncertainty at the correct unit. State quantile convention and failure
semantics.

## 5 PREDICTION vs OBSERVATION

| Prediction | Observed | Gap | Explanation of gap |
|---|---|---|---|

Was the falsifier triggered? What did you learn that you did not expect?

## 6 BREAK

What you broke, what the system did, which invariant/test caught it (or why
nothing did).

## 7 EXPLAIN / IMPROVE / FALSIFY

Competing explanations, the discriminating measurement, the change you made,
and the same-boundary remeasurement.

## 8 LIMITATIONS & TRANSFER

What this result does **not** support. Under which workload would the
conclusion change?

## Lab-specific prompts (answer inside the sections above)

- §1: predict moved/useful for stride 1, 2, 8, 32 and the AoS pattern *before* running `sweep_patterns.py`; predict the measured bandwidth ordering on T4.
- §2: state the sector size you assume and the boundary your "moved bytes" represent; name the one variable each pattern changes.
- §4: one byte equation per pattern, units stated (decimal GB/s or binary GiB/s, never mixed).
- §5: does measured useful bandwidth follow the model's efficiency ordering? Where not, which competing hypothesis (cache reuse, grid size, vectorization, timing error) fits?
- §6: the bug in `starter/bugged_transactions.py` -- the input that exposed it and the instrumentation you used.
- §8: if Nsight Compute counters were unavailable on Colab, which conclusions stay at the useful-byte boundary only?
