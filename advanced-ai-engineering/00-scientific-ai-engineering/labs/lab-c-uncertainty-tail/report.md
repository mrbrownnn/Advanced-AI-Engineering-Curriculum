# Lab Report — Module 00, LAB C: Uncertainty and Tail Audit

> Fill sections 1–2 and **commit before running any experiment**. The git log is
> your evidence that the prediction came first. Do not edit sections 1–2 after
> running; record disagreements in section 5.

## 1 PREDICTION (commit before running)

- **Claim under test** (atomic; population, estimand; label O/D/H):
- **Predicted naive interval-width change** with more within-process requests:
- **Predicted process/cluster interval-width change** with same process count:
- **Predicted effect of more process starts** at similar total requests:
- **Tail exposure prediction** for q=0.01, target=0.95:
- **Falsifier**:
- **Confidence** (0–100%):

## 2 DESIGN

- Independent variables (processes, requests/process, resampling unit, q):
- Dependent variables (interval widths, effective structure, exposure):
- Controlled variables (seed, total requests, variance components):
- Experimental unit and inferential `n` for each method:
- t and bootstrap assumptions:
- Why exposure is not P99 precision:
- Sensitivity analysis:
- Practical threshold:

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
