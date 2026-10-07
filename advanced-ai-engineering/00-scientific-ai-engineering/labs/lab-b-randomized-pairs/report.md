# Lab Report — Module 00, LAB B: Randomized Paired Benchmark

> Fill sections 1–2 and **commit before running any experiment**. The git log is
> your evidence that the prediction came first. Do not edit sections 1–2 after
> running; record disagreements in section 5.

## 1 PREDICTION (commit before running)

- **Claim under test** (atomic; population, boundary, estimand; label O/D/H):
- **Mechanism** for monotonic drift:
- **Predicted B−A under randomized order** (ms, direction):
- **Predicted B−A under fixed AB order** (ms, direction):
- **Predicted telemetry** (order/carryover pattern):
- **Falsifier**:
- **Practical effect threshold**:
- **Confidence** (0–100%):

## 2 DESIGN

- Independent variables (treatment, realized order, drift):
- Dependent variables (paired difference, interval, carryover/order effect):
- Controlled variables (seed, blocks, noise model):
- Experimental unit and `n` (what was assigned?):
- Block factor and independence argument:
- Randomization seed and realized-order record:
- Confounders and mitigation:
- Decision rule against practical threshold:

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
