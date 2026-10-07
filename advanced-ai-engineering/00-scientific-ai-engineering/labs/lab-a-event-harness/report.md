# Lab Report — Module 00, LAB A: Event-Level Benchmark Harness

> Fill sections 1–2 and **commit before running any experiment**. The git log is
> your evidence that the prediction came first. Do not edit sections 1–2 after
> running; record disagreements in section 5.

## 1 PREDICTION (commit before running)

- **Claim under test** (atomic; population, boundary, estimand; label O/D/H):
- **Predicted issued-count difference during the 2 s stall** (quantitative):
- **Predicted P95 direction** (origin, population, nearest-rank, timeout policy):
- **Why** (which feedback path suppresses observations?):
- **Falsifier** (event-level observation):
- **Confidence** (0–100%):

## 2 DESIGN

- Independent variables (load mode, stall, cold/warm state, boundary):
- Dependent variables (issue lag, latency, throughput, failures, quantiles):
- Controlled variables (seed, clock, operation, timeout, schedule):
- Experimental unit and `n`:
- Clock/synchronization boundary:
- Population, nearest-rank definition, and timeout/censoring policy:
- Confounders and mitigation:
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
