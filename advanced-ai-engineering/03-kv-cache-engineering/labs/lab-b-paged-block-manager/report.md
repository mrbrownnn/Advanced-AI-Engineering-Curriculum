# Lab Report — Module 03, LAB B: Minimal Paged KV Block Manager

> Fill sections 1–2 and **commit before running any experiment**. The git log is
> your evidence that the prediction came first. Do not edit sections 1–2 after
> running; record disagreements in section 5.

## Lab-specific prompts (answer inside the sections below)

- Before running: write down `my_cost` and predict, under it, whether delaying C lowers A's and B's mean gap in T1 and in T2, and by roughly how much (the K1/K2 rows are already published in Lesson 3.5).
- Invariant test log: which tests (refcount, free/allocated overlap, stale mapping, leaks, isolation property) caught which injected fault?
- BREAK: double-free, append through a shared mutable tail, stale block-table entry. For each: the exception raised and the invariant that caught it. Then the planted bug in `starter/bugged_block_manager.py`: the instrumentation you used, the symptom, the root cause in one sentence.
- Timeline: reproduce the table (attach your trace from `results/`), add your own cost model `my_cost`, and state which conclusions about running requests' mean gaps change under it.
- Block-size trade-off: measured internal waste vs block size for a length distribution you choose.

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
