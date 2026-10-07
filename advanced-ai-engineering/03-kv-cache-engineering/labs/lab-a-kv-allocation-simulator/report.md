# Lab Report — Module 03, LAB A: KV Analytical Model + Allocation Simulator

> Fill sections 1–2 and **commit before running any experiment**. The git log is
> your evidence that the prediction came first. Do not edit sections 1–2 after
> running; record disagreements in section 5.

## Lab-specific prompts (answer inside the sections below)

- Before running: predict, for the heterogeneous trace at capacity 6000 slots, which waste category dominates allocated capacity under `max_len` vs `exact`, and whether `exact` produces contiguous-allocation failures.
- Give your own bytes/token for one MHA, one GQA, one MQA and one MLA configuration, with every assumption labeled (O/D/H).
- Table: share of allocated capacity that is logical / internal fragmentation / reservation waste, per policy and block size, mean and std over seeds. Separate blocked-by-capacity from blocked-by-contiguity.
- BREAK: run equal fixed-length sequences with sufficient space. Did any fragmentation remain? If yes, which simulator invariant or definition was wrong?
- Which layer of the abstraction hierarchy (Lesson 3.2) does each measured category live in? Which category would paging remove, and which would it not?

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
