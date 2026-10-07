# Lab Report — Module 03, LAB C: Prefix Sharing & Multi-Worker Routing

> Fill sections 1–2 and **commit before running any experiment**. The git log is
> your evidence that the prediction came first. Do not edit sections 1–2 after
> running; record disagreements in section 5.

## Lab-specific prompts (answer inside the sections below)

- Before running: predict hit rate, p99 TTFT and SLO-goodput for round-robin, prefix-affinity and your load-aware policy under hot_share 0.1 and 0.9.
- Identity: list the extra keys your hash uses and show one token-identical pair that must not share KV, with evidence the cache refused it.
- Residency vs ownership: show a trace where blocks reach refcount 0, stay cached, and are hit again; then a trace where LRU evicts them.
- Results with p50/p95/p99 TTFT, mean and sample std over at least 5 seeds; state the quantile convention and the experimental unit.
- BREAK: the hot-prefix workload where affinity raises hit rate but lowers SLO-goodput. Does hit-rate-only selection survive? Check your simulator against the Lesson 3.7 break-even arithmetic.

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
