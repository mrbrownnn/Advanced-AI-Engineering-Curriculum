# Lab Report — Module 02, LAB D: Competing Bottleneck Diagnosis

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

- §1: before looking at any fault trace, write your hypothesis matrix edits and predict which faults timeline-only evidence cannot separate.
- §2: evidence available per fault (unprofiled baseline, timeline, counters, clocks); which faults were injected on Colab vs synthetic only.
- §4: per fault -- symptom, hypotheses, missing evidence, discriminating measurement, ranked cause, intervention, unprofiled remeasurement.
- §5: the two paired faults with similar utilization: which aligned evidence separated them?
- §7: a falsifier that fired and how it changed the ranking or the signature matrix.
- §8: interacting faults are not modeled by the synthetic traces; what would change?
