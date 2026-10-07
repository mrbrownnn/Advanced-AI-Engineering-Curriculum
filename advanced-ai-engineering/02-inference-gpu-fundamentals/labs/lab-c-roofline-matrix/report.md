# Lab Report — Module 02, LAB C: Qualified Roofline Matrix

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

- §1: for each shape in `analytic_matrix.py`, predict the screen regime under `t4_nameplate` and which one(s) flip under measured ceilings.
- §2: FMA convention, precision/instruction path per dtype, byte boundary, nameplate vs measured ceilings, and what is excluded (launch, sync, allocation).
- §4: the analytical-versus-measured table: W, Q, I, I*, both time bounds, measured time, fraction of bound, numerical error, flags.
- §5: which shape flipped, and was the predicted ridge crossing where measured time started to grow with N?
- §6: the tiny high-intensity problem and the large low-intensity problem -- what did the bound say, what did the GPU do?
- §8: no universal "prefill/decode" label: which shapes and backend does each classification hold for?
