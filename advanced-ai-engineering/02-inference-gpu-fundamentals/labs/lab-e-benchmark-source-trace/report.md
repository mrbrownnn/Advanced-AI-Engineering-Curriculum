# Lab Report — Module 02, LAB E: Pinned Benchmark Source Trace

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

## TRACE QUESTIONS

Pinned source: `pytorch/pytorch` @ `7ee5406f6686d190efc6571475f074ba1bc9a8c0`, `torch/utils/benchmark/utils/timer.py`.
For every answer give line number(s) and mark it `static` or `executed`; unexecuted claims get `TODO_VERIFY`.

| # | Question | Answer | Lines | static / executed |
|---|---|---|---|---|
| Q1 | What does the module-level `timer` function do, in order? Under what condition does it synchronize? | | | |
| Q2 | Is `timer` the default for `Timer(..., timer=...)`? What happens if a caller passes `timeit.default_timer` directly? | | | |
| Q3 | What does `Timer._timeit(number)` delegate to, and why does it clamp the result? | | | |
| Q4 | `_estimate_block_size`: how is per-call timer overhead estimated? | | | |
| Q5 | `_estimate_block_size`: list every loop-exit condition and how `number` grows. | | | |
| Q6 | Which comment states that block-size estimation also serves as warmup? Is it the only warmup `blocked_autorange` performs? | | | |
| Q7 | `blocked_autorange`: what decides how many blocks are timed? Can it stop before `min_run_time` of measured time? | | | |
| Q8 | What does the returned `Measurement` contain? Is a raw time per call or per block? | | | |
| Q9 | Where is synchronization relative to `timeit.default_timer` reads, and is it inside or outside the `number`-iteration loop? | | | |
| Q10 | Which asynchronous work does this path exclude or suppress (queued work before the start, overlap with other streams/requests)? | | | |
| Q11 | What is general measurement practice here and what is revision-specific (thresholds, accelerator abstraction, call graph)? | | | |

## COMPARISON TABLE (one operator, from LAB B's Colab run)

| Method | Boundary | Sync placement | Per-call time | Why it differs |
|---|---|---|---|---|
| `blocked_autorange` | | | | |
| CUDA events (stream 0) | | | | |
| Manual synchronized wall time | | | | |
| Any of the above with second-stream work outside the statement | | | | |

## ARTIFACT CHECKLIST

repository · commit · verification date · file · symbols · entry point · execution path · exact run commands · `TODO_VERIFY` for every path not executed
