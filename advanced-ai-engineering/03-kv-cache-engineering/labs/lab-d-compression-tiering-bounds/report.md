# Lab Report — Module 03, LAB D: Compression, Tiering, & Disaggregation Bounds

> Fill sections 1–2 and **commit before running any experiment**. The git log is
> your evidence that the prediction came first. Do not edit sections 1–2 after
> running; record disagreements in section 5.

## Lab-specific prompts (answer inside the sections below)

- Before running: predict the concurrency bound after FP8 with per-block FP32 scales, and the critical-path time of a 16-token and a 16,384-token transfer with 1 ms setup.
- Keep the four mechanisms separate in your tables: exact paging, representation quantization, lossy token retention, offload.
- Show one calculation done in GiB/s and the same in GB/s; state which the link specification uses.
- BREAK: a small transfer dominated by setup, and a lossy retention policy that fails the quality gate. Which capacity or latency claims did you withdraw?
- Prefill-to-decode handoff: which overlap fraction and setup make the transfer invisible to TTFT, and which measurement would confirm it? Do not name a transport you did not measure.

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
