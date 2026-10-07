# Lab Report — Module 00, LAB D: Reproducible LoadGen Source Trace

> Fill sections 1–2 and **commit before running any experiment**. The git log is
> your evidence that the prediction came first. Do not edit sections 1–2 after
> running; record disagreements in section 5.

## 1 PREDICTION (commit before running)

- **Claim under test** (fixed SUT, policy estimand; label O/D/H):
- **Predicted scheduled vs actual-origin score direction**:
- **Predicted success-only vs all-issued score direction**:
- **Predicted LoadGen-floor vs nearest-rank difference**:
- **Falsifier** (with SUT events unchanged):
- **Trace status prediction** (static/executed/TODO_VERIFY):
- **Confidence** (0–100%):

## 2 DESIGN

- Independent variables (origin, population, quantile convention):
- Dependent variables (population n, failure rate, quantile):
- Controlled variables (identical event log, q, timeout):
- Source revision and verification date:
- Required path: sanitation → scenario dispatch → issue → completion → results → artifacts:
- Evidence labels (`STATIC VERIFIED`, executed, `TODO_VERIFY`):
- Exact commands and environment to record:
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

### Pinned source-trace questions

For each answer, cite exact revision, file, symbol, and line(s); label it
`STATIC VERIFIED`, executed, or `TODO_VERIFY`.

1. Where does `StartTest` sanitize requested settings, and how does scenario and mode dispatch select the performance path?
2. How does `IssueQueries` reach `IssueQueryController::StartIssueQueries` and ultimately the SUT's `IssueQuery` callback?
3. For Server, SingleStream, MultiStream, and coalesced queries, where are `scheduled_time` and `issued_start_time` assigned? Which preserve an independent intended schedule?
4. Where does `QuerySamplesComplete` read completion time? Is time read once per callback or once per response, and what timestamp reaches `ResponseDelegateDetailed::SampleComplete`?
5. In `ResponseDelegateDetailed::SampleComplete`, which timestamp is latency origin, which is end, and where is the value recorded?
6. How does `GetLatenciesBlocking` handle expected and incomplete samples before result processing?
7. Where does `PerformanceSummary::ProcessLatencies` sort values and choose a percentile? Explain why `sample_latencies[sample_count * percentile]` is a 0-based `floor(n*p)` index at this revision.
8. Which code constructs/logs `PerformanceSummary`, and which summary, detail, accuracy, and trace artifacts are opened?
9. Which paths did you execute rather than inspect statically? Mark every unexecuted branch and unverified build/runtime claim `TODO_VERIFY`.

Raw boundary-experiment data path(s), checksums, and aggregation command. Table
all policy scores over identical SUT rows. State origin, population, quantile
convention, failure semantics, and trace completeness.

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
