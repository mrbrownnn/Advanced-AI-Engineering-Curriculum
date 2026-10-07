# LAB D Trace and Boundary Answer Key

Revision: `mlcommons/inference@3fbc329939999c13d0a7b5e67fb2092287e06047`.
Status: `STATIC VERIFIED` against raw upstream files; LoadGen not built/executed.
Verification date: 2026-10-08.

## Pinned execution path

1. Public entry: `loadgen/loadgen.cc:1170`, `StartTest`. It constructs
   `TestSettingsInternal` at line 1273. Scenario dispatch is
   `RunFunctions::Get` at lines 1147–1157; mode dispatch is the switch beginning
   at line 1280.
2. Performance path: `loadgen/loadgen.cc:843`, `RunPerformanceMode`, calls
   templated `IssueQueries` (`loadgen.cc:444`), which calls
   `IssueQueryController::StartIssueQueries` at line 498.
3. Controller: `loadgen/issue_query_controller.cc:292`,
   `StartIssueQueries`; `IssueQueriesInternal` begins line 343. It calls scenario
   `QueryScheduler::Wait` at line 386 then `sut->IssueQuery` at line 416.
4. Server schedule: `QueryScheduler<TestScenario::Server>` at
   `issue_query_controller.cc:163`; line 173 assigns scheduled time from the
   independent schedule, while actual issue time is assigned after wait at line
   180. Single/MultiStream assign both to current time at lines 130–133 and
   152–155. Coalesced server queries preserve original schedule at lines 92–104.
5. Completion entry: `loadgen/loadgen.cc:1312`, `QuerySamplesComplete`. It reads
   `PerfClock::now()` once at line 1314, notifies query metadata, then passes same
   timestamp to `response_delegate->SampleComplete` at lines 1330–1335.
6. Latency: `ResponseDelegateDetailed::SampleComplete` starts at
   `loadgen.cc:74`; `DurationGeneratorNs sched{query->scheduled_time}` at line
   103; latency is `sched.delta(complete_begin_time)` at line 130. Thus pinned
   sample latency origin is scheduled time, completion boundary is callback
   entry timestamp.
7. Collection: `loadgen/logging.cc:360`, `AsyncLog::RecordSampleCompletion`,
   stores latency at line 478. `AsyncLog::GetLatenciesBlocking` begins line 556;
   it waits for expected count and emits incomplete-sample error at lines
   586–600. `IssueQueries` retrieves latencies at `loadgen.cc:535`.
8. Processing: `loadgen/results.cc:21`,
   `PerformanceSummary::ProcessLatencies`; sort at line 34; target and reported
   percentile selection at lines 36–43 use
   `sample_latencies[sample_count * percentile]`. C++ conversion truncates
   nonnegative product, equivalent to `floor(n*p)`, as 0-based index. Percentile
   must be `<1` (assertions lines 40–41).
9. Outputs: `RunPerformanceMode` constructs `PerformanceSummary` at
   `loadgen.cc:926` and logs summary/detail. `LogOutputs` opens
   `<outdir>/<prefix>summary|detail|accuracy|trace<suffix>` at
   `loadgen.cc:696–709`; defaults are documented in `loadgen/test_settings.h`
   `LogOutputSettings` around lines 302–320.

## Boundary experiment

Fixed synthetic SUT rows must not change. Scheduled-origin scoring includes
issue lag; actual-issue origin removes it. All-issued includes timeout-censored
values; success-only changes denominator. Pinned LoadGen floor index and nearest
rank can select adjacent observations. `q=.75` and fixture were selected to
make all three policy effects visible.

## Unresolved

- `TODO_VERIFY`: build and execute pinned C++ LoadGen on Windows/Linux.
- `TODO_VERIFY`: runtime output keys and files for an actual scenario run.
- `TODO_VERIFY`: accuracy-only, find-peak, token-latency, and early-stopping
  branches beyond symbols/path above.
- `TODO_VERIFY`: generated build inputs, compiler, dependency versions, and
  worktree state for any future execution.
