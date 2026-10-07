# LAB E — Pinned Benchmark Source Trace

**Hardware**: TRACE (static source reading; the 1h comparison reuses LAB B's Colab run)

Spec: [Module 02 README §06 LAB E](../../README.md#lab-e--pinned-benchmark-source-trace) and Lesson 2.7.

## Target

- Repository: `pytorch/pytorch`
- Commit: `7ee5406f6686d190efc6571475f074ba1bc9a8c0`
- File: `torch/utils/benchmark/utils/timer.py`
- Raw URL: `https://raw.githubusercontent.com/pytorch/pytorch/7ee5406f6686d190efc6571475f074ba1bc9a8c0/torch/utils/benchmark/utils/timer.py`

Read that exact revision. A branch head (`main`) is a different source, and your answers would not be checkable.

## PREDICT gate

Before opening the file, write in `report.md` §1 where you expect synchronization to happen, how warmup works, and what `blocked_autorange` returns. Commit. Then trace.

## Steps

1. Answer the trace questions in `report.md`, citing line numbers at the pinned commit for every answer.
2. Mark each answer **static** (read) or **executed** (you ran it). Anything you did not execute gets `TODO_VERIFY`.
3. Comparison (1h): take LAB B's Colab `result.yaml` (`torch.utils.benchmark.blocked_autorange`, `events_stream0`, `synced` rows) for the same operator and explain each difference.
4. Break & falsify: add asynchronous work outside the timed statement (LAB B two-stream case) and explain why each boundary reports a different value.

No code and no tests: this lab is graded on cited evidence.

## Submit

`report.md`. The answer key is in `labs-solutions/02-inference-gpu-fundamentals/lab-e-benchmark-source-trace/`; open it only after submitting.
