# LAB E — Instructor Key

Verified statically from raw GitHub source on 2026-10-08:

- repository: `pytorch/pytorch`
- commit: `7ee5406f6686d190efc6571475f074ba1bc9a8c0`
- file: `torch/utils/benchmark/utils/timer.py`
- raw URL: `https://raw.githubusercontent.com/pytorch/pytorch/7ee5406f6686d190efc6571475f074ba1bc9a8c0/torch/utils/benchmark/utils/timer.py`
- execution status: **not executed**; every runtime consequence below is `TODO_VERIFY` against the pinned runtime.

Line references use that raw file and are expected to drift on other revisions.

## Answers

1. **Q1 (lines 17–20):** module-level `timer()` checks `torch.accelerator.is_available()`, calls `torch.accelerator.synchronize()` if true, then returns `timeit.default_timer()`. Static.
2. **Q2 (around lines 176–188, 237–243):** `Timer.__init__` defaults its `timer` argument to that module-level function, and passes it to the underlying timer class. A caller that explicitly passes `timeit.default_timer` replaces this default and therefore bypasses this synchronization. Static; runtime effect `TODO_VERIFY`.
3. **Q3 (lines 255–258):** `_timeit(number)` delegates to `self._timer.timeit(number)` and clamps the result with `max(..., 1e-9)`. The comment says even a timer call costs about 50 ns, no real op should take under 1 ns, and the clamp prevents division by zero. Static.
4. **Q4 (around lines 295–299):** `_estimate_block_size` evaluates `_timeit(0)` five times, forms a Torch tensor, and takes its median as overhead. Static.
5. **Q5 (around lines 300–310):** starts `number = 1`, times that block, and stops when (a) `relative_overhead <= 1e-4` **and** `time_taken >= min_run_time / 1000`, (b) `time_taken > min_run_time`, or (c) multiplying `number` by 10 would exceed `2147483647`. Otherwise `number *= 10`. Static.
6. **Q6 (around lines 297–298):** comment: `This also serves as a warmup.` For `blocked_autorange`, this estimation path is its warmup; unlike `Timer.timeit`, the method body shown has no separate warmup call. Static.
7. **Q7 (around lines 312–348, 277–293):** `blocked_autorange` chooses `number` via `_estimate_block_size`, then `_threaded_measurement_loop` collects complete block times until accumulated `total_time >= min_run_time` and `stop_hook(times)` permits stopping. Its `stop_hook` always returns `True`, so measured blocks do not stop before `min_run_time` (one final block can overshoot). Static.
8. **Q8 (around lines 349–353):** returns `Measurement(number_per_run=number, raw_times=times, task_spec=self._task_spec)`. Each `raw_time` is the duration of one block containing `number_per_run` statement executions. Per-call time is `raw_time / number_per_run`. Static.
9. **Q9 (module-level lines 17–20 plus Python `timeit` delegation):** synchronization occurs inside the timer callable immediately before each `timeit.default_timer` read. Thus an available accelerator is synchronized before both start and stop timestamps made by the underlying `timeit.Timer`; the sync is outside the statement's `number`-iteration inner loop, not after every statement. Static; exact call nesting in Python stdlib and accelerator behavior `TODO_VERIFY` if not executed.
10. **Q10:** pre-start synchronization drains accelerator work queued before the timed block; stop-side synchronization waits for work submitted by the block. This suppresses overlap between the timed block and surrounding accelerator work. It is operator/block scope, not request queueing or end-to-end serving latency. Work on another stream/device may or may not be included depending on `torch.accelerator.synchronize()` semantics at that runtime; `TODO_VERIFY` rather than universalize.
11. **Q11:** general: synchronize asynchronous work at a declared boundary, warm up, repeat, retain raw times. Revision-specific: accelerator abstraction, exact default timer, `1e-4`, `min_run_time / 1000`, decimal block growth, max integer, `min_run_time=0.2`, and call graph.

## Expected call path

```text
Timer.blocked_autorange
  -> Timer._estimate_block_size
       -> Timer._timeit
            -> underlying timeit.Timer.timeit
                 -> default timer callable
                      -> torch.accelerator.synchronize (if available)
                      -> timeit.default_timer
  -> Timer._threaded_measurement_loop
       -> time_hook
            -> Timer._timeit(number)
  -> Measurement(number_per_run, raw_times, task_spec)
```

## BREAK / falsification

Add asynchronous second-stream work outside the timed statement. The utility's pre-start synchronization can drain earlier work and suppress overlap, while events on stream 0 exclude unrelated stream-1 work and an application boundary includes all required completion. Exact observed ordering is `TODO_VERIFY` until the Colab LAB B run exists.
