# LAB B — BREAK Key

## Planted bug

`starter/bugged_timing.py:time_synced` synchronizes only after `work`. It starts the host clock before draining work already queued by another caller. The result therefore includes earlier unrelated device work, violating the timing contract "work alone".

## Discriminating probe

On `SimDevice`, queue a 10 ms kernel. Then call the bugged timer for a 1 ms kernel. It reports roughly 11 ms plus submission/sync costs. A correct timer calls `dev.synchronize()` before `t0 = timer()` and reports roughly 1 ms plus costs.

## Fix

Synchronize at both boundaries: drain before start, enqueue work, enforce completion, then stop.

## Expected observations

Unsynchronized host timing approaches submission cost as kernel duration grows; synchronized wall time includes host launch and sync-return costs; events bracket device work on their stream; application time includes setup and all required completion. Per-op synchronization serializes the simulated streams. Real Colab values remain `TODO_VERIFY`.
