# LAB D — Instructor / BREAK Key

Open only after submitting the lab report.

## Worked bounds

Lesson 3.8 assumptions give FP8 payload 256 KiB/token plus 512 bytes/token of
scales = 256.5 KiB/token. The 63 GiB, 4096-token payload-only concurrency bound
moves 31 to 62, not exactly 2x in bytes and not a throughput claim.

At 50 GiB/s, 2 GiB serializes in 40 ms and 8 GiB in 160 ms. Using 50 GB/s instead
changes the result by the GiB/GB factor. Bare bandwidth numbers are rejected.

## Break results

- Sixteen tokens of the example KV are 8 MiB: serialization at 50 GiB/s is
  0.15625 ms, so a 1 ms setup dominates and effective bandwidth is far below the
  link number. A serialization-only latency claim is withdrawn.
- Keeping only the recent half of the seeded toy evidence removes enough paired
  score to fail the 0.02 mean-drop gate. `capacity_claim` returns `None`: capacity
  arithmetic remains computable, but no deployment claim survives.
