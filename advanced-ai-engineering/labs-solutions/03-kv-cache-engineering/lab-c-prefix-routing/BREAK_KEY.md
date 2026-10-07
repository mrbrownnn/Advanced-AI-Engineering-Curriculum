# LAB C — Instructor / BREAK Key

Open only after submitting the lab report.

## Break result

With 90% hot-prefix traffic, 50 requests/s, 4 workers and the shipped cost model,
prefix affinity raises hit rate over round-robin but pins roughly 90% of requests
to one worker. Queue growth pushes tail TTFT beyond the 300 ms SLO and lowers
SLO-goodput. Exact numbers vary by seed; the required comparison is repeated-seed
direction plus mean and sample standard deviation.

Load-aware routing estimates `queue_wait + prefill(given hit)` per worker. It can
keep much of the cache gain without accepting an unbounded queue. Hit rate alone is
therefore rejected as the routing objective.

## Mutation expectations

- Evict a block with refcount > 0: active-hit / isolation or invariant tests fail.
- Omit model, adapter, or salt from `block_hashes`: identity tests fail.
- Index a partial block: full-block-count or invariant tests fail.
- Immediately erase a cached block at refcount 0: residency/reuse test fails.
