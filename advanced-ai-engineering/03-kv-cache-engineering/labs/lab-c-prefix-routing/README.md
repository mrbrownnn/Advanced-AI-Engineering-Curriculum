# LAB C — Prefix Sharing & Multi-Worker Routing

**Hardware**: CPU
**Effort**: 3h

Spec: [README §06 LAB C](../../README.md#06-engineering-labs) (Lessons 3.6, 3.7).

**Artifact reuse:** imports your Lab B `block_manager.py`; `tests/conftest.py` puts
`../lab-b-paged-block-manager/starter` on `sys.path`. Finish Lab B first.

## PREDICT gate

Commit `report.md` §1-2 before running `experiments/run_routing.py`.

## What to build (`starter/`)

- `prefix_cache.py`: chained full-block hashes with identity extra keys (model and
  adapter in every block; tenant salt in the first block only), and
  `PrefixCachingBlockManager` with longest-hit lookup, refcount-0 residency in the free
  queue, and LRU eviction when a cached block is reused.
- `router_sim.py`: discrete-event multi-worker simulation. Implement
  `prefix_affinity`, your `load_aware` policy, `summarize` (p50/p95/p99 TTFT, hit rate,
  SLO-goodput, load skew) and the Lesson 3.7 break-even helpers. `repeated` runs seeded
  repetitions and reports mean and sample std.

## BREAK

Hot-prefix workload (`hot_share=0.9`): affinity raises hit rate but overloads the
resident worker and lowers SLO-goodput. Hit-rate-only selection is falsified.

## Run

```bash
cd 03-kv-cache-engineering/labs/lab-c-prefix-routing
python -m pytest tests -q                     # your starter/ (+ your Lab B)
python experiments/run_routing.py             # results/<stamp>-routing/{raw.jsonl,result.yaml}
```

Instructor: `LAB_IMPL=solution python -m pytest tests -q`.

## Submit

`report.md` (sections 1-2 committed before any experiment), your `starter/`, and the
`results/` run directories you cite. Results are append-only.
