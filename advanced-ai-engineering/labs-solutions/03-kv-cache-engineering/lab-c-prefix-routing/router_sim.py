"""Multi-worker cache-aware routing simulation (Lesson 3.7). Reference implementation.

Each worker is a single FIFO server with its own PrefixCachingBlockManager. A
request's service time is its prefill: miss tokens cost more than hit tokens.
TTFT = service end - arrival. Requests on one worker run one at a time, so the
cache state seen at service start is exactly the state after the previous
request on that worker; routing sees the same state (ponytail: a real router's
view is stale/approximate; add lag when modelling a remote index).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from prefix_cache import Identity, PrefixCachingBlockManager

IDENT = Identity(model="m1")


@dataclass
class Costs:
    base_ms: float = 5.0
    miss_ms_per_token: float = 0.5
    hit_ms_per_token: float = 0.05


def ttft_estimate(wait_ms: float, prefill_ms: float) -> float:
    return wait_ms + prefill_ms


def break_even_wait(other_wait_ms: float, miss_ms: float, hit_ms: float) -> float:
    """Largest hit-holder wait at which routing to the hit still ties the miss worker."""
    return other_wait_ms + (miss_ms - hit_ms)


def make_workload(n: int, seed: int, *, hot_share: float, num_prefixes: int = 8,
                  prefix_tokens: int = 256, suffix_tokens: int = 32, rate_per_s: float = 50.0):
    """Poisson arrivals; with prob hot_share use prefix 0, else a uniform other prefix."""
    rng = random.Random(seed)
    t, out = 0.0, []
    for i in range(n):
        t += rng.expovariate(rate_per_s) * 1000.0
        p = 0 if rng.random() < hot_share else rng.randrange(1, num_prefixes)
        prefix = [p * 100_000 + j for j in range(prefix_tokens)]
        suffix = [10_000_000 + i * 1000 + j for j in range(suffix_tokens)]
        out.append({"rid": i, "arrival_ms": t, "prefix_id": p, "tokens": prefix + suffix})
    return out


class Worker:
    def __init__(self, wid: int, num_blocks: int, block_size: int):
        self.wid = wid
        self.cache = PrefixCachingBlockManager(num_blocks, block_size)
        self.free_at = 0.0
        self.served = 0

    def wait(self, now: float) -> float:
        return max(0.0, self.free_at - now)

    def hit_tokens(self, tokens) -> int:
        return len(self.cache.longest_hit(tokens, IDENT)) * self.cache.block_size


def prefill_ms(tokens: int, hit: int, c: Costs) -> float:
    return c.base_ms + c.miss_ms_per_token * (tokens - hit) + c.hit_ms_per_token * hit


# ---- policies: (request, workers, now, costs, state) -> worker index -------

def round_robin(req, workers, now, c, state):
    state["rr"] = (state.get("rr", -1) + 1) % len(workers)
    return state["rr"]


def prefix_affinity(req, workers, now, c, state):
    """Send to the worker with the longest cached prefix; ties -> lowest index, but
    a never-seen prefix is pinned to its home worker (prefix_id mod W)."""
    hits = [w.hit_tokens(req["tokens"]) for w in workers]
    best = max(hits)
    if best == 0:
        return req["prefix_id"] % len(workers)
    return hits.index(best)


def load_aware(req, workers, now, c, state):
    """Lesson 3.7 rule: minimise estimated TTFT = queue wait + prefill given that worker's hits."""
    n = len(req["tokens"])
    est = [ttft_estimate(w.wait(now), prefill_ms(n, w.hit_tokens(req["tokens"]), c)) for w in workers]
    return int(np.argmin(est))


POLICIES = {"round_robin": round_robin, "prefix_affinity": prefix_affinity, "load_aware": load_aware}


def simulate(workload, policy, *, num_workers: int = 4, num_blocks: int = 64, block_size: int = 16,
             costs: Costs | None = None, slo_ms: float = 300.0) -> dict:
    c = costs or Costs()
    workers = [Worker(i, num_blocks, block_size) for i in range(num_workers)]
    state, rows = {}, []
    for req in sorted(workload, key=lambda r: r["arrival_ms"]):
        now = req["arrival_ms"]
        w = workers[policy(req, workers, now, c, state)]
        start = max(now, w.free_at)
        hit = w.cache.allocate_cached(req["rid"], req["tokens"], IDENT)
        end = start + prefill_ms(len(req["tokens"]), hit, c)
        w.cache.free(req["rid"])
        w.free_at = end
        w.served += 1
        rows.append({"rid": req["rid"], "worker": w.wid, "prefix_id": req["prefix_id"],
                     "arrival_ms": now, "start_ms": start, "ttft_ms": end - now,
                     "hit_tokens": hit, "prompt_tokens": len(req["tokens"])})
    return {"rows": rows, "summary": summarize(rows, num_workers, slo_ms)}


def quantile(xs, q: float) -> float:
    """Convention: numpy linear interpolation (method='linear')."""
    return float(np.quantile(np.asarray(xs, dtype=float), q))


def summarize(rows, num_workers: int, slo_ms: float) -> dict:
    ttft = [r["ttft_ms"] for r in rows]
    load = [0] * num_workers
    for r in rows:
        load[r["worker"]] += 1
    span_s = (max(r["arrival_ms"] for r in rows) - min(r["arrival_ms"] for r in rows)) / 1000 or 1.0
    good = sum(1 for t in ttft if t <= slo_ms)
    return {
        "n": len(rows),
        "hit_rate": sum(r["hit_tokens"] for r in rows) / sum(r["prompt_tokens"] for r in rows),
        "ttft_p50_ms": quantile(ttft, 0.50),
        "ttft_p95_ms": quantile(ttft, 0.95),
        "ttft_p99_ms": quantile(ttft, 0.99),
        "slo_attainment": good / len(rows),
        "slo_goodput_rps": good / span_s,
        "max_worker_share": max(load) / len(rows),
        "per_worker_load": load,
    }


def repeated(policy_name: str, seeds, **kw) -> dict:
    """Run one policy over several seeds; mean and sample std of each scalar metric."""
    wl_kw = {k: kw.pop(k) for k in ("n", "hot_share", "num_prefixes", "rate_per_s") if k in kw}
    n = wl_kw.pop("n", 400)
    runs = [simulate(make_workload(n, s, **wl_kw), POLICIES[policy_name], **kw)["summary"] for s in seeds]
    out = {"runs": runs}
    for key in ("hit_rate", "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms", "slo_attainment",
                "slo_goodput_rps", "max_worker_share"):
        vals = np.array([r[key] for r in runs], dtype=float)
        out[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    return out
