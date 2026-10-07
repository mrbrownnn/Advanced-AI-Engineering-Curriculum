"""Multi-worker cache-aware routing simulation (Lesson 3.7). Fill the TODOs.

Each worker is a single FIFO server with its own PrefixCachingBlockManager. A
request's service time is its prefill: miss tokens cost more than hit tokens.
TTFT = service end - arrival. Routing sees the exact worker cache state
(a real router's view is stale/approximate).
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
    raise NotImplementedError("HINT: TODO: Lesson 3.7")


def break_even_wait(other_wait_ms: float, miss_ms: float, hit_ms: float) -> float:
    """Largest hit-holder queue wait at which routing to the hit still ties the miss worker."""
    raise NotImplementedError("HINT: TODO: Lesson 3.7")


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
    """Send to the worker with the longest cached prefix (ties -> lowest index).
    A prefix cached nowhere goes to its home worker, prefix_id mod len(workers)."""
    raise NotImplementedError("HINT: TODO")


def load_aware(req, workers, now, c, state):
    """Your utility-aware policy. Suggested: minimise estimated TTFT
    (queue wait + prefill given that worker's hit tokens)."""
    raise NotImplementedError("HINT: TODO")


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
    """Convention: numpy linear interpolation (method='linear'). State it in your report."""
    return float(np.quantile(np.asarray(xs, dtype=float), q))


def summarize(rows, num_workers: int, slo_ms: float) -> dict:
    """Return n, hit_rate (hit tokens / prompt tokens), ttft_p50_ms, ttft_p95_ms,
    ttft_p99_ms, slo_attainment (share with TTFT <= slo_ms), slo_goodput_rps
    (SLO-meeting requests per second of arrival span), max_worker_share, per_worker_load."""
    raise NotImplementedError("HINT: TODO")


def repeated(policy_name: str, seeds, **kw) -> dict:
    """Run one policy over several seeds; mean and sample std (ddof=1) of each scalar metric."""
    wl_kw = {k: kw.pop(k) for k in ("n", "hot_share", "num_prefixes", "rate_per_s") if k in kw}
    n = wl_kw.pop("n", 400)
    runs = [simulate(make_workload(n, s, **wl_kw), POLICIES[policy_name], **kw)["summary"] for s in seeds]
    out = {"runs": runs}
    for key in ("hit_rate", "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms", "slo_attainment",
                "slo_goodput_rps", "max_worker_share"):
        vals = np.array([r[key] for r in runs], dtype=float)
        out[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    return out
