"""LAB D reference: trace features, direction comparison, hypothesis matrix, ranking.

Trace format: see fixtures/gen_traces.py. Times in microseconds, one clock.
Only events with pid == meta["pid"] are "ours"; others are foreign activity.
"""
import statistics

COUNTERS = ("grid_blocks", "regs_per_thread", "spill_bytes", "sectors_per_request",
            "branch_efficiency", "sm_clock_mhz")


def _ours(trace, kind="kernel"):
    pid = trace["meta"]["pid"]
    return sorted((e for e in trace["events"] if e["pid"] == pid and e["kind"] == kind), key=lambda e: e["start"])


def elapsed(trace):
    """First start to last end of OUR events (kernels and syncs)."""
    pid = trace["meta"]["pid"]
    ev = [e for e in trace["events"] if e["pid"] == pid]
    return max(e["end"] for e in ev) - min(e["start"] for e in ev)


def utilization(trace):
    """Aggregate busy fraction of our kernels over our elapsed window (the dashboard number)."""
    return sum(e["end"] - e["start"] for e in _ours(trace)) / elapsed(trace)


def _mean(values):
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def features(trace):
    ks = _ours(trace)
    gaps = sum(max(0.0, b["start"] - a["end"]) for a, b in zip(ks, ks[1:]))
    pid = trace["meta"]["pid"]
    out = {"gap_time": gaps, "kernel_time": sum(e["end"] - e["start"] for e in ks),
           "sync_calls": len(_ours(trace, "sync")),
           "foreign_kernel_time": sum(e["end"] - e["start"] for e in trace["events"]
                                      if e["kind"] == "kernel" and e["pid"] != pid)}
    for c in COUNTERS:
        out[c] = _mean(e.get(c) for e in ks)
    return out


def direction(base, new, rel_tol=0.10):
    """'up' | 'down' | 'same' | 'unknown' (either side missing). Zero baseline: any increase is 'up'."""
    if base is None or new is None:
        return "unknown"
    if base == 0:
        return "up" if new > 0 else "same"
    rel = (new - base) / abs(base)
    if rel > rel_tol:
        return "up"
    if rel < -rel_tol:
        return "down"
    return "same"


def compare(base_trace, new_trace, rel_tol=0.10):
    fb, fn = features(base_trace), features(new_trace)
    return {k: direction(fb[k], fn[k], rel_tol) for k in fb}


def score(hypotheses, observed):
    """Per hypothesis: features that support, contradict, or cannot be checked (unknown)."""
    out = {}
    for h, pred in hypotheses.items():
        s = {"support": [], "contradict": [], "unknown": []}
        for feat, want in pred.items():
            got = observed.get(feat, "unknown")
            s["unknown" if got == "unknown" else "support" if got == want else "contradict"].append(feat)
        out[h] = s
    return out


def diagnose(hypotheses, observed, min_ruled_down=2):
    """Rank hypotheses. A hypothesis is ruled down by >= 1 contradicted prediction.

    `top` is named only if it is the unique survivor with the most support AND at
    least `min_ruled_down` alternatives were ruled down (the scoring rule);
    otherwise top is None and `unresolved` lists the tied survivors.
    """
    s = score(hypotheses, observed)
    ranked = sorted(s, key=lambda h: (len(s[h]["contradict"]), -len(s[h]["support"]), h))
    ruled_down = [h for h in ranked if s[h]["contradict"]]
    survivors = [h for h in ranked if not s[h]["contradict"]]
    best = max((len(s[h]["support"]) for h in survivors), default=0)
    leaders = [h for h in survivors if len(s[h]["support"]) == best]
    top = leaders[0] if len(leaders) == 1 and len(ruled_down) >= min_ruled_down and best > 0 else None
    return {"top": top, "ranked": ranked, "ruled_down": ruled_down,
            "unresolved": [] if top else leaders, "scores": s}


def discriminating_features(hypotheses, a, b):
    """Features both hypotheses predict, with different predicted directions: measure these next."""
    pa, pb = hypotheses[a], hypotheses[b]
    return sorted(f for f in set(pa) & set(pb) if pa[f] != pb[f])


def attribute_regression(base_trace, new_trace):
    """Split the elapsed-time change into kernel-time and between-kernel components (Lesson 2.6 example)."""
    fb, fn = features(base_trace), features(new_trace)
    total = elapsed(new_trace) - elapsed(base_trace)
    k = fn["kernel_time"] - fb["kernel_time"]
    return {"total_delta": total, "kernel_delta": k, "between_kernel_delta": total - k}


def recovered_fraction(base_elapsed, broken_elapsed, fixed_elapsed):
    """Share of the regression an intervention removed (1.0 = fully recovered). Use unprofiled times."""
    reg = broken_elapsed - base_elapsed
    if reg <= 0:
        raise ValueError("no regression to recover")
    return (broken_elapsed - fixed_elapsed) / reg
