"""LAB D starter: trace features, direction comparison, hypothesis matrix, ranking.

Trace format: see fixtures/gen_traces.py. Times in microseconds, one clock.
Only events with pid == trace["meta"]["pid"] are "ours"; others are foreign activity.
Hypothesis matrix format: see fixtures/signatures.yaml.
Fill every TODO.
"""
import statistics

COUNTERS = ("grid_blocks", "regs_per_thread", "spill_bytes", "sectors_per_request",
            "branch_efficiency", "sm_clock_mhz")


def _ours(trace, kind="kernel"):
    pid = trace["meta"]["pid"]
    return sorted((e for e in trace["events"] if e["pid"] == pid and e["kind"] == kind), key=lambda e: e["start"])


def elapsed(trace):
    """First start to last end of OUR events (kernels and syncs). Provided."""
    pid = trace["meta"]["pid"]
    ev = [e for e in trace["events"] if e["pid"] == pid]
    return max(e["end"] for e in ev) - min(e["start"] for e in ev)


def utilization(trace):
    """Aggregate busy fraction of our kernels over our elapsed window (the dashboard number). Provided."""
    return sum(e["end"] - e["start"] for e in _ours(trace)) / elapsed(trace)


def _mean(values):
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def features(trace):
    """Return dict with keys:
    gap_time            sum of idle time between consecutive OUR kernels (never negative)
    kernel_time         summed duration of OUR kernels
    sync_calls          number of OUR sync events
    foreign_kernel_time summed duration of kernels from other pids
    and one key per name in COUNTERS: mean over OUR kernels (use _mean), None when every value is None.
    Mean, not median: a fault confined to one of four kernels must still move the feature.
    """
    raise NotImplementedError("TODO: features")


def direction(base, new, rel_tol=0.10):
    """'up' | 'down' | 'same' by relative change vs rel_tol; 'unknown' if either side is None.
    Zero baseline: any increase is 'up', otherwise 'same'."""
    raise NotImplementedError("TODO: direction")


def compare(base_trace, new_trace, rel_tol=0.10):
    """direction() for every feature key."""
    raise NotImplementedError("TODO: compare")


def score(hypotheses, observed):
    """For each hypothesis -> {"support": [...], "contradict": [...], "unknown": [...]} listing the
    features whose observed direction matches, contradicts, or cannot check the prediction."""
    raise NotImplementedError("TODO: score")


def diagnose(hypotheses, observed, min_ruled_down=2):
    """Return {"top", "ranked", "ruled_down", "unresolved", "scores"}.

    A hypothesis is ruled down when at least one prediction is contradicted.
    ranked: all hypotheses, fewest contradictions first, then most support.
    top: the single surviving hypothesis with the most support -- but ONLY if it
         is unique and at least `min_ruled_down` alternatives were ruled down;
         otherwise None, and `unresolved` lists the tied leading survivors.
    """
    raise NotImplementedError("TODO: diagnose")


def discriminating_features(hypotheses, a, b):
    """Sorted features that both hypotheses predict but in different directions."""
    raise NotImplementedError("TODO: discriminating_features")


def attribute_regression(base_trace, new_trace):
    """{"total_delta", "kernel_delta", "between_kernel_delta"} in microseconds, where
    total = elapsed(new) - elapsed(base) and between = total - kernel."""
    raise NotImplementedError("TODO: attribute_regression")


def recovered_fraction(base_elapsed, broken_elapsed, fixed_elapsed):
    """Share of the regression removed by an intervention. ValueError if broken <= base."""
    raise NotImplementedError("TODO: recovered_fraction")
