"""LAB B reference: timing methods over an asynchronous device.

`timer` is any zero-argument callable returning host seconds (time.perf_counter
on Colab, SimClock.now in tests). `dev` exposes the CUDA-like subset used here:
launch, record_event, synchronize, event_synchronize; events expose elapsed(end).
"""
import statistics


def time_unsynced(dev, timer, work):
    """Anti-pattern kept for comparison: host timer around submission only."""
    t0 = timer()
    work(dev)
    return timer() - t0


def time_synced(dev, timer, work):
    """Synchronized host interval: drain earlier work, start, run, enforce completion, stop."""
    dev.synchronize()
    t0 = timer()
    work(dev)
    dev.synchronize()
    return timer() - t0


def time_events(dev, work, stream=0):
    """Device interval on one stream: events bracket work enqueued on `stream` only."""
    start = dev.record_event(stream)
    work(dev)
    end = dev.record_event(stream)
    dev.event_synchronize(end)
    return start.elapsed(end)


def time_application(dev, timer, setup, work, sync_each_op=False):
    """Application interval: setup + work + required final synchronization.

    sync_each_op=True wraps `work` so the device is drained after every launch,
    which is how per-operator synchronization destroys overlap.
    """
    dev.synchronize()
    t0 = timer()
    setup(dev)
    if sync_each_op:
        work(_SyncEachOp(dev))
    else:
        work(dev)
    dev.synchronize()
    return timer() - t0


class _SyncEachOp:
    def __init__(self, dev):
        self._dev = dev

    def launch(self, *args, **kwargs):
        out = self._dev.launch(*args, **kwargs)
        self._dev.synchronize()
        return out

    def __getattr__(self, name):
        return getattr(self._dev, name)


def summarize(samples, warmup):
    """Keep cold samples visible; warm statistics exclude them. Median is the stated estimator."""
    if warmup < 0 or warmup >= len(samples):
        raise ValueError("need 0 <= warmup < len(samples)")
    warm = list(samples[warmup:])
    return {"cold": list(samples[:warmup]), "warm": warm, "n_warm": len(warm),
            "warm_median": statistics.median(warm), "warm_min": min(warm), "warm_max": max(warm),
            "estimator": "median of warm samples"}
