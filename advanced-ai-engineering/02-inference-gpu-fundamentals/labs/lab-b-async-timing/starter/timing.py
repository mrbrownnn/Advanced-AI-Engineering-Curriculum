"""LAB B starter: timing methods over an asynchronous device.

`timer` is any zero-argument callable returning host seconds (time.perf_counter
on Colab, SimClock.now in tests). `dev` exposes the CUDA-like subset used here:
launch, record_event, synchronize, event_synchronize; events expose elapsed(end).
On Colab, experiments/torch_device.py adapts torch.cuda to this interface.

`work` is a callable work(dev) that enqueues kernels via dev.launch(...).
Fill every TODO.
"""
import statistics  # noqa: F401  (useful for summarize)


def time_unsynced(dev, timer, work):
    """Anti-pattern kept for comparison: host timer around submission only. Provided."""
    t0 = timer()
    work(dev)
    return timer() - t0


def time_synced(dev, timer, work):
    """Synchronized host interval for `work` alone.

    Earlier queued work must not leak into the interval, and the interval must
    end only after `work` has completed on the device.
    """
    raise NotImplementedError("TODO: time_synced")


def time_events(dev, work, stream=0):
    """Device interval measured with events recorded on `stream` around `work`.

    Read the elapsed time only after the end event is known to have completed.
    """
    raise NotImplementedError("TODO: time_events")


def time_application(dev, timer, setup, work, sync_each_op=False):
    """Application interval: setup(dev) + work(dev) + the final synchronization a caller needs.

    When sync_each_op is True, the device must be drained after EVERY launch that
    `work` performs (hint: pass work a thin wrapper whose launch() synchronizes).
    """
    raise NotImplementedError("TODO: time_application")


def summarize(samples, warmup):
    """Return dict: cold (first `warmup` samples), warm (rest), n_warm, warm_median,
    warm_min, warm_max, estimator (a string naming the estimator).

    Raise ValueError unless 0 <= warmup < len(samples).
    """
    raise NotImplementedError("TODO: summarize")
