"""LAB B: prove which timing boundaries include queued device work. Simulated clock only."""
import pytest
from hypothesis import Phase, given, settings
from hypothesis import strategies as st

import timing as M
from simdevice import SimClock, SimDevice, kernels_work

DET = settings(derandomize=True, max_examples=100, deadline=None, database=None, phases=[Phase.explicit, Phase.generate])
H_SYNC = "HINT: does the reported time change when you add a synchronization before stopping the timer?"
H_BOUND = "HINT: submission versus completion boundaries -- which earlier or concurrent work is inside your interval?"


def fresh(**kw):
    clock = SimClock()
    return clock, SimDevice(clock, **kw)


def device_busy(dev, stream=None):
    return sum(e - s for st_, _, s, e in dev.kernels if stream is None or st_ == stream)


def test_unsynced_timer_underreports_long_kernels():
    clock, dev = fresh()
    work = kernels_work(4, 1e-3)
    unsynced = M.time_unsynced(dev, clock.now, work)
    clock2, dev2 = fresh()
    synced = M.time_synced(dev2, clock2.now, work)
    assert unsynced < 0.1 * synced, f"{H_SYNC} The unsynchronized number is submission time."
    assert synced >= 4e-3, f"{H_SYNC} A completion boundary must include all four 1 ms kernels."


@DET
@given(n=st.integers(1, 20), dur=st.floats(1e-6, 5e-3), launch=st.floats(1e-7, 5e-5), lat=st.floats(0, 5e-5))
def test_synced_and_event_intervals_bound_device_work(n, dur, launch, lat):
    clock, dev = fresh(launch_cost=launch, queue_latency=lat)
    synced = M.time_synced(dev, clock.now, kernels_work(n, dur))
    assert synced >= n * dur * (1 - 1e-9), f"{H_SYNC} Synchronized host time cannot be shorter than the work it bounds."
    clock, dev = fresh(launch_cost=launch, queue_latency=lat)
    ev = M.time_events(dev, kernels_work(n, dur))
    assert ev >= n * dur * (1 - 1e-9), f"{H_BOUND} Events bracket every kernel enqueued between them on the stream."
    assert ev <= synced + 1e-12, f"{H_BOUND} The device interval excludes host launch and sync-return costs."


def test_events_ordering_matches_worked_example_shape():
    # one host clock, same stream: submission < device interval < synchronized host interval
    clock, dev = fresh(launch_cost=20e-6, queue_latency=5e-6, sync_return=5e-6)
    sub = M.time_unsynced(dev, clock.now, kernels_work(1, 170e-6))
    clock, dev = fresh(launch_cost=20e-6, queue_latency=5e-6, sync_return=5e-6)
    ev = M.time_events(dev, kernels_work(1, 170e-6))
    clock, dev = fresh(launch_cost=20e-6, queue_latency=5e-6, sync_return=5e-6)
    syn = M.time_synced(dev, clock.now, kernels_work(1, 170e-6))
    assert sub < ev < syn, f"{H_BOUND} Three boundaries, three different, non-conflicting numbers."


def test_event_read_requires_completion():
    clock, dev = fresh()
    ev = M.time_events(dev, kernels_work(3, 2e-3))
    busy = device_busy(dev)
    assert busy <= ev < busy + 3 * dev.launch_cost + dev.queue_latency + 1e-12, (
        "HINT: record start, enqueue, record end, then synchronize on the END event before reading elapsed time.")


def test_synced_excludes_earlier_queued_work():
    clock, dev = fresh()
    kernels_work(1, 10e-3)(dev)  # someone else's 10 ms kernel still queued
    t = M.time_synced(dev, clock.now, kernels_work(1, 1e-3))
    assert t < 5e-3, (f"{H_BOUND} Drain the queue before reading the start time, or earlier work leaks "
                      "into your interval.")


def test_event_on_one_stream_does_not_bound_another_stream():
    clock, dev = fresh()

    def two_streams(d):
        d.launch(1e-3, stream=0)
        d.launch(8e-3, stream=1)

    ev = M.time_events(dev, two_streams, stream=0)
    clock2, dev2 = fresh()
    app = M.time_application(dev2, clock2.now, lambda d: None, two_streams)
    assert ev < 2e-3, f"{H_BOUND} Events on stream 0 do not wait for stream-1 work."
    assert app >= 8e-3, f"{H_BOUND} The application boundary must include the queued stream-1 work."


def test_per_op_sync_destroys_overlap():
    def overlapped(d):
        for _ in range(4):
            d.launch(1e-3, stream=0)
            d.launch(1e-3, stream=1)

    c1, d1 = fresh()
    c2, d2 = fresh()
    free = M.time_application(d1, c1.now, lambda d: None, overlapped)
    serial = M.time_application(d2, c2.now, lambda d: None, overlapped, sync_each_op=True)
    assert serial > 1.5 * free, ("HINT: synchronizing after every operator serializes the streams; that number "
                                 "is not production overlap.")


def test_application_includes_setup():
    clock, dev = fresh()
    app = M.time_application(dev, clock.now, lambda d: d.clock.advance(3e-3), kernels_work(1, 1e-3))
    assert app >= 4e-3, "HINT: the application interval includes host setup, dispatch, and the final synchronization."


def test_summarize_keeps_cold_separate():
    clock, dev = fresh(first_launch_cost=50e-3)
    samples = [M.time_synced(dev, clock.now, kernels_work(2, 1e-3)) for _ in range(6)]
    s = M.summarize(samples, warmup=1)
    assert s["cold"] == samples[:1] and s["n_warm"] == 5, (
        "HINT: report cold start and steady state separately; do not silently drop the first sample.")
    assert s["cold"][0] > 10 * s["warm_median"], "HINT: the first launch pays context/init cost; keep it visible."
    assert s["warm_min"] <= s["warm_median"] <= s["warm_max"], "HINT: median lies between min and max."
    assert isinstance(s["estimator"], str) and s["estimator"], "HINT: name the estimator you report."
    for bad in (-1, 6):
        with pytest.raises(ValueError):
            M.summarize(samples, bad)


def test_break_module_excludes_earlier_work():
    import bugged_timing as B
    clock, dev = fresh()
    kernels_work(1, 10e-3)(dev)
    t = B.time_synced(dev, clock.now, kernels_work(1, 1e-3))
    assert t < 5e-3, ("HINT: BREAK -- bugged_timing.time_synced reports the wrong interval in some scenario. "
                      "Build that scenario with simdevice, fix it in place, and record it in report.md section 6.")
