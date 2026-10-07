import math
import time

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import harness as h

PERIOD, DURATION, TIMEOUT, STALL_AT, STALL_S = 0.05, 10.0, 5.0, 5.0, 2.0
CO_HINT = ("HINT: during an injected 2-second stall, how many requests did your "
           "generator intend to issue, and how many latency samples did it record?")
Q_HINT = "HINT: what is n in your quantile, and where do timeouts sit in the sorted list?"


def run(gen, seed=7, stall=True, timeout=TIMEOUT, **kw):
    op = h.NoisyOp(seed, stall_at=STALL_AT if stall else None, stall_s=STALL_S, **kw)
    return gen(op, h.VirtualClock(), PERIOD, DURATION, timeout)


def ev(i, intended, actual, end, status="ok"):
    return h.Event(i, intended, actual, actual, end, status)


# ---- quantile convention -------------------------------------------------

@settings(max_examples=60, deadline=None)
@given(st.lists(st.floats(0, 1e3, allow_nan=False), min_size=1, max_size=60),
       st.floats(0.01, 1.0))
def test_nearest_rank_definition(xs, q):
    v = h.nearest_rank(xs, q)
    r = math.ceil(q * len(xs))
    assert v in xs, "HINT: nearest rank returns an observed value; it never interpolates."
    assert sum(x <= v for x in xs) >= r and sum(x < v for x in xs) < r, (
        "HINT: nearest rank is the ceil(q*n)-th smallest value (1-based). " + Q_HINT)


def test_nearest_rank_rejects_bad_input():
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            h.nearest_rank([1.0, 2.0], bad)
    with pytest.raises(ValueError):
        h.nearest_rank([], 0.5)


# ---- aggregation: population and censoring -------------------------------

def test_timeouts_stay_in_denominator_and_sort_last():
    rng = np.random.default_rng(3)
    n_ok, n_to, to = 90, 8, 2.0
    lat = rng.uniform(0.01, 0.5, n_ok)
    events = [ev(i, i, i, i + float(x)) for i, x in enumerate(lat)]
    events += [ev(n_ok + j, n_ok + j, n_ok + j, n_ok + j + to, "timeout") for j in range(n_to)]
    s = h.summarize(events, "intended", to, quantiles=(0.5, 0.95))
    assert (s["n_issued"], s["n_ok"], s["n_timeout"]) == (n_ok + n_to, n_ok, n_to), \
        "HINT: count every issued request; timeouts are outcomes, not missing data."
    assert math.isclose(s["failure_rate"], n_to / (n_ok + n_to)), \
        "HINT: failure rate's denominator is all issued requests."
    assert s["censored"][0.95] and s["quantiles"][0.95] == to, (
        "HINT: with timeouts in the sorted list, a high quantile can land on a timeout; "
        "report it censored at the timeout value. " + Q_HINT)
    assert not s["censored"][0.5] and s["quantiles"][0.5] < to, Q_HINT
    assert math.isclose(s["mean_ok_s"], float(np.mean(lat))), \
        "HINT: mean_ok_s is a success-only mean; label its population."


def test_origin_changes_the_number():
    events = [ev(0, 0.0, 0.3, 1.0), ev(1, 1.0, 1.0, 1.2)]
    a = h.summarize(events, "intended", 5.0, quantiles=(1.0,))
    b = h.summarize(events, "actual", 5.0, quantiles=(1.0,))
    assert a["quantiles"][1.0] > b["quantiles"][1.0], (
        "HINT: latency from intended arrival includes issue lag; latency from actual "
        "issue hides it. Which timestamp is your origin?")
    with pytest.raises(ValueError):
        h.summarize(events, "start", 5.0)


# ---- generators ----------------------------------------------------------

def test_open_loop_schedule_is_independent_of_completions():
    with_stall, without = run(h.run_open_loop), run(h.run_open_loop, stall=False)
    sched = [e.intended for e in with_stall]
    assert sched == [e.intended for e in without], (
        "HINT: an open-loop schedule is fixed before the run; a stall must not move it. " + CO_HINT)
    assert len(sched) == math.ceil(DURATION / PERIOD) and np.allclose(np.diff(sched), PERIOD), \
        "HINT: request k is intended at k*period for every k*period < duration."
    assert all(e.actual >= e.intended for e in with_stall), "HINT: nothing issues before it is intended."


def test_closed_loop_is_response_coupled():
    events = run(h.run_closed_loop)
    for prev, cur in zip(events, events[1:]):
        assert cur.actual >= prev.end, \
            "HINT: a closed-loop generator waits for the previous outcome before issuing."
        assert math.isclose(cur.intended, prev.actual + PERIOD), \
            "HINT: record when the next request was MEANT to go (prev actual + period), not when it went."
    assert all(e.actual < DURATION for e in events), \
        "HINT: stop before issuing any request at or beyond measurement duration."


def test_stall_suppresses_closed_loop_samples():
    open_ev, closed_ev = run(h.run_open_loop), run(h.run_closed_loop)
    in_win = lambda es: sum(STALL_AT <= e.intended < STALL_AT + STALL_S for e in es)
    assert in_win(closed_ev) * 5 < in_win(open_ev), CO_HINT
    assert len(closed_ev) < len(open_ev), CO_HINT


def test_coordinated_omission_hides_the_tail():
    o = h.summarize(run(h.run_open_loop), "intended", TIMEOUT)
    closed = run(h.run_closed_loop)
    ca = h.summarize(closed, "actual", TIMEOUT)
    assert ca["quantiles"][0.95] * 10 < o["quantiles"][0.95], (
        "HINT: response-coupled sampling measured from actual issue should look "
        "far better than open-loop during a stall. " + CO_HINT)
    lag = max(e.actual - e.intended for e in closed)
    assert lag > STALL_S / 2, (
        "HINT: a growing intended-minus-actual issue lag reveals coordinated omission; "
        "if your lag is ~0 you are not recording intended time independently.")


def test_timeouts_recorded_at_timeout_boundary():
    events = run(h.run_open_loop, timeout=0.5)
    tos = [e for e in events if e.status == "timeout"]
    assert tos, "HINT: a 2 s stall with a 0.5 s timeout must produce timeouts in the event log."
    assert all(math.isclose(e.end - e.actual, 0.5) for e in tos), \
        "HINT: a timed-out request's observed end is actual issue + timeout."
    s = h.summarize(events, "intended", 0.5)
    assert s["n_timeout"] == len(tos) and s["n_issued"] == len(events), Q_HINT


def test_virtual_clock_only(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("HINT: use the injected VirtualClock; wall-clock reads make tests nondeterministic.")
    for name in ("time", "perf_counter", "monotonic", "sleep"):
        monkeypatch.setattr(time, name, boom)
    a = [vars(e) for e in run(h.run_closed_loop, seed=11)]
    b = [vars(e) for e in run(h.run_closed_loop, seed=11)]
    assert a == b, "HINT: same seed and same virtual clock must give identical event logs."
