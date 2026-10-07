import math

import numpy as np
import pytest

import audit as a

UNIT_HINT = ("HINT: if you doubled requests per process but kept process count, "
             "what new process-level information did you gain?")
TAIL_HINT = ("HINT: tail exposure 1-(1-q)^n is a chance to see an event, not a "
             "confidence interval or percentile guarantee.")


def width(interval):
    return interval[2] - interval[1]


def test_generator_has_shared_process_effect_and_is_seeded():
    rows = a.simulate_clusters(7, 8, 100, process_sd=8.0, request_sd=0.01)
    again = a.simulate_clusters(7, 8, 100, process_sd=8.0, request_sd=0.01)
    assert rows == again, "HINT: preserve generator seed so raw clustered data are reproducible."
    groups = [[r["latency_ms"] for r in rows if r["process"] == p] for p in range(8)]
    within = np.mean([np.std(g) for g in groups])
    between = np.std([np.mean(g) for g in groups])
    assert between > within * 100, \
        "HINT: requests in a process share one process effect; they are correlated experimental units."


def test_request_interval_pseudoreplicates():
    rows = a.simulate_clusters(11, 6, 100, process_sd=8.0, request_sd=1.0)
    duplicated = [dict(r, request=r["request"] + k * 100) for k in range(20) for r in rows]
    naive_ratio = width(a.naive_request_interval(duplicated)) / width(a.naive_request_interval(rows))
    cluster_ratio = width(a.process_interval(duplicated)) / width(a.process_interval(rows))
    assert naive_ratio < 0.3, "HINT: naive request-level SE shrinks with copied within-process rows."
    assert math.isclose(cluster_ratio, 1.0, rel_tol=1e-12), \
        "HINT: process means and process count did not change. " + UNIT_HINT


def test_more_processes_help_more_than_more_requests_at_similar_total():
    few = a.simulate_clusters(17, 5, 200, process_sd=6.0, request_sd=1.0)
    many = a.simulate_clusters(17, 20, 50, process_sd=6.0, request_sd=1.0)
    assert width(a.process_interval(many)) < width(a.process_interval(few)), \
        "HINT: at similar total requests, more independent process starts support process-level claim."
    assert width(a.naive_request_interval(few)) < width(a.process_interval(few)) / 5, \
        "HINT: a narrow interval from many dependent requests is pseudoreplication. " + UNIT_HINT


def test_cluster_bootstrap_resamples_processes_and_is_reproducible():
    rows = a.simulate_clusters(2, 10, 80, process_sd=5.0, request_sd=1.0)
    x = a.cluster_bootstrap_interval(rows, seed=44, resamples=1000)
    assert x == a.cluster_bootstrap_interval(rows, seed=44, resamples=1000), \
        "HINT: record bootstrap seed; same seed and rows must reproduce interval."
    duplicated = [dict(r, request=r["request"] + k * 80) for k in range(3) for r in rows]
    y = a.cluster_bootstrap_interval(duplicated, seed=44, resamples=1000)
    assert np.allclose(x, y), \
        "HINT: whole-process bootstrap uses one process mean per draw, not request count. " + UNIT_HINT
    assert x[1] < x[0] < x[2], "HINT: report estimate with lower and upper cluster-bootstrap bounds."


def test_tail_exposure_formula_and_minimal_integer():
    q, target = 0.01, 0.95
    n = a.trials_for_exposure(q, target)
    assert a.tail_exposure(q, n) >= target, TAIL_HINT
    assert a.tail_exposure(q, n - 1) < target, \
        "HINT: round the logarithmic solution upward and prove n-1 fails. " + TAIL_HINT
    assert n == 299, "HINT: Lesson 0.4's 1% event / 95% exposure calculation rounds 298.07 upward."


def test_tail_exposure_is_monotone_and_validates_inputs():
    values = [a.tail_exposure(0.01, n) for n in (0, 1, 10, 100, 1000)]
    assert values == sorted(values) and values[0] == 0 and values[-1] < 1, TAIL_HINT
    for q, n in ((0, 1), (1, 1), (0.1, -1)):
        with pytest.raises(ValueError):
            a.tail_exposure(q, n)
    with pytest.raises(ValueError):
        a.trials_for_exposure(0.01, 1.0)


def test_intervals_reject_one_process():
    rows = a.simulate_clusters(1, 1, 100)
    with pytest.raises(ValueError):
        a.process_interval(rows)
    with pytest.raises(ValueError):
        a.cluster_bootstrap_interval(rows, seed=1)
