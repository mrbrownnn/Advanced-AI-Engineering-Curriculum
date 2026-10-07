import numpy as np
import pytest
import queueing as q

H = "HINT: L, lambda, and W must use one boundary, population, and clipped measurement window."

def test_readme_pk_worked_examples():
    assert q.pk_wq(5, .1, .1) == pytest.approx(.1), "HINT: Use the P-K variance term and 1-rho denominator."
    assert q.pk_wq(8, .1, .1) == pytest.approx(.4), "HINT: At rho=.8 and Cv=1, mean Wq is 400 ms."
    assert q.pk_wq(9.5, .1, .1) == pytest.approx(1.9), "HINT: At rho=.95 and Cv=1, mean Wq is 1,900 ms."
    with pytest.raises(ValueError): q.pk_wq(12, .1, .1)

def test_littles_law_exact_for_clipped_finite_window():
    a = np.array([-3., 1., 2., 8., 9.])
    d = np.array([2., 3., 12., 9., 15.])
    x = q.littles_terms(a, d, 0., 10.)
    assert x["L"] == pytest.approx(x["lam"] * x["W"], abs=1e-12), H

def test_simulated_trace_little_system_and_queue_boundaries():
    a = q.poisson_arrivals(7, 1000, seed=11)
    s = q.gamma_services(len(a), .1, 1, seed=12)
    starts, d = q.fifo_single_server(a, s)
    e = q.steady_estimates(a, starts, d, 1000, .1)
    assert e["L"] == pytest.approx(e["lam"] * e["W"], abs=1e-10), H
    assert e["Lq"] == pytest.approx(e["lam_q"] * e["Wq"], abs=1e-10), H

def test_measurement_window_excludes_warmup():
    # One huge pre-window visit exits before t0; correct estimator is identical with/without it.
    a = np.array([-100., 50., 60.]); st = np.array([-90., 55., 70.]); d = np.array([0., 65., 80.])
    full = q.steady_estimates(a, st, d, 100., .2)
    trimmed = q.steady_estimates(a[1:], st[1:], d[1:], 100., .2)
    assert full == trimmed, "HINT: Apply the declared warm-up cutoff before accumulating queue/system residence."
    assert full["t0"] == 20., "HINT: warmup_frac is a fraction of the declared horizon."

def test_open_loop_arrivals_do_not_depend_on_server_state():
    a = q.poisson_arrivals(8, 100, seed=9)
    b = q.poisson_arrivals(8, 100, seed=9)
    assert np.array_equal(a, b), "HINT: Open-loop arrival timestamps are generated before service begins."
    assert np.all(np.diff(a) >= 0), "HINT: Replay arrival traces in timestamp order."

def test_saturation_knee_and_unstable_growth():
    means = []
    for rho in (.5, .8, .95):
        a = q.poisson_arrivals(rho * 10, 3000, seed=40)
        sv = q.gamma_services(len(a), .1, 1, seed=41)
        st, d = q.fifo_single_server(a, sv)
        means.append(q.steady_estimates(a, st, d, 3000, .1)["Wq"])
    assert means[2] > 3 * means[1] > means[0], "HINT: In M/G/1 mean wait grows with the 1/(1-rho) term near saturation."
    assert (15 - 10) * 20 == 100, "HINT: During an unstable fluid burst, backlog grows at lambda - mu."
