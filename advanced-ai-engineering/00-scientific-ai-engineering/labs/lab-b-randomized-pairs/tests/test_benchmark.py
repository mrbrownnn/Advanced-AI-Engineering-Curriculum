import math

import pytest
from scipy.stats import t

import benchmark as b

UNIT_HINT = "HINT: what was randomly assigned—requests, processes, or machines? That is your n."
ORDER_HINT = ("HINT: record the realized seeded A/B order inside every block; "
              "fixed A-then-B aliases monotonic drift with treatment.")


def test_randomized_orders_are_reproducible_balanced_and_not_fixed():
    a = b.randomized_orders(31, 2026)
    same = b.randomized_orders(31, 2026)
    other = b.randomized_orders(31, 2027)
    assert a == same, "HINT: recorded seed must exactly reproduce treatment assignment."
    assert a != other and len(set(a)) == 2, ORDER_HINT
    assert all(set(x) == {"A", "B"} for x in a), \
        "HINT: each block must contain A and B exactly once."
    assert abs(a.count(("A", "B")) - a.count(("B", "A"))) <= 1, \
        "HINT: balance AB and BA counts before seeded shuffling."


def test_simulation_records_assignment_and_nuisance_position():
    rows = b.simulate(3, 9, effect_ms=-1, drift_per_run_ms=0.4, noise_ms=0.0)
    assert len(rows) == 18 and {r.block for r in rows} == set(range(9)), UNIT_HINT
    for block in range(9):
        pair = [r for r in rows if r.block == block]
        assert [r.position for r in pair] == [0, 1], \
            "HINT: record within-block run position so carryover can be checked."
        assert "".join(r.treatment for r in pair) == pair[0].order == pair[1].order, ORDER_HINT
        assert [r.global_position for r in pair] == [2 * block, 2 * block + 1], \
            "HINT: monotonic drift follows global run position, not treatment label."
        assert all(r.seed == 3 for r in pair), "HINT: provenance includes assignment seed on raw rows."


def test_paired_differences_use_one_value_per_block():
    rows = [b.Run(8, 0, 0, "B", "BA", 9.0, 1), b.Run(8, 1, 1, "A", "BA", 10.0, 1),
            b.Run(2, 0, 2, "A", "AB", 20.0, 1), b.Run(2, 1, 3, "B", "AB", 23.5, 1)]
    assert b.paired_differences(rows) == [3.5, -1.0], \
        "HINT: calculate B-A within each block, then sort by block ID. " + UNIT_HINT
    with pytest.raises(ValueError):
        b.paired_differences(rows[:-1])
    with pytest.raises(ValueError):
        b.paired_differences(rows + [rows[0]])


def test_t_interval_is_block_level_and_shrinks_with_independent_blocks():
    d = [-1.0, 0.0, 1.0, 2.0]
    mean, lo, hi = b.t_interval(d)
    assert math.isclose(mean, 0.5), "HINT: interval center is mean of paired block differences."
    assert lo < mean < hi and math.isclose(mean - lo, hi - mean), \
        "HINT: two-sided t interval is symmetric around mean difference."
    expected_half = t.ppf(0.975, len(d) - 1) * 1.2909944487358056 / math.sqrt(len(d))
    assert math.isclose(hi - mean, expected_half), \
        "HINT: use Student t critical value, sample SD (ddof=1), and number of blocks."
    _, lo_more, hi_more = b.t_interval(d * 4)
    assert hi_more - lo_more < hi - lo, \
        "HINT: only additional independent block differences increase interval n. " + UNIT_HINT
    with pytest.raises(ValueError):
        b.t_interval([1.0])


def test_block_offsets_cancel_under_pairing():
    rows = b.simulate(4, 20, effect_ms=-2.0, drift_per_run_ms=0.0,
                      noise_ms=0.0, randomized=True)
    diffs = b.paired_differences(rows)
    assert all(math.isclose(x, -2.0) for x in diffs), \
        "HINT: a shared block nuisance offset cancels in within-block B-A differences."


def test_randomization_resists_drift_but_fixed_order_reverses():
    args = dict(seed=9, n_blocks=40, effect_ms=-1.0,
                drift_per_run_ms=1.5, noise_ms=0.1)
    random_est = b.t_interval(b.paired_differences(b.simulate(**args, randomized=True)))[0]
    fixed_est = b.t_interval(b.paired_differences(b.simulate(**args, randomized=False)))[0]
    assert random_est < -0.5, \
        "HINT: balanced within-block order should preserve direction of true B effect under monotonic drift."
    assert fixed_est > 0.0, \
        "HINT: when B always runs second, positive drift can reverse a true B improvement. " + ORDER_HINT
    assert abs(random_est - args["effect_ms"]) < abs(fixed_est - args["effect_ms"]), \
        "HINT: compare estimates to planted effect, not only p-values."


def test_as_records_preserves_raw_assignment():
    rows = b.simulate(1, 2, -1.0, 0.1, 0.0)
    records = b.as_records(rows)
    needed = {"block", "position", "global_position", "treatment", "order", "value_ms", "seed"}
    assert all(set(x) == needed for x in records), \
        "HINT: raw rows need treatment, block, realized order, nuisance position, value, and seed."
