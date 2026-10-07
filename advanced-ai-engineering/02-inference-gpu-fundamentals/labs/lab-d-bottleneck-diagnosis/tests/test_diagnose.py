"""LAB D: analysis functions over seeded synthetic traces with planted causes."""
import sys
from pathlib import Path

import pytest
import yaml

FIX = Path(__file__).resolve().parents[1] / "fixtures"
sys.path.insert(0, str(FIX))
import diagnose as D  # noqa: E402
from gen_traces import FAULTS, make_trace  # noqa: E402

HYP = yaml.safe_load((FIX / "signatures.yaml").read_text())["hypotheses"]
BASE = make_trace(None, seed=1)
H_WHERE = "HINT: does the regression live inside kernels or between them on the timeline?"
H_RULE = "HINT: no credit for naming a bottleneck without ruling down at least two alternatives."


def test_every_fault_slows_elapsed():
    for f in FAULTS:
        assert D.elapsed(make_trace(f, seed=2)) > 1.2 * D.elapsed(BASE), f"HINT: fixture sanity -- {f} must regress elapsed time by >20%."


def test_features_shape():
    f = D.features(BASE)
    assert set(f) >= {"gap_time", "kernel_time", "sync_calls", "foreign_kernel_time", *D.COUNTERS}, (
        "HINT: return every documented feature key.")
    assert f["gap_time"] > 0 and f["sync_calls"] == 0 and f["foreign_kernel_time"] == 0, (
        "HINT: gaps are idle time between OUR consecutive kernels; foreign = other pids only.")


def test_foreign_kernels_do_not_count_as_ours():
    f = D.features(make_trace("background_activity", seed=2))
    assert f["gap_time"] >= 0.5 * D.features(BASE)["gap_time"], (
        "HINT: overlapping foreign kernels are another process's work, not negative gaps in yours.")
    assert f["foreign_kernel_time"] > 0, "HINT: filter by pid to separate our kernels from background activity."


def test_direction_rules():
    assert [D.direction(100, x) for x in (130, 70, 105)] == ["up", "down", "same"], "HINT: relative change vs rel_tol."
    assert D.direction(0, 5) == "up" and D.direction(0, 0) == "same", "HINT: a zero baseline has no relative change."
    assert D.direction(None, 1) == "unknown" == D.direction(1, None), (
        "HINT: missing evidence is 'unknown', never 'same' -- absence of a counter is not a measurement.")


def test_regression_attribution_inside_vs_between():
    gap = D.attribute_regression(BASE, make_trace("host_launch_gap", seed=2))
    assert gap["between_kernel_delta"] > 0.9 * gap["total_delta"], H_WHERE
    stride = D.attribute_regression(BASE, make_trace("strided_loads", seed=2))
    assert stride["kernel_delta"] > 0.9 * stride["total_delta"], H_WHERE
    for a in (gap, stride):
        assert a["kernel_delta"] + a["between_kernel_delta"] == pytest.approx(a["total_delta"]), H_WHERE


def test_utilization_alone_cannot_separate_paired_faults():
    pairs = [("host_launch_gap", "synchronization"), ("strided_loads", "register_pressure")]
    for a, b in pairs:
        ua, ub = D.utilization(make_trace(a, seed=3)), D.utilization(make_trace(b, seed=3))
        assert abs(ua - ub) < 0.05, f"HINT: utilization is non-discriminating -- {a} and {b} must look alike."
        assert D.discriminating_features(HYP, a, b), (
            f"HINT: name the measurement whose predicted direction differs between {a} and {b}.")


@pytest.mark.parametrize("fault", FAULTS)
def test_full_evidence_diagnoses_planted_fault(fault):
    obs = D.compare(BASE, make_trace(fault, seed=4))
    d = D.diagnose(HYP, obs)
    assert d["top"] == fault, f"{H_RULE} Planted {fault}: check score() and the ruled-down rule."
    assert len(d["ruled_down"]) >= 2 and fault not in d["ruled_down"], H_RULE
    assert set(d["ranked"]) == set(HYP), "HINT: rank every hypothesis, not only the winner."


def test_without_counters_strided_and_register_pressure_stay_unresolved():
    obs = D.compare(make_trace(None, seed=1, counters=False), make_trace("strided_loads", seed=4, counters=False))
    d = D.diagnose(HYP, obs)
    assert d["top"] is None, ("HINT: timeline-only evidence cannot tell strided loads from register pressure; "
                              "refuse to name one and request the discriminating counter instead.")
    assert {"strided_loads", "register_pressure"} <= set(d["unresolved"]), (
        "HINT: report which hypotheses remain tied, and why.")


def test_timeline_alone_separates_gap_from_sync():
    obs = D.compare(make_trace(None, seed=1, counters=False), make_trace("synchronization", seed=4, counters=False))
    assert D.diagnose(HYP, obs)["top"] == "synchronization", (
        "HINT: the Systems timeline shows sync calls; you do not need kernel counters to separate these two.")


def test_needs_two_rulings_down():
    tiny = {"a": {"gap_time": "up"}, "b": {"gap_time": "down"}}
    d = D.diagnose(tiny, {"gap_time": "up"})
    assert d["top"] is None, H_RULE


def test_recovered_fraction():
    assert D.recovered_fraction(4.0, 6.0, 4.1) == pytest.approx(0.95), "HINT: (broken - fixed) / (broken - base)."
    with pytest.raises(ValueError):
        D.recovered_fraction(4.0, 4.0, 3.0)
