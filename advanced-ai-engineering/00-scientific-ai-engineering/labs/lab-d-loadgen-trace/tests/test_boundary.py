import json
import math
from pathlib import Path

import pytest

import boundary as b

BOUNDARY_HINT = ("HINT: scheduled-time origin retains issue delay; actual-issue "
                 "origin removes it while SUT completion stays fixed.")
POP_HINT = "HINT: what is n in your quantile, and where do timeouts sit in the sorted list?"


def test_fixture_and_generator_are_same_fixed_sut_log():
    path = Path(__file__).resolve().parents[1] / "fixtures" / "sut-events.jsonl"
    fixture = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    generated = b.synthetic_log()
    assert generated == fixture, \
        "HINT: re-score one unchanged synthetic SUT event log; do not regenerate work per policy."
    assert len(generated) == 10 and sum(x["status"] == "timeout" for x in generated) == 2, \
        "HINT: preserve all issued outcomes, including timeout rows."


def test_scheduled_origin_includes_issue_lag():
    events = b.synthetic_log()
    sched = b.rescore(events, "scheduled", "success", "nearest_rank", 0.80, 2000)
    actual = b.rescore(events, "actual_issue", "success", "nearest_rank", 0.80, 2000)
    assert sched["quantile_ms"] > actual["quantile_ms"], BOUNDARY_HINT
    assert events == b.synthetic_log(), \
        "HINT: change metric boundary only; fixed SUT event fields must not move."


def test_all_issued_population_exposes_timeouts():
    events = b.synthetic_log()
    success = b.rescore(events, "actual_issue", "success", "nearest_rank", 0.9, 2000)
    all_issued = b.rescore(events, "actual_issue", "all", "nearest_rank", 0.9, 2000)
    assert success["n_population"] == 8 and all_issued["n_population"] == 10, POP_HINT
    assert all_issued["quantile_ms"] == 2000 > success["quantile_ms"], \
        "HINT: all-issued high quantile lands on timeout censoring; success-only silently changes population."
    assert math.isclose(all_issued["failure_rate"], 0.2), \
        "HINT: report failures over all issued requests even for a success-only latency metric."


def test_quantile_conventions_use_observed_values_and_can_differ():
    values = [1, 2, 3, 4]
    assert b.nearest_rank(values, 0.5) == 2, \
        "HINT: nearest rank is ceil(q*n)-th sorted value using 1-based rank."
    assert b.loadgen_floor(values, 0.5) == 3, \
        "HINT: pinned LoadGen uses sorted[floor(n*q)] with a 0-based index."
    events = b.synthetic_log()
    nr = b.rescore(events, "scheduled", "success", "nearest_rank", 0.75, 2000)
    lg = b.rescore(events, "scheduled", "success", "loadgen_floor", 0.75, 2000)
    assert nr["quantile_ms"] != lg["quantile_ms"], \
        "HINT: same sample can change when quantile convention changes; declare convention."


def test_invalid_policies_are_rejected():
    events = b.synthetic_log()
    for args in (("start", "all", "nearest_rank"),
                 ("scheduled", "completed", "nearest_rank"),
                 ("scheduled", "all", "interpolated")):
        with pytest.raises(ValueError):
            b.rescore(events, *args, q=0.5, timeout_ms=100)
    with pytest.raises(ValueError):
        b.nearest_rank([], 0.5)
    with pytest.raises(ValueError):
        b.loadgen_floor([1], 1.0)


def test_score_metadata_makes_boundary_auditable():
    score = b.rescore(b.synthetic_log(), "scheduled", "all", "loadgen_floor", 0.75, 2000)
    needed = {"n_population", "n_issued", "n_failures", "failure_rate", "q",
              "quantile_ms", "origin", "population", "quantile_convention"}
    assert set(score) == needed, \
        "HINT: a percentile without population, origin, convention, and outcome counts is incomplete."
