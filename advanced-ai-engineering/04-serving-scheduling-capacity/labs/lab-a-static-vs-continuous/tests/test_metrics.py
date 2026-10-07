import pytest
import sim

H = "HINT: Use one declared arrival boundary; TTFT = first output - arrival, not first output - admission."

def record(arrival=100.0, admit=220.0, tokens=(285.0, 310.0, 335.0)):
    r = sim.Record(sim.Request(0, arrival, 2048, len(tokens)), admit=admit)
    r.tokens[:] = tokens
    return r

def test_lesson_41_latency_identities_and_worked_example():
    r = record(tokens=(285.0,) + tuple(285 + 25*i for i in range(1, 200)))
    assert sim.queue_time(r) == 120.0, H
    assert sim.ttft(r) == 185.0, H
    assert sim.tpot(r) == pytest.approx(25.0), "HINT: TPOT divides post-first-output time by N_out - 1 gaps."
    assert sim.e2e(r) == pytest.approx(5160.0), "HINT: T_e2e = TTFT + sum(ITL)."
    assert sim.e2e(r) == pytest.approx(sim.ttft(r) + sim.itls(r).sum()), "HINT: Align output timestamps before decomposing latency."

def test_single_output_has_no_tpot():
    assert sim.tpot(record(tokens=(285.0,))) is None, "HINT: N_out=1 has no inter-output gap; report it separately."

def test_request_and_gap_weighting_are_explicit():
    a = record(tokens=(0.0, 10.0)); b = record(tokens=(0.0, 2.0, 4.0, 6.0))
    assert sim.mean_tpot([a, b], "request") == pytest.approx(6.0), "HINT: Request-weighted TPOT gives each eligible request one vote."
    assert sim.mean_tpot([a, b], "gap") == pytest.approx(4.0), "HINT: Gap-weighted TPOT divides total post-first time by total gaps."
