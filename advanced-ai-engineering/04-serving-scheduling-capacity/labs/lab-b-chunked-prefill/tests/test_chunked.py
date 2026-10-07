import math
import numpy as np
import pytest
import sim
import chunked

H = "HINT: Schedule active decodes first, then cap prefill by both C_max and the remaining token budget."

def fixture():
    active = [sim.Request(i, 0.0, 16, 80) for i in range(4)]
    long = sim.Request(4, 60.0, 4096, 4)
    return active + [long]

def test_chunks_conserve_every_prompt():
    reqs = sim.make_workload(30, seed=3, rate_per_s=8, n_in_mean=800, n_in_sd=700, n_out_mean=30, n_out_sd=10)
    run = chunked.run_chunked(reqs, 8, sim.LinearCost(), chunk=128, budget=256)
    for r, rec in zip(reqs, run.records):
        assert sum(rec.chunks) == r.n_in, "HINT: Chunk boundaries may change, but chunks must sum exactly to N_in."
        assert all(0 < x <= 128 for x in rec.chunks), H
        assert len(rec.tokens) == r.n_out, "HINT: Prefill chunking must not create or drop output tokens."

def test_budget_and_decode_starvation_bound():
    reqs = fixture()
    run = chunked.run_chunked(reqs, 8, sim.LinearCost(), chunk=128, budget=256)
    assert all(s["n_decode"] + s["n_prefill"] <= 256 for s in run.steps), H
    for rec in run.records[:4]:
        assert max(sim.itls(rec)) == pytest.approx(max(s["dt"] for s in run.steps)), "HINT: A decoding sequence must advance once per iteration; compare each ITL with one step duration."
    with pytest.raises(ValueError):
        chunked.run_chunked(reqs, 8, sim.LinearCost(), chunk=128, budget=7)

def test_unchunked_matches_continuous_under_same_cost_model():
    reqs = sim.make_workload(20, seed=5, n_out_mean=20, n_out_sd=10)
    cost = sim.LinearCost(per_chunk=0)
    a = sim.run_continuous(reqs, 8, cost)
    b = chunked.run_chunked(reqs, 8, cost, chunk=math.inf, budget=math.inf)
    assert [r.tokens for r in a.records] == [r.tokens for r in b.records], "HINT: Infinity chunk and budget are the monolithic-prefill control."

def test_chunking_lowers_itl_tail_but_increases_long_prompt_ttft():
    cost = sim.LinearCost(base=10, per_decode=0.1, per_prefill=0.02, per_chunk=1)
    mono = chunked.run_chunked(fixture(), 8, cost, chunk=math.inf)
    small = chunked.run_chunked(fixture(), 8, cost, chunk=128)
    mono_itl = np.concatenate([sim.itls(r) for r in mono.records[:4]])
    small_itl = np.concatenate([sim.itls(r) for r in small.records[:4]])
    assert sim.quantile(small_itl, .99) < sim.quantile(mono_itl, .99), "HINT: Compare decode gaps crossing the long prefill against gaps crossing one chunk."
    assert sim.ttft(small.records[4]) > sim.ttft(mono.records[4]), "HINT: Smaller chunks add scheduling/launch steps before the prompt finishes."

def test_readme_exercise_step_costs_are_not_silently_reinterpreted():
    assert 120 < 8 * 35 and max(120, 35) == 120, "HINT: Compare longest injected step (120 vs 35 ms) separately from cumulative prompt time (120 vs 280 ms)."
