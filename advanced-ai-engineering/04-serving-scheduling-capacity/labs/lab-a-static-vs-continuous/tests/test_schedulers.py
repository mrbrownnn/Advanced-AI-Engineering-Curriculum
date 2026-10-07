import sim

H = "HINT: Evict finished sequences, release their slot, then admit waiting work on the next iteration."

def test_static_worked_example_has_55_percent_bubbles():
    reqs = [sim.Request(i, 0.0, 8, n) for i, n in enumerate((10, 50, 20, 100))]
    run = sim.run_static(reqs, 4, sim.LinearCost())
    assert sim.bubble_fraction(run) == 0.55, "HINT: Static useful slots are 180 of 4 x 100 = 400."

def test_token_conservation_and_arrival_boundary():
    reqs = sim.make_workload(40, seed=41, rate_per_s=12, n_out_sd=80)
    for fn in (sim.run_static, sim.run_continuous):
        run = fn(reqs, 8, sim.LinearCost())
        assert [len(x.tokens) for x in run.records] == [r.n_out for r in reqs], "HINT: Every request must emit exactly N_out tokens."
        assert all(x.admit >= x.req.arrival and x.tokens[0] >= x.req.arrival for x in run.records), "HINT: No request may be admitted or served before arrival."
        assert all(a <= b for x in run.records for a, b in zip(x.tokens, x.tokens[1:])), "HINT: Output timestamps must be monotone."

def test_continuous_refills_first_freed_slot():
    reqs = [sim.Request(i, 0.0, 1, n) for i, n in enumerate((2, 9, 9))]
    run = sim.run_continuous(reqs, 2, sim.LinearCost(base=1, per_decode=0, per_prefill=0))
    assert run.records[2].admit < run.records[1].finish, H

def test_continuous_throughput_not_lower_when_slots_free_early():
    reqs = [sim.Request(i, 0.0, 8, n) for i, n in enumerate((2, 40, 3, 35, 4, 30, 5, 25))]
    cost = sim.LinearCost(base=1, per_decode=0.2, per_prefill=0)
    static = sim.summarize(sim.run_static(reqs, 4, cost), slo_ttft_ms=1e9, slo_tpot_ms=1e9)
    cont = sim.summarize(sim.run_continuous(reqs, 4, cost), slo_ttft_ms=1e9, slo_tpot_ms=1e9)
    assert cont["throughput_tok_s"] >= static["throughput_tok_s"], H

def test_identical_lengths_falsify_universal_superiority_with_overhead():
    reqs = [sim.Request(i, 0.0, 128, 128) for i in range(8)]
    cost = sim.LinearCost(base=1, per_decode=0, per_prefill=0, sched_ms=0.1)
    a = sim.summarize(sim.run_static(reqs, 4, cost), slo_ttft_ms=1e9, slo_tpot_ms=1e9)
    b = sim.summarize(sim.run_continuous(reqs, 4, cost), slo_ttft_ms=1e9, slo_tpot_ms=1e9)
    assert a["throughput_tok_s"] > b["throughput_tok_s"], "HINT: Static has zero padding bubbles when every length is identical; count scheduler decisions."
