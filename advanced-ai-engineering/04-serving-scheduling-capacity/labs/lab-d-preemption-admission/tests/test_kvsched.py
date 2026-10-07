import pytest
import sim
import kvsched as k

H = "HINT: Align KV allocation failures, preemption events, recompute/transfer work, and useful progress before diagnosing thrashing."

def workload(n=50):
    return sim.make_workload(n, seed=5, n_in_mean=256, n_in_sd=64, n_out_mean=350, n_out_sd=100)

def test_block_counter_rounding_and_conservation():
    assert [k.blocks_for(x, 16) for x in (0, 1, 16, 17)] == [0, 1, 1, 2], "HINT: Physical allocation rounds logical token demand up to whole blocks."
    states, steps = k.run_kv(workload(20), sim.LinearCost(), k.Policy(n_blocks=400, B=16))
    assert all(0 <= s["used"] <= 400 for s in steps), "HINT: Allocated blocks must stay between zero and the physical pool capacity."
    assert all(x.status in {"done", "rejected", "cancelled"} for x in states), "HINT: Every admitted request must reach a terminal state."
    assert all(len(x.rec.tokens) == x.rec.req.n_out for x in states if x.status == "done"), "HINT: Completion requires exactly N_out emitted tokens."

def test_recompute_is_charged_and_swap_is_accounted():
    cost = sim.LinearCost(base=10, per_decode=.1, per_prefill=.2)
    rec, rs = k.run_kv(workload(), cost, k.Policy(n_blocks=400, B=32, mode="recompute"))
    sw, ss = k.run_kv(workload(), cost, k.Policy(n_blocks=400, B=32, mode="swap"))
    ro = k.outcomes(rec, rs, slo_ttft_ms=1e9); so = k.outcomes(sw, ss, slo_ttft_ms=1e9)
    assert ro["preemptions"] > 0 and ro["wasted_tokens"] > 0, "HINT: Recompute must re-prefill prompt plus emitted prefix and charge those tokens to T_step."
    assert so["preemptions"] > 0 and so["swap_blocks"] > 0, "HINT: Swap must charge both eviction and restore transfer blocks."
    assert so["wasted_tokens"] == 0, "HINT: Swapping preserves computed KV; do not label it recompute waste."

def test_admission_control_reports_rejected_utility():
    w = sim.make_workload(100, seed=8, rate_per_s=20, n_in_mean=256, n_in_sd=64, n_out_mean=400, n_out_sd=100)
    states, steps = k.run_kv(w, sim.LinearCost(), k.Policy(n_blocks=400, B=32, admission="reserve", queue_limit=4))
    o = k.outcomes(states, steps, slo_ttft_ms=2000)
    assert o["rejected"] > 0 and o["completed"] > 0, "HINT: Bounded admission trades rejected utility for protected admitted work; report both populations."
    assert o["completed"] + o["rejected"] + o["cancelled"] == len(w), "HINT: Completion, rejection, and cancellation populations must sum to offered requests."

def test_thrashing_needs_aligned_failure_and_progress_evidence():
    churn_only = [{"t": 0., "dt": 100., "preempted": 2, "alloc_fail": 0, "progress": 100}]
    aligned = [{"t": 0., "dt": 100., "preempted": 2, "alloc_fail": 2, "progress": 1}]
    assert not k.detect_thrashing(churn_only, window_ms=1000, baseline_rate=1), H
    assert k.detect_thrashing(aligned, window_ms=1000, baseline_rate=1), H

def test_memory_pressure_creates_detectable_progress_collapse():
    cost = sim.LinearCost(base=10, per_decode=.1, per_prefill=.2)
    _, roomy = k.run_kv(workload(60), cost, k.Policy(n_blocks=2000, B=32))
    _, tight = k.run_kv(workload(60), cost, k.Policy(n_blocks=400, B=32))
    baseline = sum(x["progress"] for x in roomy) / sum(x["dt"] for x in roomy)
    tight_rate = sum(x["progress"] for x in tight) / sum(x["dt"] for x in tight)
    assert sum(x["preempted"] for x in tight) > 0 and tight_rate < .7 * baseline, H
    assert k.detect_thrashing(tight, window_ms=2000, baseline_rate=baseline, drop=.7), H

def test_readme_preemption_worked_example():
    mib, gib = 2**20, 2**30
    swap = k.swap_roundtrip_ms(500*mib, 25*gib)
    assert swap == pytest.approx(39.0625), "HINT: Convert MiB/GiB explicitly and include both transfer directions."
    assert 500*mib/(128*1024) == 4000, "HINT: Payload tokens = 500 MiB / 128 KiB per token."
    assert k.recompute_bound_ms(8e9, 4000, 500e12) == pytest.approx(128), "HINT: Dense recompute bound is 2*P*tokens/peak FLOP/s."
    assert k.breakeven_bw(500*mib, 310)/gib == pytest.approx(3.1502, rel=1e-4), "HINT: Break-even uses measured recompute time, not either lower bound."
