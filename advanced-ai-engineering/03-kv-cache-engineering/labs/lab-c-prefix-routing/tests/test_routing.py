"""Lab C: cache-aware routing (Lesson 3.7). Worked-example numbers are the README's own."""
from router_sim import POLICIES, break_even_wait, make_workload, repeated, simulate, ttft_estimate

HSKEW = "HINT: If 99% of requests use Prefix X, and Prefix X is only on Worker 1, what happens to Worker 2?"
HOBJ = "HINT: hit rate is not the objective; compare SLO-goodput and tail TTFT."
SEEDS = range(3)
HOT = dict(n=200, hot_share=0.9, rate_per_s=50)


def test_lesson37_break_even_arithmetic():
    assert ttft_estimate(400, 30) == 430 and ttft_estimate(50, 150) == 200, "HINT: TTFT estimate = queue wait + prefill."
    assert break_even_wait(50, 150, 30) == 170, "HINT: route to the hit only while its wait < other wait + (miss - hit)."
    assert ttft_estimate(break_even_wait(50, 150, 30), 30) == ttft_estimate(50, 150), "HINT: break-even means equal TTFT."


def test_summary_quantiles_ordered_and_loads_sum():
    out = simulate(make_workload(150, 0, hot_share=0.5), POLICIES["round_robin"])
    s = out["summary"]
    assert s["ttft_p50_ms"] <= s["ttft_p95_ms"] <= s["ttft_p99_ms"], "HINT: quantiles must be monotone."
    assert sum(s["per_worker_load"]) == s["n"] == 150, "HINT: every request lands on exactly one worker."
    assert 0 <= s["hit_rate"] <= 1 and 0 <= s["slo_attainment"] <= 1, "HINT: rates are shares in [0, 1]."
    assert abs(s["max_worker_share"] - 0.25) < 0.02, "HINT: round-robin spreads load evenly."


def test_deterministic_given_seed():
    a = simulate(make_workload(100, 7, hot_share=0.5), POLICIES["load_aware"])["summary"]
    b = simulate(make_workload(100, 7, hot_share=0.5), POLICIES["load_aware"])["summary"]
    assert a == b, "HINT: seeded workload + simulated clock must give identical results."


def test_repeated_reports_variance():
    r = repeated("round_robin", SEEDS, **HOT)
    assert len(r["runs"]) == 3 and r["ttft_p99_ms"]["std"] >= 0, "HINT: report mean and sample std over seeds."
    assert len({round(x["ttft_p99_ms"], 6) for x in r["runs"]}) > 1, \
        "HINT: different seeds should give different runs; a zero-variance p99 suggests the seed is ignored."


def test_break_hot_prefix_affinity_raises_hits_but_lowers_goodput():
    rr = repeated("round_robin", SEEDS, **HOT)
    aff = repeated("prefix_affinity", SEEDS, **HOT)
    assert aff["hit_rate"]["mean"] > rr["hit_rate"]["mean"], f"affinity should raise hit rate. {HSKEW}"
    assert aff["max_worker_share"]["mean"] > 0.6, f"affinity should concentrate the hot prefix. {HSKEW}"
    assert aff["slo_goodput_rps"]["mean"] < rr["slo_goodput_rps"]["mean"], f"{HOBJ} {HSKEW}"


def test_load_aware_beats_pure_affinity_on_goodput_under_skew():
    aff = repeated("prefix_affinity", SEEDS, **HOT)
    la = repeated("load_aware", SEEDS, **HOT)
    assert la["slo_goodput_rps"]["mean"] > aff["slo_goodput_rps"]["mean"], HOBJ
    assert la["ttft_p99_ms"]["mean"] < aff["ttft_p99_ms"]["mean"], HOBJ


def test_affinity_helps_when_prefixes_are_diverse():
    wl = dict(n=200, hot_share=0.1, rate_per_s=50)
    rr = repeated("round_robin", SEEDS, **wl)
    aff = repeated("prefix_affinity", SEEDS, **wl)
    assert aff["hit_rate"]["mean"] > rr["hit_rate"]["mean"], \
        "HINT: with many prefixes and small caches, round-robin scatters each prefix across workers."
