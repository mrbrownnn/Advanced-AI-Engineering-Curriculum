"""Lab C experiment: routing policies x prefix skew x load, repeated seeds.

Writes results/<stamp>-routing/{raw.jsonl,result.yaml}. raw.jsonl has one row per
request per run (seed, policy, workload) so quantiles can be recomputed.
    python experiments/run_routing.py
    LAB_IMPL=solution python experiments/run_routing.py
"""
import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "_shared"))
import labkit  # noqa: E402

labkit.use_impl(LAB, "lab-b-paged-block-manager")
import numpy as np  # noqa: E402
from router_sim import POLICIES, make_workload, simulate  # noqa: E402

SEEDS = range(5)
rows, agg = [], []
for hot_share in (0.1, 0.9):
    for rate in (30, 50):
        for name, policy in POLICIES.items():
            summaries = []
            for seed in SEEDS:
                out = simulate(make_workload(300, seed, hot_share=hot_share, rate_per_s=rate), policy)
                for r in out["rows"]:
                    rows.append({"seed": seed, "policy": name, "hot_share": hot_share, "rate_per_s": rate, **r})
                summaries.append(out["summary"])
            row = {"policy": name, "hot_share": hot_share, "rate_per_s": rate, "seeds": len(summaries)}
            for k in ("hit_rate", "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms", "slo_goodput_rps", "max_worker_share"):
                v = np.array([s[k] for s in summaries])
                row[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1))}
            agg.append(row)
            print(name, hot_share, rate, {k: round(row[k]["mean"], 3) for k in ("hit_rate", "ttft_p99_ms", "slo_goodput_rps")})

path = labkit.write_run(LAB, "routing", rows, {
    "question": "Does prefix affinity improve SLO-goodput, or only hit rate, as prefix skew and load change?",
    "workload": {"dataset": "router_sim.make_workload n=300", "configuration": "4 workers, 64 blocks x 16 tokens, SLO 300 ms"},
    "aggregated_metrics": {"by_condition": agg},
    "confidence": {"method": "mean and sample std over seeds", "seeds": list(SEEDS)},
    "known_limitations": ["prefill-only service model; no decode", "router sees exact cache state",
                          "quantiles use numpy linear interpolation"],
})
print(f"wrote {len(rows)} rows to {path}")
