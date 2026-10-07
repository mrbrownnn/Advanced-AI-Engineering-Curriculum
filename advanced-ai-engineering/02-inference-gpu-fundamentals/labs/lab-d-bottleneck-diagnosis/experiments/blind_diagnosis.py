"""CPU experiment: blind diagnosis of every planted fault, with and without kernel counters.

Writes results/<stamp>-blind/{diagnoses.jsonl,result.yaml}. The fault label is
only joined AFTER diagnosis, so the diagnostician code never sees it.

    python experiments/blind_diagnosis.py [--seeds 5]
"""
import argparse
import sys
from pathlib import Path

import yaml

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "common"))
sys.path.insert(0, str(LAB / "fixtures"))
import labkit  # noqa: E402

labkit.use_impl(LAB)
import diagnose as D  # noqa: E402
from gen_traces import FAULTS, make_trace  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    hyp = yaml.safe_load((LAB / "fixtures" / "signatures.yaml").read_text())["hypotheses"]
    rows = []
    for counters in (True, False):
        base = make_trace(None, seed=1000, counters=counters)
        for seed in range(args.seeds):
            for fault in FAULTS:
                new = make_trace(fault, seed=seed, counters=counters)
                new_blind = {"meta": {**new["meta"], "fault": "hidden"}, "events": new["events"]}
                d = D.diagnose(hyp, D.compare(base, new_blind))
                rows.append({"counters": counters, "seed": seed, "planted": fault, "top": d["top"],
                             "unresolved": d["unresolved"], "ruled_down": len(d["ruled_down"]),
                             "util_base": D.utilization(base), "util_new": D.utilization(new),
                             "false_attribution": d["top"] is not None and d["top"] != fault})
    out = labkit.new_run_dir(LAB / "results", "blind")
    labkit.write_jsonl(out / "diagnoses.jsonl", rows)
    agg = {}
    for counters in (True, False):
        sub = [r for r in rows if r["counters"] == counters]
        key = "with_counters" if counters else "timeline_only"
        agg[key] = {"correct": sum(r["top"] == r["planted"] for r in sub),
                    "abstained": sum(r["top"] is None for r in sub),
                    "false_attribution": sum(r["false_attribution"] for r in sub), "n": len(sub)}
    labkit.write_result(
        out / "result.yaml", rid=out.name,
        question="Does the hypothesis matrix identify each planted fault, and where must it abstain without counters?",
        env=labkit.manifest(), workload={"faults": list(FAULTS), "seeds": args.seeds},
        execution={"kind": "synthetic traces, CPU", "warmup_iterations": 0, "measurement_iterations": 1},
        raw_metrics={"jsonl": "diagnoses.jsonl"}, aggregated_metrics=agg,
        limitations=["Synthetic traces with planted, single, non-interacting faults.",
                     "Signature matrix is an engineering hypothesis; real faults interact."])
    print(yaml.safe_dump(agg, sort_keys=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
