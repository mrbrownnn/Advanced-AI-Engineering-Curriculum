"""Append LAB C clustered uncertainty audit as JSONL + YAML."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import yaml

HERE = Path(__file__).resolve().parents[1]
MODULE = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true")
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args()
    root = MODULE.parent / "labs-solutions" / MODULE.name / HERE.name if args.solution else HERE / "starter"
    sys.path.insert(0, str(root))
    import audit as a
    designs = [("few_processes", 5, 200), ("many_processes", 20, 50)]
    rows, summary = [], {}
    for name, n_processes, requests in designs:
        data = a.simulate_clusters(args.seed, n_processes, requests)
        rows.extend({"design": name, **r} for r in data)
        summary[name] = {"n_processes": n_processes, "requests_per_process": requests,
                         "naive_request_t95": list(a.naive_request_interval(data)),
                         "process_t95": list(a.process_interval(data)),
                         "cluster_bootstrap95": list(a.cluster_bootstrap_interval(data, args.seed, 2000))}
    summary["tail_exposure"] = {"q": 0.01, "target": 0.95,
        "minimum_n": a.trials_for_exposure(0.01, 0.95), "exposure": a.tail_exposure(0.01, 299)}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = HERE / "results" / stamp
    out.mkdir(parents=True)
    raw = out / "raw.jsonl"
    raw.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=MODULE.parent, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    result = {"result": {"id": stamp, "date": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}",
        "question": "Does adding correlated requests overstate process-level precision?",
        "environment": {"hardware": {"gpu": "none", "gpu_count": 0, "cpu": platform.processor() or "unknown",
            "memory_gb": "not recorded", "interconnect": "n/a"},
            "software": {"os": platform.platform(), "python": platform.python_version(),
                         "numpy": np.__version__, "scipy": scipy.__version__}, "commit": commit},
        "workload": {"model": "random-intercept process clusters", "dataset": "seeded synthetic",
                     "configuration": {"seed": args.seed, "designs": designs}},
        "execution": {"warmup_iterations": 0, "measurement_iterations": len(rows), "total_duration_seconds": 0},
        "raw_metrics": {"path": raw.name, "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()},
        "aggregated_metrics": summary,
        "confidence": {"method": "request t, process t, whole-process percentile bootstrap",
                       "confidence_level": 0.95, "intervals": summary},
        "known_limitations": ["synthetic random-intercept model", "tail events assumed independent"],
        "notes": "299 is exposure only; it is not a precise P99 interval"}}
    (out / "result.yaml").write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
