"""Append randomized and fixed-order LAB B simulations as JSONL + YAML."""
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
    ap.add_argument("--blocks", type=int, default=40)
    args = ap.parse_args()
    root = MODULE.parent / "labs-solutions" / MODULE.name / HERE.name if args.solution else HERE / "starter"
    sys.path.insert(0, str(root))
    import benchmark as b
    config = dict(seed=args.seed, n_blocks=args.blocks, effect_ms=-1.0,
                  drift_per_run_ms=1.5, noise_ms=0.1)
    rows, summaries = [], {}
    for design, randomized in (("randomized", True), ("fixed_AB", False)):
        trial = b.simulate(**config, randomized=randomized)
        rows.extend({"design": design, **x} for x in b.as_records(trial))
        mean, lo, hi = b.t_interval(b.paired_differences(trial))
        summaries[design] = {"n_blocks": args.blocks, "mean_B_minus_A_ms": mean,
                             "ci95_ms": [lo, hi]}
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
        "question": "Does randomized paired order resist monotonic drift better than fixed AB?",
        "environment": {"hardware": {"gpu": "none", "gpu_count": 0, "cpu": platform.processor() or "unknown",
            "memory_gb": "not recorded", "interconnect": "n/a"},
            "software": {"os": platform.platform(), "python": platform.python_version(),
                         "numpy": np.__version__, "scipy": scipy.__version__}, "commit": commit},
        "workload": {"model": "simulated process blocks", "dataset": "seeded synthetic", "configuration": config},
        "execution": {"warmup_iterations": 0, "measurement_iterations": len(rows), "total_duration_seconds": 0},
        "raw_metrics": {"path": raw.name, "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()},
        "aggregated_metrics": summaries,
        "confidence": {"method": "block-level paired t interval", "confidence_level": 0.95,
                       "intervals": {k: v["ci95_ms"] for k, v in summaries.items()}},
        "known_limitations": ["simulated linear drift", "one machine population not modeled"],
        "notes": "n is independent process block count, not run/request count"}}
    (out / "result.yaml").write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
