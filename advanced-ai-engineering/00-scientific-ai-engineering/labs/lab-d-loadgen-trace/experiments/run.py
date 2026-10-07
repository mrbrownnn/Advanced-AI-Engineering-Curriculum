"""Append LAB D fixed-log policy rescoring as JSONL + YAML."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
MODULE = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true")
    args = ap.parse_args()
    root = MODULE.parent / "labs-solutions" / MODULE.name / HERE.name if args.solution else HERE / "starter"
    sys.path.insert(0, str(root))
    import boundary as b
    events = b.synthetic_log()
    scores = []
    for origin in ("scheduled", "actual_issue"):
        for population in ("success", "all"):
            for convention in ("nearest_rank", "loadgen_floor"):
                scores.append(b.rescore(events, origin, population, convention, 0.75, 2000))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = HERE / "results" / stamp
    out.mkdir(parents=True)
    raw = out / "raw.jsonl"
    raw.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in scores), encoding="utf-8")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=MODULE.parent, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    result = {"result": {"id": stamp, "date": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}",
        "question": "How do boundary, population, and quantile policy change a fixed SUT metric?",
        "environment": {"hardware": {"gpu": "none", "gpu_count": 0, "cpu": platform.processor() or "unknown",
            "memory_gb": "not recorded", "interconnect": "n/a"},
            "software": {"os": platform.platform(), "python": platform.python_version()}, "commit": commit},
        "workload": {"model": "fixed synthetic SUT log", "dataset": "fixtures/sut-events.jsonl",
                     "configuration": {"q": 0.75, "timeout_ms": 2000}},
        "execution": {"warmup_iterations": 0, "measurement_iterations": len(events), "total_duration_seconds": 0},
        "raw_metrics": {"path": raw.name, "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "events": events},
        "aggregated_metrics": {"policy_scores": scores},
        "confidence": {"method": "none; deterministic boundary audit", "confidence_level": None, "intervals": {}},
        "known_limitations": ["static trace not executed", "synthetic event log", "q=.75 chosen to expose index difference"],
        "notes": "pinned source commit 3fbc329939999c13d0a7b5e67fb2092287e06047"}}
    (out / "result.yaml").write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
