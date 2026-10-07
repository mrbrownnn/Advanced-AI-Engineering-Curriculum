"""Run LAB A against selected implementation; append JSONL + YAML results."""
from __future__ import annotations

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


def load_impl(solution: bool):
    root = MODULE.parent / "labs-solutions" / MODULE.name / HERE.name if solution else HERE / "starter"
    sys.path.insert(0, str(root))
    import harness
    return harness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true", help="use instructor reference")
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args()
    h = load_impl(args.solution)
    config = dict(seed=args.seed, period_s=0.05, duration_s=10.0, timeout_s=5.0,
                  stall_at_s=5.0, stall_s=2.0, base_s=0.010, jitter=0.3)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = HERE / "results" / stamp
    out.mkdir(parents=True)
    rows, summaries = [], {}
    for mode, gen in (("open", h.run_open_loop), ("closed", h.run_closed_loop)):
        op = h.NoisyOp(args.seed, base_s=config["base_s"], jitter=config["jitter"],
                       stall_at=config["stall_at_s"], stall_s=config["stall_s"])
        events = gen(op, h.VirtualClock(), config["period_s"], config["duration_s"], config["timeout_s"])
        rows.extend({"mode": mode, **vars(e)} for e in events)
        summaries[mode] = h.summarize(events, "intended", config["timeout_s"])
    raw = out / "events.jsonl"
    raw.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=MODULE.parent, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=MODULE.parent, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    result = {"result": {
        "id": stamp, "date": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}",
        "question": "Does response-coupled generation under-observe a 2 s stall?",
        "environment": {"hardware": {"gpu": "none", "gpu_count": 0,
            "cpu": platform.processor() or "unknown", "memory_gb": "not recorded", "interconnect": "n/a"},
            "software": {"os": platform.platform(), "python": platform.python_version(),
                         "numpy": __import__("numpy").__version__},
            "commit": commit, "dirty_tree": dirty},
        "workload": {"model": "NoisyOp", "dataset": "seeded synthetic", "configuration": config},
        "execution": {"warmup_iterations": 0, "measurement_iterations": len(rows),
                      "total_duration_seconds": config["duration_s"]},
        "raw_metrics": {"path": raw.name, "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()},
        "aggregated_metrics": summaries,
        "confidence": {"method": "none; deterministic demonstration", "confidence_level": None, "intervals": {}},
        "known_limitations": ["synthetic FIFO server", "single seed", "virtual time"],
        "notes": "nearest-rank quantiles; timeouts included as censored at timeout value"}}
    (out / "result.yaml").write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
