"""Shared helpers for Module 02 lab experiments: implementation selection, manifests,
append-only result directories, raw JSONL, result.yaml (benchmarks/result-schema.md).

Used by every lab's experiments/ scripts and colab.ipynb. Not graded.
"""
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import yaml

LABS_DIR = Path(__file__).resolve().parents[1]
MODULE = LABS_DIR.parent.name
REPO = LABS_DIR.parents[1]


def use_impl(lab_dir):
    """Put starter/ (default) or the reference (LAB_IMPL=solution) on sys.path, like tests/conftest.py."""
    lab_dir = Path(lab_dir).resolve()
    if os.environ.get("LAB_IMPL") == "solution":
        path = REPO / "labs-solutions" / MODULE / lab_dir.name
    else:
        path = lab_dir / "starter"
    sys.path.insert(0, str(path))
    return path


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def manifest(seed=None, extra_packages=()):
    """Hardware/software manifest. GPU fields stay None unless nvidia-smi exists (Colab)."""
    pkgs = {}
    for name in ("numpy", "scipy", "yaml", *extra_packages):
        try:
            pkgs[name] = getattr(__import__(name), "__version__", "unknown")
        except ImportError:
            pkgs[name] = None
    smi = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem",
                "--format=csv,noheader"])
    return {
        "hardware": {"gpu": smi, "gpu_count": len(smi.splitlines()) if smi else 0,
                     "cpu": platform.processor() or platform.machine()},
        "software": {"os": platform.platform(), "python": platform.python_version(), **pkgs,
                     "nvidia_smi_full": _run(["nvidia-smi"])},
        "commit": _run(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "dirty_tree": bool(_run(["git", "-C", str(REPO), "status", "--porcelain"])),
        "seed": seed,
    }


def new_run_dir(results_root, tag):
    """Create results_root/<UTC timestamp>-<tag>/. Never reuses an existing directory (append-only)."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(results_root) / f"{stamp}-{tag}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_jsonl(path, rows):
    with open(path, "x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_result(path, *, rid, question, env, workload, execution, raw_metrics,
                 aggregated_metrics, limitations, notes="", confidence=None):
    doc = {"result": {
        "id": rid, "date": _dt.date.today().isoformat(), "question": question,
        "environment": dict(env),  # manifest() keys plus any extras (e.g. ncu probe status)
        "workload": workload, "execution": execution, "raw_metrics": raw_metrics,
        "aggregated_metrics": aggregated_metrics,
        "confidence": confidence or {"method": "none (single run; descriptive only)"},
        "known_limitations": list(limitations), "notes": notes,
    }}
    with open(path, "x", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
