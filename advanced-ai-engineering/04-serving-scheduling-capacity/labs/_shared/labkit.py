"""Shared plumbing for Module 04 labs: implementation selection and result writing.

Used by every lab's tests/conftest.py and experiments/*.py. Not part of any exercise.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import platform
import subprocess
import sys

import yaml

MODULE = "04-serving-scheduling-capacity"


def impl_dir(lab_dir: pathlib.Path, lab_name: str | None = None) -> pathlib.Path:
    """starter/ of the lab, or labs-solutions/<module>/<lab>/ when LAB_IMPL=solution."""
    lab_dir = pathlib.Path(lab_dir)
    name = lab_name or lab_dir.name
    if os.environ.get("LAB_IMPL", "starter") == "solution":
        root = lab_dir.parents[2]  # advanced-ai-engineering/
        return root / "labs-solutions" / MODULE / name
    return lab_dir.parent / name / "starter"


def use_impl(lab_dir: pathlib.Path, *depends_on: str) -> None:
    """Put the selected implementation (and reused earlier labs) on sys.path."""
    lab_dir = pathlib.Path(lab_dir)
    for name in (lab_dir.name, *depends_on):
        p = str(impl_dir(lab_dir, name))
        if p not in sys.path:
            sys.path.insert(0, p)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def write_run(lab_dir: pathlib.Path, name: str, rows: list[dict], result: dict) -> pathlib.Path:
    """Append-only: results/<UTC stamp>-<name>/raw.jsonl + result.yaml. Never overwrites."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = pathlib.Path(lab_dir) / "results"
    out = base / f"{stamp}-{name}"
    i = 1
    while out.exists():
        out = base / f"{stamp}-{name}-{i}"
        i += 1
    out.mkdir(parents=True)
    with open(out / "raw.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    doc = {
        "id": out.name,
        "date": _dt.date.today().isoformat(),
        "environment": {
            "hardware": {"gpu": None, "gpu_count": 0, "cpu": platform.processor() or platform.machine()},
            "software": {"os": platform.platform(), "python": platform.python_version()},
            "commit": _git_commit(),
            "lab_impl": os.environ.get("LAB_IMPL", "starter"),
        },
        "raw_data": "raw.jsonl",
    }
    doc.update(result)
    with open(out / "result.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"result": doc}, f, sort_keys=False)
    return out
