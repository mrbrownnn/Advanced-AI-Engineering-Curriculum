"""Tiny helpers shared by Module 01 lab tests and experiments.

- select_impl(lab_dir): put starter/ (default) or the reference solution
  (LAB_IMPL=solution) on sys.path, plus the lab's fixtures/.
- write_result(lab_dir, name, rows, result): append-only results/<stamp>-<name>/
  with raw.jsonl and result.yaml (benchmarks/result-schema.md fields).
"""
import datetime
import json
import os
import pathlib
import platform
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # never touch a GPU

MODULE = "01-foundation-model-internals"


def select_impl(lab_dir):
    lab_dir = pathlib.Path(lab_dir).resolve()
    root = lab_dir.parents[2]
    if os.environ.get("LAB_IMPL") == "solution":
        impl = root / "labs-solutions" / MODULE / lab_dir.name
    else:
        impl = lab_dir / "starter"
    # impl first; starter last so provided-only files (bugged_*.py) resolve in both modes
    for p in (lab_dir / "starter", lab_dir / "fixtures", impl):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    if str(impl) in sys.path:
        sys.path.remove(str(impl))
    sys.path.insert(0, str(impl))
    return impl


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def write_result(lab_dir, name, rows, result):
    import torch
    import yaml

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = pathlib.Path(lab_dir) / "results" / f"{stamp}-{name}"
    out.mkdir(parents=True, exist_ok=False)  # append-only: never reuse a directory
    with open(out / "raw.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    doc = {
        "result": {
            "id": f"{MODULE}/{pathlib.Path(lab_dir).name}/{stamp}-{name}",
            "date": datetime.date.today().isoformat(),
            "environment": {
                "hardware": {"gpu": "none (CPU only)", "gpu_count": 0, "cpu": platform.processor() or platform.machine()},
                "software": {"os": platform.platform(), "python": platform.python_version(), "torch": str(torch.__version__),
                             "cuda": "not used", "torch_threads": torch.get_num_threads()},
                "commit": _git("rev-parse", "HEAD"),
                "dirty_tree": bool(_git("status", "--porcelain")),
                "impl": os.environ.get("LAB_IMPL", "starter"),
            },
            "raw_metrics": {"path": "raw.jsonl", "rows": len(rows)},
            **result,
        }
    }
    with open(out / "result.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    print(f"wrote {out}")
    return out
