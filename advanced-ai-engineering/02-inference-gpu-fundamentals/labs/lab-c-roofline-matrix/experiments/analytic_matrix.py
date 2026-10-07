"""CPU experiment: analytical Roofline matrix (no measurement) for the shapes the
Colab notebook measures. Writes results/<stamp>-analytic/{bounds.jsonl,result.yaml}.

    python experiments/analytic_matrix.py [--ceilings exercise|t4_nameplate]
"""
import argparse
import sys
from pathlib import Path

import yaml

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "common"))
import labkit  # noqa: E402

labkit.use_impl(LAB)
import roofline as R  # noqa: E402


def load_ceilings(name):
    return yaml.safe_load((LAB / "fixtures" / "ceilings.yaml").read_text())["ceiling_sets"][name]


def workloads():
    """Shape matrix shared with colab.ipynb. Paths: fp16 -> tensor path (assumed), fp32 -> CUDA cores."""
    ws = []
    for dtype, path in (("fp16", "fp16_tensor"), ("fp32", "fp32_cuda")):
        ws.append(R.elementwise(1 << 24, dtype, path))
        ws.append(R.reduction(1 << 24, dtype, path))
        for n in (1, 8, 64, 512):
            ws.append(R.gemm(4096, n, 4096, dtype, path))
        ws.append(R.gemm(64, 64, 64, dtype, path))
    return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ceilings", default="exercise")
    args = ap.parse_args()
    ceil = load_ceilings(args.ceilings)
    rows = []
    for w in workloads():
        try:
            rows.append({**R.bound(w, ceil), "ceilings": args.ceilings})
        except ValueError as e:  # refused: no ceiling for this path/boundary in the set
            rows.append({"workload": w.name, "path": w.path, "refused": str(e), "ceilings": args.ceilings})
    out = labkit.new_run_dir(LAB / "results", f"analytic-{args.ceilings}")
    labkit.write_jsonl(out / "bounds.jsonl", rows)
    labkit.write_result(
        out / "result.yaml", rid=out.name,
        question=f"Which regime does the one-pass screen assign each shape under ceiling set '{args.ceilings}'?",
        env=labkit.manifest(), workload={"ceilings": ceil, "shapes": [w.name for w in workloads()]},
        execution={"kind": "analytical bound, CPU", "warmup_iterations": 0, "measurement_iterations": 1},
        raw_metrics={"jsonl": "bounds.jsonl"},
        aggregated_metrics={r["workload"]: r.get("screen", "refused") for r in rows},
        limitations=["Lower bounds under one-pass DRAM bytes and declared ceilings; not latency.",
                     f"Ceiling kind: {ceil.get('kind')}; source: {ceil.get('source')}"])
    for r in rows:
        if "refused" in r:
            print(f"{r['workload']:<28} REFUSED ({r['path']})")
        else:
            print(f"{r['workload']:<28} I={r['I']:9.2f} I*={r['I_star']:7.1f} T>={r['t_lower_s'] * 1e6:10.2f} us  {r['screen']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
