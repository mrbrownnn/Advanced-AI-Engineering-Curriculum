"""CPU experiment: modeled useful vs moved bytes for every fixture pattern (no GPU).

Writes results/<stamp>-model/{traffic.jsonl,result.yaml}. The Colab notebook
reuses `model_rows` to put the model next to measured bandwidth.

    python experiments/sweep_patterns.py [--sector-bytes 32]
"""
import argparse
import sys
from pathlib import Path

import yaml

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "common"))
import labkit  # noqa: E402

labkit.use_impl(LAB)
import transactions as T  # noqa: E402


def load_patterns():
    return yaml.safe_load((LAB / "fixtures" / "patterns.yaml").read_text())["patterns"]


def addresses_for(p, n):
    if p["kind"] == "strided":
        return T.strided_addresses(n, p["elem_bytes"], p["stride_elems"], p["offset_elems"])
    if p["kind"] == "aos":
        return T.aos_addresses(n, p["struct_bytes"], p["field_offset"], p["field_bytes"])
    return T.soa_addresses(n, p["field_bytes"])


def model_rows(sector_bytes=32, n_lanes=32 * 64):
    rows = []
    for p in load_patterns():
        access = p.get("elem_bytes") or p["field_bytes"]
        t = T.pattern_traffic(addresses_for(p, n_lanes), access, sector_bytes)
        rows.append({"pattern": p["name"], "sector_bytes": sector_bytes, "n_lanes": n_lanes,
                     "boundary": "request-level sectors (model, no cache reuse)", **t,
                     "moved_over_useful": t["moved_bytes"] / t["useful_bytes"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector-bytes", type=int, default=32)
    args = ap.parse_args()
    rows = model_rows(args.sector_bytes)
    out = labkit.new_run_dir(LAB / "results", "model")
    labkit.write_jsonl(out / "traffic.jsonl", rows)
    labkit.write_result(
        out / "result.yaml", rid=out.name,
        question="Under the declared sector model, how much traffic does each layout move per useful byte?",
        env=labkit.manifest(), workload={"patterns": str(LAB / "fixtures" / "patterns.yaml"),
                                         "sector_bytes": args.sector_bytes},
        execution={"kind": "analytical model, CPU", "warmup_iterations": 0, "measurement_iterations": 1},
        raw_metrics={"jsonl": "traffic.jsonl"},
        aggregated_metrics={r["pattern"]: round(r["moved_over_useful"], 4) for r in rows},
        limitations=["Model, not measurement: no cache reuse, no L2/DRAM distinction, sector size is a parameter.",
                     "Says nothing about time; pair with colab.ipynb measurements."])
    for r in rows:
        print(f"{r['pattern']:<20} useful={r['useful_bytes']:>7} moved={r['moved_bytes']:>7} eff={r['efficiency']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
