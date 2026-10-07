"""COLAB ONLY: measured useful bandwidth for the fixture patterns, next to the sector model.

Each pattern copies n useful elements (read n, write n contiguous) out of a source
laid out by the pattern. Element count and arithmetic are identical across
patterns; only the source layout changes. Kernel = PyTorch's strided copy
(launch geometry chosen by PyTorch -- record it as a confounder).

    python experiments/gpu_bandwidth.py --out /content/drive/MyDrive/aie-labs/02/lab-a
"""
import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "common"))
sys.path.insert(0, str(LAB / "experiments"))
import gpu  # noqa: E402
import labkit  # noqa: E402
from sweep_patterns import load_patterns, model_rows  # noqa: E402  (also selects starter/solution)

DTYPES = {2: "float16", 4: "float32", 8: "float64"}


def source_view(torch, p, n):
    eb = p.get("elem_bytes") or p["field_bytes"]
    dt = getattr(torch, DTYPES[eb])
    if p["kind"] == "strided":
        s, o = p["stride_elems"], p["offset_elems"]
        buf = torch.randn(o + n * s, device="cuda").to(dt)
        return buf[o::s][:n]
    if p["kind"] == "aos":
        width = p["struct_bytes"] // p["field_bytes"]
        buf = torch.randn(n, width, device="cuda").to(dt)
        return buf[:, p["field_offset"] // p["field_bytes"]]
    return torch.randn(n, device="cuda").to(dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1 << 22)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    torch = gpu.require_colab_gpu()
    torch.manual_seed(0)
    model = {r["pattern"]: r for r in model_rows()}
    rows = []
    for p in load_patterns():
        src = source_view(torch, p, args.n)
        dst = torch.empty(args.n, dtype=src.dtype, device="cuda")
        dst.copy_(src)
        assert torch.equal(dst, src), "correctness oracle: copy must be exact"
        eb = src.element_size()
        useful = 2 * args.n * eb  # B_r + B_w
        times = gpu.cuda_event_times(torch, lambda: dst.copy_(src), reps=args.reps)
        for i, t in enumerate(times):
            rows.append({"pattern": p["name"], "rep": i, "seconds": t, "useful_bytes": useful,
                         "useful_GBps": useful / t / 1e9, "model_read_efficiency": model[p["name"]]["efficiency"],
                         "boundary": "useful bytes (not DRAM traffic)", "timer": "cuda.Event, current stream"})
    out = labkit.new_run_dir(args.out, "bandwidth")
    labkit.write_jsonl(out / "raw.jsonl", rows)
    agg = {}
    for p in load_patterns():
        ts = [r["seconds"] for r in rows if r["pattern"] == p["name"]]
        d = gpu.describe(ts)
        useful = next(r["useful_bytes"] for r in rows if r["pattern"] == p["name"])
        agg[p["name"]] = {**d, "useful_GBps_at_median": useful / d["median_s"] / 1e9}
    ncu = gpu.ncu_status()
    labkit.write_result(
        out / "result.yaml", rid=out.name,
        question="Does strided/misaligned/AoS layout reduce useful bandwidth relative to aligned contiguous access on T4?",
        env={**labkit.manifest(seed=0, extra_packages=("torch",)), "ncu": ncu},
        workload={"n_elements": args.n, "patterns": [p["name"] for p in load_patterns()], "op": "Tensor.copy_"},
        execution={"warmup_iterations": 5, "measurement_iterations": args.reps},
        raw_metrics={"jsonl": "raw.jsonl"}, aggregated_metrics=agg,
        limitations=["Useful bandwidth only; hierarchy traffic needs ncu (see environment.ncu.available).",
                     "PyTorch chooses launch geometry and vectorization per layout (confounder).",
                     "Write side is always contiguous; only the read layout varies."])
    print(f"wrote {out}; ncu available: {ncu['available']}")


if __name__ == "__main__":
    main()
