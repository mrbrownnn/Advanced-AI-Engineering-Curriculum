"""COLAB ONLY: measured time, numerical error, and sustained ceilings for the analytic matrix.

1. Probes sustained ceilings: large contiguous copy (beta at useful-byte level)
   and large fp16/fp32 GEMM (P on the path PyTorch selected).
2. Times every workload from analytic_matrix.workloads() with CUDA events.
3. Writes the bound under t4_nameplate AND under the measured set, qualify() flags,
   and classification_flips between them.

    python experiments/gpu_roofline.py --out /content/drive/MyDrive/aie-labs/02/lab-c
"""
import argparse
import re
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "common"))
sys.path.insert(0, str(LAB / "experiments"))
import gpu  # noqa: E402
import labkit  # noqa: E402
from analytic_matrix import R, load_ceilings, workloads  # noqa: E402  (R = selected roofline impl)


def run_op(torch, w):
    """Build inputs and a callable for workload w; return (fn, error_vs_fp64)."""
    dt = torch.float16 if w.name.endswith("fp16") else torch.float32
    nums = [int(x) for x in re.findall(r"\d+", w.name.split("_", 1)[1].rsplit("_", 1)[0])]
    if w.name.startswith("elementwise"):
        x, y = (torch.randn(nums[0], device="cuda", dtype=dt) for _ in range(2))
        out = torch.empty_like(x)
        fn = lambda: torch.add(x, y, out=out)  # noqa: E731
        ref = x.double() + y.double()
    elif w.name.startswith("reduction"):
        x = torch.randn(nums[0], device="cuda", dtype=dt)
        fn = lambda: x.sum()  # noqa: E731
        out, ref = None, x.double().sum()
    else:
        m, n, k = nums
        a, b = torch.randn(m, k, device="cuda", dtype=dt), torch.randn(k, n, device="cuda", dtype=dt)
        out = torch.empty(m, n, device="cuda", dtype=dt)
        fn = lambda: torch.matmul(a, b, out=out)  # noqa: E731
        ref = a.double() @ b.double()
    got = fn() if out is None else (fn(), out)[1]
    err = ((got.double() - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()
    return fn, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    torch = gpu.require_colab_gpu()
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    rows = []

    # sustained ceilings (useful-byte bandwidth; achieved GEMM rate on whatever path cuBLAS picked)
    n = 1 << 27
    src, dst = torch.empty(n, dtype=torch.uint8, device="cuda"), torch.empty(n, dtype=torch.uint8, device="cuda")
    beta = 2 * n / gpu.describe(gpu.cuda_event_times(torch, lambda: dst.copy_(src), reps=args.reps))["median_s"]
    peaks = {}
    for dt, path in ((torch.float16, "fp16_tensor"), (torch.float32, "fp32_cuda")):
        a = torch.randn(4096, 4096, device="cuda", dtype=dt)
        t = gpu.describe(gpu.cuda_event_times(torch, lambda: a @ a, reps=args.reps))["median_s"]
        peaks[path] = 2 * 4096 ** 3 / t
    measured = {"kind": "measured", "compute": peaks, "bandwidth": {"dram": beta}, "sms": props.multi_processor_count,
                "source": "this run: 128 MiB uint8 copy; 4096^3 GEMM per dtype"}
    nameplate = load_ceilings("t4_nameplate")

    for w in workloads():
        fn, err = run_op(torch, w)
        times = gpu.cuda_event_times(torch, fn, reps=args.reps)
        med = gpu.describe(times)["median_s"]
        for label, ceil in (("t4_nameplate", nameplate), ("measured", measured)):
            q = R.qualify(R.bound(w, ceil), med, sms=props.multi_processor_count)
            rows.append({**q, "ceilings": label, "raw_seconds": times, "rel_max_err_vs_fp64": err,
                         "achieved_flops": w.flops / med, "achieved_useful_Bps": w.bytes / med,
                         "grid_blocks": "TODO: from profiler trace (torch.profiler kernel args.grid)"})
    out = labkit.new_run_dir(args.out, "roofline")
    labkit.write_jsonl(out / "raw.jsonl", rows)
    flips = R.classification_flips(workloads(), nameplate, measured)
    labkit.write_result(
        out / "result.yaml", rid=out.name,
        question="Does any shape change regime when measured ceilings replace T4 nameplate values?",
        env={**labkit.manifest(seed=0, extra_packages=("torch",)), "ncu": gpu.ncu_status(),
             "tf32_matmul": torch.backends.cuda.matmul.allow_tf32},
        workload={"shapes": [w.name for w in workloads()], "measured_ceilings": measured},
        execution={"warmup_iterations": 5, "measurement_iterations": args.reps},
        raw_metrics={"jsonl": "raw.jsonl"},
        aggregated_metrics={"classification_flips_nameplate_to_measured": flips,
                            "median_s": {r["workload"]: sorted(r["raw_seconds"])[len(r["raw_seconds"]) // 2]
                                         for r in rows if r["ceilings"] == "measured"}},
        limitations=["Q is algorithmic one-pass bytes; measured DRAM traffic needs ncu (environment.ncu).",
                     "Measured 'peak' is the achieved large-GEMM rate on cuBLAS's chosen path, not a hardware ceiling.",
                     "Instruction path (tensor core or not) is ASSUMED from dtype; verify with a profiler kernel name.",
                     "fp32 GEMM may run TF32 if allowed (environment.tf32_matmul)."])
    print(f"flips: {flips}\nwrote {out}")


if __name__ == "__main__":
    main()
