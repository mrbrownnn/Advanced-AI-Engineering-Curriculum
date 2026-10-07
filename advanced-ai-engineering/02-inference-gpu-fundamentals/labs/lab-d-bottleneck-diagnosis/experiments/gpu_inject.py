"""COLAB ONLY: inject faults into a 4-op loop on a real GPU, capture an unprofiled baseline
and a torch.profiler timeline per fault, convert it, and run the same diagnose.py.

Injected here (PyTorch-expressible): host_launch_gap, synchronization, small_grid,
strided_loads. NOT injected (need a custom kernel or are unsafe/unavailable on
Colab): divergence, register_pressure (write a Triton/CuPy kernel -- TODO),
background_activity (second process -- TODO), clock_throttle (no clock control
on Colab; never attempt). Those stay synthetic-only (fixtures/gen_traces.py).

    python experiments/gpu_inject.py --out /content/drive/MyDrive/aie-labs/02/lab-d
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "common"))
sys.path.insert(0, str(LAB / "experiments"))
import chrome_trace  # noqa: E402
import gpu  # noqa: E402
import labkit  # noqa: E402

labkit.use_impl(LAB)
import diagnose as D  # noqa: E402

INJECTED = ("host_launch_gap", "synchronization", "small_grid", "strided_loads")


def make_loop(torch, fault, n=1 << 22, iters=10):
    x = torch.randn(n * 8, device="cuda")
    a = torch.randn(1024, 1024, device="cuda")
    tiny = torch.randn(64, 64, device="cuda")
    out = torch.empty(n, device="cuda")

    def loop():
        for _ in range(iters):
            src = x[::8] if fault == "strided_loads" else x[:n]
            out.copy_(src)                                       # load
            if fault == "host_launch_gap":
                time.sleep(200e-6)
            if fault == "small_grid":
                # Many tiny launches keep the phase slow while every grid underfills the T4.
                # Total FLOPs differ from baseline: report shape/backend as a confounder.
                for _ in range(512):
                    tiny @ tiny
            else:
                a @ a                                             # gemm
            torch.relu_(out)                                     # act
            if fault == "synchronization":
                torch.cuda.synchronize()
            out.mul_(0.5)                                        # store-ish
    return loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    torch = gpu.require_colab_gpu()
    from torch.profiler import ProfilerActivity, profile

    torch.manual_seed(0)
    hyp = yaml.safe_load((LAB / "fixtures" / "signatures.yaml").read_text())["hypotheses"]
    out = labkit.new_run_dir(args.out, "inject")
    rows, traces = [], {}
    for fault in (None, *INJECTED):
        loop = make_loop(torch, fault)
        times = gpu.cuda_event_times(torch, loop, reps=args.reps)  # unprofiled baseline first
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            loop()
            torch.cuda.synchronize()
        path = out / f"trace-{fault or 'baseline'}.json"
        prof.export_chrome_trace(str(path))
        raw = json.loads(path.read_text())
        traces[fault] = chrome_trace.convert(raw)
        rows.append({"fault": fault or "baseline", "unprofiled_s": times, "trace": path.name})
    base = traces[None]
    diag = {}
    for fault in INJECTED:
        d = D.diagnose(hyp, D.compare(base, traces[fault]))
        diag[fault] = {"top": d["top"], "unresolved": d["unresolved"], "ruled_down": d["ruled_down"],
                       "attribution_us": D.attribute_regression(base, traces[fault])}
    labkit.write_jsonl(out / "raw.jsonl", rows)
    labkit.write_result(
        out / "result.yaml", rid=out.name,
        question="With timeline-only evidence on a real T4, which injected faults can be discriminated?",
        env={**labkit.manifest(seed=0, extra_packages=("torch",)), "ncu": gpu.ncu_status()},
        workload={"injected": list(INJECTED), "not_injected": ["divergence", "register_pressure",
                                                                "background_activity", "clock_throttle"]},
        execution={"warmup_iterations": 5, "measurement_iterations": args.reps,
                   "profiler": "torch.profiler CPU+CUDA, one loop per fault, after unprofiled timing"},
        raw_metrics={"jsonl": "raw.jsonl", "traces": [r["trace"] for r in rows]},
        aggregated_metrics={"diagnoses": diag},
        limitations=["Profiled runs are for structure only; report unprofiled_s for effects.",
                     "chrome_trace.convert field names are TODO_VERIFY against this export.",
                     "Kernel counters absent unless ncu works here; counter features are 'unknown'."])
    print(yaml.safe_dump(diag, sort_keys=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
