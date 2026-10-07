"""COLAB ONLY: the same timing.py functions against a real GPU, plus torch.utils.benchmark.

Methods per sample: unsynced host, synced host, CUDA events (stream 0),
application (incl. allocation setup), application with per-op sync, and the
blocked_autorange utility. Then the second-stream case.

    python experiments/gpu_timing.py --out /content/drive/MyDrive/aie-labs/02/lab-b
"""
import argparse
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "common"))
sys.path.insert(0, str(LAB / "experiments"))
import gpu  # noqa: E402
import labkit  # noqa: E402

labkit.use_impl(LAB)
import timing as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--ops", type=int, default=10)
    ap.add_argument("--samples", type=int, default=30)
    args = ap.parse_args()
    torch = gpu.require_colab_gpu()
    from torch_device import TorchDevice
    import torch.utils.benchmark as tb

    torch.manual_seed(0)
    dev = TorchDevice(n_streams=2)
    a = torch.randn(args.size, args.size, device="cuda")
    b = torch.randn(args.size, args.size, device="cuda")
    side = torch.randn(args.size, args.size, device="cuda")
    torch.cuda.synchronize()  # inputs ready before any timing (stream 1 must not race setup)
    clock = time.perf_counter

    def work(d):
        for _ in range(args.ops):
            d.launch(lambda: a @ b)

    def work_two_streams(d):
        for _ in range(args.ops):
            d.launch(lambda: a @ b, stream=0)
            d.launch(lambda: side @ side, stream=1)

    def setup(d):
        d.launch(lambda: torch.empty(args.size, args.size, device="cuda"))

    rows = []
    for i in range(args.samples):  # sample 0 is cold for this process; summarize() keeps it separate
        for name, fn in (("unsynced", lambda: M.time_unsynced(dev, clock, work)),
                         ("synced", lambda: M.time_synced(dev, clock, work)),
                         ("events_stream0", lambda: M.time_events(dev, work)),
                         ("application", lambda: M.time_application(dev, clock, setup, work)),
                         ("application_sync_each_op", lambda: M.time_application(dev, clock, setup, work, True)),
                         ("events_stream0_two_streams", lambda: M.time_events(dev, work_two_streams)),
                         ("application_two_streams", lambda: M.time_application(dev, clock, setup, work_two_streams)),
                         ("application_two_streams_sync_each_op",
                          lambda: M.time_application(dev, clock, setup, work_two_streams, True))):
            dev.synchronize()
            rows.append({"sample": i, "method": name, "seconds": fn(), "ops": args.ops, "size": args.size})
    m = tb.Timer(stmt="for _ in range(ops): a @ b", globals={"a": a, "b": b, "ops": args.ops}).blocked_autorange()
    for t in m.raw_times:
        rows.append({"sample": None, "method": "torch.utils.benchmark.blocked_autorange", "ops": args.ops,
                     "size": args.size, "seconds": t / m.number_per_run, "number_per_run": m.number_per_run})
    out = labkit.new_run_dir(args.out, "timing")
    labkit.write_jsonl(out / "raw.jsonl", rows)
    agg = {}
    for name in dict.fromkeys(r["method"] for r in rows):
        s = [r["seconds"] for r in rows if r["method"] == name]
        agg[name] = M.summarize(s, warmup=1 if len(s) > 1 and name != "torch.utils.benchmark.blocked_autorange" else 0)
        agg[name] = {k: v for k, v in agg[name].items() if k != "warm"}
    labkit.write_result(
        out / "result.yaml", rid=out.name,
        question="Which timing boundary includes queued GPU work, and which destroys two-stream overlap?",
        env=labkit.manifest(seed=0, extra_packages=("torch",)),
        workload={"op": f"{args.ops} x fp32 matmul {args.size}^2", "streams": [1, 2]},
        execution={"warmup_iterations": 1, "measurement_iterations": args.samples},
        raw_metrics={"jsonl": "raw.jsonl"}, aggregated_metrics=agg,
        limitations=["One process, one GPU, no profiler attached.",
                     "blocked_autorange rows are per-iteration-of-stmt averages over number_per_run, not samples.",
                     "fp32 matmul may use TF32 depending on torch defaults; recorded torch version matters."])
    for k, v in agg.items():
        print(f"{k:<42} median {v['warm_median'] * 1e3:9.3f} ms")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
