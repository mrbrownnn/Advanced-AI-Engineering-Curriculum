"""CPU experiment: the four timing methods over the simulated device, swept over
workload size and stream count. Deterministic; illustrates the mechanism only.

    python experiments/sim_timing_sweep.py
"""
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "common"))
import labkit  # noqa: E402

labkit.use_impl(LAB)
import timing as M  # noqa: E402
from simdevice import SimClock, SimDevice  # noqa: E402


def workload(n, dur, streams):
    def work(d):
        for i in range(n):
            d.launch(dur, stream=i % streams)
    return work


def run():
    rows = []
    for n in (1, 10, 100):
        for dur in (5e-6, 1e-4, 1e-3):
            for streams in (1, 2):
                for method in ("unsynced", "synced", "events_stream0", "application", "application_sync_each_op"):
                    clock = SimClock()
                    dev = SimDevice(clock, first_launch_cost=0.0)
                    w = workload(n, dur, streams)
                    if method == "unsynced":
                        t = M.time_unsynced(dev, clock.now, w)
                    elif method == "synced":
                        t = M.time_synced(dev, clock.now, w)
                    elif method == "events_stream0":
                        t = M.time_events(dev, w, 0)
                    else:
                        t = M.time_application(dev, clock.now, lambda d: None, w,
                                               sync_each_op=method.endswith("each_op"))
                    rows.append({"n_kernels": n, "kernel_s": dur, "streams": streams, "method": method,
                                 "seconds": t, "device_busy_s": n * dur, "clock": "SimClock"})
    return rows


def main():
    rows = run()
    out = labkit.new_run_dir(LAB / "results", "sim")
    labkit.write_jsonl(out / "timings.jsonl", rows)
    ratio = {f"n{r['n_kernels']}_k{r['kernel_s']}_s{r['streams']}": round(r["seconds"] / r["device_busy_s"], 4)
             for r in rows if r["method"] == "unsynced"}
    labkit.write_result(
        out / "result.yaml", rid=out.name,
        question="In the simulated async device, how far does each timing boundary sit from device-busy time?",
        env=labkit.manifest(), workload={"simdevice": "starter/simdevice.py defaults"},
        execution={"kind": "deterministic simulation", "warmup_iterations": 0, "measurement_iterations": 1},
        raw_metrics={"jsonl": "timings.jsonl"},
        aggregated_metrics={"unsynced_over_device_busy": ratio},
        limitations=["Simulation of declared semantics; no real GPU, driver, or clock behavior.",
                     "Streams have no shared-resource contention in the model."])
    for r in rows:
        if r["n_kernels"] == 10 and r["kernel_s"] == 1e-3:
            print(f"streams={r['streams']} {r['method']:<26} {r['seconds'] * 1e3:8.3f} ms")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
