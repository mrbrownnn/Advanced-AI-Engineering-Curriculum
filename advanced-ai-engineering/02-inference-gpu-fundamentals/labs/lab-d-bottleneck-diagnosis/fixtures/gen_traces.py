"""Seeded synthetic timelines with ONE planted cause each (plus a baseline).

Trace = {"meta": {...}, "events": [...]}; times in microseconds on one clock.
kernel event: kind, name, pid, stream, start, end, grid_blocks, regs_per_thread,
              spill_bytes, sectors_per_request, branch_efficiency, sm_clock_mhz
sync event:   kind="sync", pid, start, end
With `counters=False` (Nsight Compute unavailable) the KERNEL_COUNTERS are None;
grid size (launch config, timeline) and SM clock (nvidia-smi) stay available.

The faults are SYNTHETIC: magnitudes are chosen to be plausible and to make two
pairs (launch_gap vs synchronization, strided_loads vs register_pressure) look
alike in elapsed time and utilization. They are not measurements of any GPU.

    python fixtures/gen_traces.py   # writes fixtures/traces/*.json for inspection
"""
import json
from pathlib import Path

import numpy as np

FAULTS = ("host_launch_gap", "synchronization", "small_grid", "strided_loads",
          "register_pressure", "divergence", "background_activity", "clock_throttle")
KERNELS = ("load", "gemm", "act", "store")
KERNEL_COUNTERS = ("regs_per_thread", "spill_bytes", "sectors_per_request", "branch_efficiency")  # Nsight Compute only
COUNTER_KEYS = ("grid_blocks", "regs_per_thread", "spill_bytes", "sectors_per_request",
                "branch_efficiency", "sm_clock_mhz")


def make_trace(fault=None, seed=0, iters=10, counters=True, sms=40):
    if fault is not None and fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}")
    rng = np.random.default_rng(seed)
    t = 0.0
    events = []
    for _ in range(iters):
        for name in KERNELS:
            gap = 10.0
            dur = 100.0
            k = {"grid_blocks": 160, "regs_per_thread": 64, "spill_bytes": 0, "sectors_per_request": 4.0,
                 "branch_efficiency": 1.0, "sm_clock_mhz": 1590}
            if fault == "host_launch_gap":
                gap += 190.0
            if fault == "small_grid" and name == "gemm":
                k["grid_blocks"], dur = 8, dur * 3.0
            if fault == "strided_loads" and name == "load":
                k["sectors_per_request"], dur = 16.0, dur * 4.0
            if fault == "register_pressure" and name == "load":
                k["regs_per_thread"], k["spill_bytes"], dur = 128, 4096, dur * 4.0
            if fault == "divergence" and name == "act":
                k["branch_efficiency"], dur = 0.45, dur * 2.5
            if fault == "clock_throttle":
                k["sm_clock_mhz"], dur = 900, dur * 1590 / 900
            if fault == "background_activity":
                dur *= 1.35
            t += gap * (1 + 0.01 * rng.standard_normal())
            dur *= 1 + 0.01 * rng.standard_normal()
            ev = {"kind": "kernel", "name": name, "pid": 1, "stream": 0, "start": t, "end": t + dur}
            ev.update(k if counters else {**k, **{key: None for key in KERNEL_COUNTERS}})
            events.append(ev)
            if fault == "background_activity":
                events.append({"kind": "kernel", "name": "foreign", "pid": 2, "stream": 0, "start": t,
                               "end": t + 0.8 * dur, **{key: None for key in COUNTER_KEYS}})
            t += dur
            if fault == "synchronization":
                events.append({"kind": "sync", "name": "cudaDeviceSynchronize", "pid": 1, "start": t, "end": t + 185.0})
                t += 180.0
    return {"meta": {"fault": fault or "baseline", "seed": seed, "sms": sms, "pid": 1, "counters": counters,
                     "unit": "us", "synthetic": True}, "events": events}


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "traces"
    out.mkdir(exist_ok=True)
    for f in (None, *FAULTS):
        (out / f"{f or 'baseline'}.json").write_text(json.dumps(make_trace(f), indent=1))
    print(f"wrote {out}")
