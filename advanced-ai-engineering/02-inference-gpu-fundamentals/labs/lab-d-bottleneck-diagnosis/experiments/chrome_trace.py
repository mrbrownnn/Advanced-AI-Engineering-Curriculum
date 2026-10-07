"""Convert a PyTorch-profiler Chrome trace (prof.export_chrome_trace) into the lab trace format.

Used by colab.ipynb so the same diagnose.py runs on real GPU timelines. Pure JSON,
no torch import. Field names below follow the Kineto Chrome-trace export as the
author understood it -- TODO_VERIFY against a real Colab export (category names,
"grid", "registers per thread"). Kernel counters that the timeline cannot provide
(spills, sectors, branch efficiency) stay None, i.e. 'unknown' to diagnose().
"""
import json

SYNC_NAMES = ("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize")


def convert(chrome, our_pid=None, sm_clock_mhz=None, sms=None):
    """Convert one process's export. Chrome `pid` differs across CPU/runtime/device lanes,
    so it is NOT a process identity here: every recognized event belongs to pid 1.
    `our_pid` is accepted for compatibility but ignored. A multi-process trace needs
    external process metadata and is deliberately unsupported (TODO_VERIFY)."""
    events = []
    raw = chrome["traceEvents"] if isinstance(chrome, dict) else chrome
    for e in raw:
        if e.get("ph") != "X":
            continue
        cat, name = e.get("cat", ""), e.get("name", "")
        start, end = float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0))
        if cat == "kernel":
            a = e.get("args", {})
            grid = a.get("grid")
            events.append({"kind": "kernel", "name": name, "pid": 1, "stream": a.get("stream"),
                           "start": start, "end": end,
                           "grid_blocks": (grid[0] * grid[1] * grid[2]) if isinstance(grid, list) else None,
                           "regs_per_thread": a.get("registers per thread"), "spill_bytes": None,
                           "sectors_per_request": None, "branch_efficiency": None, "sm_clock_mhz": sm_clock_mhz})
        elif cat == "cuda_runtime" and name in SYNC_NAMES:
            events.append({"kind": "sync", "name": name, "pid": 1, "start": start, "end": end})
    return {"meta": {"fault": "unknown", "pid": 1, "sms": sms,
                     "counters": False, "unit": "us", "synthetic": False}, "events": events}


def load(path, **kw):
    with open(path, encoding="utf-8") as f:
        return convert(json.load(f), **kw)
