"""Colab-only GPU helpers (CUDA torch). Never imported by tests; never run on the local GPU.

Rule: refuse to run unless COLAB_GPU / COLAB_RELEASE_TAG is set or LAB_ALLOW_GPU=1,
so a learner cannot accidentally use a local GPU.
"""
import os
import shutil
import statistics
import subprocess


def require_colab_gpu():
    if not (os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") or os.environ.get("LAB_ALLOW_GPU") == "1"):
        raise SystemExit("GPU experiments run on Colab only (set LAB_ALLOW_GPU=1 to override deliberately).")
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device: Runtime > Change runtime type > T4 GPU.")
    return torch


def cuda_event_times(torch, fn, warmup=5, reps=30):
    """Per-call device intervals (seconds) bracketed by events on the current stream."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        out.append(s.elapsed_time(e) / 1e3)
    return out


def describe(samples):
    return {"n": len(samples), "median_s": statistics.median(samples), "min_s": min(samples),
            "max_s": max(samples), "estimator": "median of per-call CUDA-event intervals"}


def ncu_status():
    """Probe whether Nsight Compute can read device counters here. TODO_VERIFY on Colab:
    ERR_NVGPUCTRPERM (counter permission) is the expected failure on shared VMs."""
    exe = shutil.which("ncu")
    if not exe:
        return {"available": False, "reason": "ncu not on PATH"}
    probe = ("import torch; x=torch.ones(1<<20,device='cuda'); y=x*2; torch.cuda.synchronize()")
    try:
        r = subprocess.run([exe, "--metrics", "dram__bytes_read.sum", "--target-processes", "all",
                            "python", "-c", probe], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        return {"available": False, "reason": repr(e)}
    text = r.stdout + r.stderr
    ok = r.returncode == 0 and "dram__bytes_read.sum" in text and "ERR_NVGPUCTRPERM" not in text
    return {"available": ok, "returncode": r.returncode, "tail": text[-2000:]}
