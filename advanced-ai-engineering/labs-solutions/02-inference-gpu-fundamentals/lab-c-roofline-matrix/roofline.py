"""LAB C reference: qualified Roofline screen.

Conventions (D): FMA = 2 FLOPs; one-pass algorithmic bytes = each input read
once, each output written once, at the boundary named in `Workload.boundary`
(default "dram"); beta*C = 0 for GEMM. Units: FLOPs, bytes, seconds.
"""
from dataclasses import dataclass

DTYPE_BYTES = {"fp64": 8, "fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}


@dataclass(frozen=True)
class Workload:
    name: str
    flops: float
    bytes: float
    path: str               # instruction path the FLOPs execute on, e.g. "fp16_tensor"
    boundary: str = "dram"  # memory level the bytes were counted at
    blocks: int | None = None  # launch grid size, if known

    @property
    def intensity(self):
        return self.flops / self.bytes


def _s(dtype):
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype!r}")
    return DTYPE_BYTES[dtype]


def gemm(m, n, k, dtype, path, blocks=None):
    """C[MxN] = A[MxK] B[KxN]; W = 2MNK; Q = s(MK + KN + MN)."""
    if min(m, n, k) <= 0:
        raise ValueError("dims must be positive")
    s = _s(dtype)
    return Workload(f"gemm_{m}x{n}x{k}_{dtype}", 2.0 * m * n * k, float(s * (m * k + k * n + m * n)), path,
                    blocks=blocks)


def gemv(m, k, dtype, path, blocks=None):
    """y[M] = A[MxK] x[K]: a GEMM with N = 1."""
    w = gemm(m, 1, k, dtype, path, blocks)
    return Workload(f"gemv_{m}x{k}_{dtype}", w.flops, w.bytes, path, blocks=blocks)


def elementwise(n, dtype, path, n_inputs=2, flops_per_elem=1, blocks=None):
    """out = f(in_1..in_k): reads n_inputs*n elements, writes n."""
    if n <= 0:
        raise ValueError("n must be positive")
    s = _s(dtype)
    return Workload(f"elementwise_{n}_{dtype}", float(flops_per_elem * n), float(s * n * (n_inputs + 1)), path,
                    blocks=blocks)


def reduction(n, dtype, path, blocks=None):
    """sum of n elements: n-1 adds (counted as n), reads n, writes 1."""
    if n <= 0:
        raise ValueError("n must be positive")
    s = _s(dtype)
    return Workload(f"reduction_{n}_{dtype}", float(n), float(s * (n + 1)), path, blocks=blocks)


def ridge(peak_flops, bandwidth):
    """I* = P_peak / beta (FLOP/byte)."""
    if peak_flops <= 0 or bandwidth <= 0:
        raise ValueError("ceilings must be positive")
    return peak_flops / bandwidth


def attainable(intensity, peak_flops, bandwidth):
    """P <= min(P_peak, beta * I)."""
    return min(peak_flops, bandwidth * intensity)


def bound(workload, ceilings):
    """Time lower bounds and screen regime for `workload` under one ceiling set.

    Refuses (ValueError) when the ceiling set has no compute ceiling for the
    workload's path or no bandwidth ceiling at its byte boundary.
    """
    peak = ceilings["compute"].get(workload.path)
    beta = ceilings["bandwidth"].get(workload.boundary)
    if peak is None:
        raise ValueError(f"no compute ceiling for path {workload.path!r}: W and P_peak must share an instruction path")
    if beta is None:
        raise ValueError(f"no bandwidth ceiling at boundary {workload.boundary!r}: Q and beta must share a boundary")
    t_c = workload.flops / peak
    t_m = workload.bytes / beta
    return {"workload": workload.name, "path": workload.path, "boundary": workload.boundary,
            "W": workload.flops, "Q": workload.bytes, "I": workload.intensity, "I_star": ridge(peak, beta),
            "t_compute_s": t_c, "t_memory_s": t_m, "t_lower_s": max(t_c, t_m),
            "screen": "compute" if t_c >= t_m else "memory",
            "ceiling_kind": ceilings.get("kind", "unstated")}


def qualify(b, measured_s, blocks=None, sms=None, near=0.5):
    """Compare a measured, synchronized time with the bound. Never relabels the screen.

    Returns fraction_of_bound = t_lower / measured (<= 1 unless the bound or the
    measurement is wrong) and flags that send the analyst to other limiters.
    """
    if measured_s <= 0:
        raise ValueError("measured_s must be positive")
    frac = b["t_lower_s"] / measured_s
    flags = []
    if frac > 1.0:
        flags.append("faster_than_bound: check counts, ceilings, path/boundary, or timing")
    elif frac < near:
        flags.append("far_below_roof: unmodeled limiter (launch, parallelism, dependencies, lower sustainable ceiling)")
    if blocks is not None and sms is not None and blocks < sms:
        flags.append("underfilled_grid: fewer blocks than SMs")
    return {**b, "measured_s": measured_s, "fraction_of_bound": frac, "flags": flags}


def classification_flips(workloads, ceilings_a, ceilings_b):
    """Names of workloads whose screen regime differs between two ceiling sets."""
    return [w.name for w in workloads if bound(w, ceilings_a)["screen"] != bound(w, ceilings_b)["screen"]]
