"""LAB C starter: qualified Roofline screen.

Conventions (D): FMA = 2 FLOPs; one-pass algorithmic bytes = each input read
once, each output written once, at the boundary named in `Workload.boundary`
(default "dram"); beta*C = 0 for GEMM. Units: FLOPs, bytes, seconds.

A ceiling set is a dict like fixtures/ceilings.yaml entries:
    {"kind": "nameplate", "compute": {"fp16_tensor": 65e12}, "bandwidth": {"dram": 320e9}}
Fill every TODO.
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
    """C[MxN] = A[MxK] B[KxN]. Return a Workload with one-pass W and Q. ValueError on non-positive dims."""
    raise NotImplementedError("TODO: gemm")


def gemv(m, k, dtype, path, blocks=None):
    """y[M] = A[MxK] x[K] -- the N = 1 case of gemm."""
    raise NotImplementedError("TODO: gemv")


def elementwise(n, dtype, path, n_inputs=2, flops_per_elem=1, blocks=None):
    """out = f(in_1..in_k) over n elements: reads n_inputs*n elements, writes n."""
    raise NotImplementedError("TODO: elementwise")


def reduction(n, dtype, path, blocks=None):
    """Sum of n elements: count n FLOPs, read n elements, write 1."""
    raise NotImplementedError("TODO: reduction")


def ridge(peak_flops, bandwidth):
    """I* in FLOP/byte. ValueError on non-positive ceilings."""
    raise NotImplementedError("TODO: ridge")


def attainable(intensity, peak_flops, bandwidth):
    """Basic Roofline attainable performance (FLOP/s)."""
    raise NotImplementedError("TODO: attainable")


def bound(workload, ceilings):
    """Return dict with keys: workload, path, boundary, W, Q, I, I_star, t_compute_s, t_memory_s,
    t_lower_s, screen ("compute" | "memory"), ceiling_kind.

    Raise ValueError if `ceilings` has no compute ceiling for workload.path or no
    bandwidth ceiling for workload.boundary -- never borrow a ceiling from another path/boundary.
    """
    raise NotImplementedError("TODO: bound")


def qualify(b, measured_s, blocks=None, sms=None, near=0.5):
    """Compare a measured synchronized time with bound() output `b`.

    Return {**b, measured_s, fraction_of_bound (= t_lower_s / measured_s), flags (list[str])}.
    Flags (substring names tests look for): "faster_than_bound" when fraction > 1;
    "far_below_roof" when fraction < near; "underfilled_grid" when blocks < sms.
    Do NOT change b["screen"]: a slow measurement is a reason to look for another limiter,
    not a reclassification. ValueError on measured_s <= 0.
    """
    raise NotImplementedError("TODO: qualify")


def classification_flips(workloads, ceilings_a, ceilings_b):
    """Names of workloads whose screen regime differs between two ceiling sets."""
    raise NotImplementedError("TODO: classification_flips")
