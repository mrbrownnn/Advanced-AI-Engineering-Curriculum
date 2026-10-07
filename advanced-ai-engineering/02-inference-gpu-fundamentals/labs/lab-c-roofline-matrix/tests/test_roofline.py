"""LAB C: regime-classification invariants for a qualified Roofline screen."""
from pathlib import Path

import pytest
import yaml
from hypothesis import Phase, given, settings
from hypothesis import strategies as st

import roofline as R

SETS = yaml.safe_load((Path(__file__).resolve().parents[1] / "fixtures" / "ceilings.yaml").read_text())["ceiling_sets"]
EX = SETS["exercise"]
DET = settings(derandomize=True, max_examples=150, deadline=None, database=None, phases=[Phase.explicit, Phase.generate])
H_MATCH = "HINT: are W and P_peak counted on the same instruction path, and are Q and beta measured at the same boundary?"
H_BOUND = "HINT: a Roofline is a bound, and the boundary must match."
H_RIDGE = "HINT: does the measured time grow with N before the predicted ridge? If not, what else limits it?"


def test_counts_scale_correctly():
    a, b = R.gemm(64, 32, 16, "fp16", "fp16_tensor"), R.gemm(128, 32, 16, "fp16", "fp16_tensor")
    assert b.flops == 2 * a.flops, "HINT: FMA = 2 FLOPs; W = 2MNK is linear in each dimension."
    assert R.gemm(8, 8, 8, "fp32", "fp32_cuda").bytes == 2 * R.gemm(8, 8, 8, "fp16", "fp16_tensor").bytes, (
        "HINT: one-pass bytes scale with bytes per element.")
    v, g1 = R.gemv(256, 128, "fp16", "fp16_tensor"), R.gemm(256, 1, 128, "fp16", "fp16_tensor")
    assert (v.flops, v.bytes) == (g1.flops, g1.bytes), "HINT: a GEMV is the N = 1 GEMM."
    e = R.elementwise(1000, "fp32", "fp32_cuda", n_inputs=2)
    assert e.bytes == 4 * 1000 * 3, "HINT: elementwise reads every input element once and writes every output once."
    r = R.reduction(1000, "fp32", "fp32_cuda")
    assert r.intensity < 1, "HINT: a reduction does about one add per element loaded."
    for bad in (lambda: R.gemm(0, 1, 1, "fp16", "p"), lambda: R.gemm(1, 1, 1, "fp7", "p")):
        with pytest.raises(ValueError):
            bad()


def test_gemm_intensity_grows_with_batch_and_regime_flips_once():
    ns = [1, 8, 64, 512, 4096]
    ws = [R.gemm(4096, n, 4096, "fp16", "fp16_tensor") for n in ns]
    its = [w.intensity for w in ws]
    assert all(a < b for a, b in zip(its, its[1:])), f"{H_RIDGE} More columns reuse A more: intensity must rise with N."
    screens = [R.bound(w, EX)["screen"] for w in ws]
    assert screens[0] == "memory" and screens[-1] == "compute", f"{H_RIDGE} GEMV-like and large-GEMM ends differ."
    assert screens == sorted(screens, key=lambda s: s == "compute"), f"{H_RIDGE} The screen crosses the ridge once."


@DET
@given(m=st.integers(1, 8192), n=st.integers(1, 8192), k=st.integers(1, 8192),
       peak=st.floats(1e11, 1e15), beta=st.floats(1e10, 1e13))
def test_screen_agrees_with_ridge_and_attainable(m, n, k, peak, beta):
    w = R.gemm(m, n, k, "fp16", "p")
    c = {"compute": {"p": peak}, "bandwidth": {"dram": beta}}
    b = R.bound(w, c)
    assert b["t_lower_s"] == max(b["t_compute_s"], b["t_memory_s"]), f"{H_BOUND} T >= max(W/P, Q/beta)."
    if w.intensity > R.ridge(peak, beta) * (1 + 1e-9):
        assert b["screen"] == "compute", f"{H_BOUND} Above the ridge the compute term dominates."
    if w.intensity < R.ridge(peak, beta) * (1 - 1e-9):
        assert b["screen"] == "memory", f"{H_BOUND} Below the ridge the memory term dominates."
    att = R.attainable(w.intensity, peak, beta)
    assert att <= peak * (1 + 1e-12) and att <= beta * w.intensity * (1 + 1e-12), f"{H_BOUND} P <= min(P_peak, beta*I)."
    assert w.flops / b["t_lower_s"] == pytest.approx(att, rel=1e-9), f"{H_BOUND} W / T_lower equals attainable P."


def test_raising_bandwidth_never_turns_compute_into_memory():
    ws = [R.gemm(4096, n, 4096, "fp16", "fp16_tensor") for n in (1, 64, 128, 256, 512)]
    fast = {"compute": EX["compute"], "bandwidth": {"dram": EX["bandwidth"]["dram"] * 4}}
    for w in ws:
        if R.bound(w, EX)["screen"] == "compute":
            assert R.bound(w, fast)["screen"] == "compute", f"{H_BOUND} More bandwidth moves the ridge left."


def test_measured_ceiling_can_flip_classification():
    ws = [R.gemm(4096, n, 4096, "fp16", "fp16_tensor") for n in (1, 64, 128, 256, 512)]
    sustained = {"kind": "measured", "compute": EX["compute"], "bandwidth": {"dram": EX["bandwidth"]["dram"] * 0.5}}
    flips = R.classification_flips(ws, EX, sustained)
    assert flips, ("HINT: the pre-registered hypothesis -- classification changes for at least one shape when a "
                   "measured sustainable ceiling replaces the optimistic one.")
    assert set(flips) <= {w.name for w in ws}, "HINT: return workload names."


def test_mismatched_path_or_boundary_is_refused():
    fp32_kernel = R.gemm(1024, 1024, 1024, "fp32", "fp32_cuda")
    with pytest.raises(ValueError):
        R.bound(fp32_kernel, EX)  # EX only has an fp16 tensor ceiling
    l2_bytes = R.Workload("l2_counted", 1e9, 1e8, "fp16_tensor", boundary="l2")
    with pytest.raises(ValueError):
        R.bound(l2_bytes, EX)  # EX only has a DRAM ceiling


def test_qualify_flags_without_relabeling():
    w = R.gemv(4096, 4096, "fp16", "fp16_tensor", blocks=8)
    b = R.bound(w, EX)
    slow = R.qualify(b, b["t_lower_s"] * 3, blocks=8, sms=EX["sms"])
    assert slow["screen"] == b["screen"], ("HINT: a measured point far below the roof sends you looking for another "
                                           "limiter; it does not refute the classification.")
    assert 0 < slow["fraction_of_bound"] < 1, H_BOUND
    assert any("far_below_roof" in f for f in slow["flags"]), "HINT: below both ceilings implies an unmodeled limiter."
    assert any("underfilled_grid" in f for f in slow["flags"]), "HINT: intensity does not encode grid size."
    fast = R.qualify(b, b["t_lower_s"] / 2)
    assert any("faster_than_bound" in f for f in fast["flags"]), ("HINT: faster than a lower bound means the counts, "
                                                                  "ceilings, boundary, or timing are wrong.")
    ok = R.qualify(b, b["t_lower_s"] * 1.1, blocks=1000, sms=EX["sms"])
    assert ok["flags"] == [], "HINT: near the roof with a full grid raises no flag."
    with pytest.raises(ValueError):
        R.qualify(b, 0)


def test_tiny_high_intensity_problem_is_flagged_by_grid_not_intensity():
    tiny = R.gemm(64, 64, 64, "fp16", "fp16_tensor", blocks=1)
    b = R.bound(tiny, EX)
    q = R.qualify(b, max(b["t_lower_s"] * 100, 5e-6), blocks=1, sms=EX["sms"])
    assert any("underfilled_grid" in f for f in q["flags"]), ("HINT: a 64x64 output may be one thread block; "
                                                              "the GPU is underfilled whatever its intensity.")
