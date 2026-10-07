"""Lab D: payload/metadata, transfer bounds, quality gate. Worked numbers are the README's own."""
import pytest

import make_eval
from bounds import (GIB, KIB, Bandwidth, capacity_claim, concurrency_bound, critical_path_s, disagg_ttft_s,
                    payload_bytes_per_token, quality_gate, retained_fraction_scores, scale_bytes_per_token,
                    setup_dominated, transfer_lower_bound_s)

HQ = "HINT: Does quantizing the KV cache shrink the model weights?"
HU = "HINT: Ensure units align before dividing by bandwidth."
HGATE = "HINT: do not preserve a capacity claim after the quality gate fails."
HSETUP = "HINT: serialization time is a lower bound; setup, synchronization and contention add to it."
MHA = dict(layers=32, kv_heads=32, head_dim=128)


def test_lesson38_payload_plus_metadata():
    fp16 = payload_bytes_per_token(**MHA, bits=16)
    fp8 = payload_bytes_per_token(**MHA, bits=8)
    scales = scale_bytes_per_token(32, 32, scale_bytes=4, group_tokens=16)
    assert fp16 == 512 * KIB and fp8 == 256 * KIB, HQ
    assert scales == 512, f"one FP32 scale per (layer, K/V, head, 16-token block): got {scales}. {HQ}"
    assert (fp8 + scales) / KIB == 256.5, HQ
    assert fp16 / (fp8 + scales) < 2, f"metadata keeps the reduction below the bit-width ratio. {HQ}"


def test_lesson38_concurrency_screening_bound():
    budget = 63 * GIB
    n16 = concurrency_bound(budget, 4096, payload_bytes_per_token(**MHA, bits=16))
    n8 = concurrency_bound(budget, 4096, payload_bytes_per_token(**MHA, bits=8) + scale_bytes_per_token(32, 32, 4, 16))
    assert (n16, n8) == (31, 62), f"Lesson 3.8: got {(n16, n8)}. {HQ}"


def test_metadata_grows_as_groups_shrink():
    coarse = scale_bytes_per_token(32, 32, 4, 64)
    fine = scale_bytes_per_token(32, 32, 4, 1)
    assert fine == 64 * coarse, "HINT: a different scale granularity changes the metadata term."
    with pytest.raises(ValueError):
        scale_bytes_per_token(32, 32, 4, 0)


def test_lesson39_transfer_bounds():
    per_tok = payload_bytes_per_token(**MHA, bits=16)
    t4k = transfer_lower_bound_s(4096 * per_tok, Bandwidth(50, "GiB/s"))
    t16k = transfer_lower_bound_s(16384 * per_tok, Bandwidth(50, "GiB/s"))
    assert t4k == pytest.approx(0.040) and t16k == pytest.approx(0.160), f"Lesson 3.9: {t4k}, {t16k}. {HU}"


def test_unit_guard_gib_vs_gb():
    payload = 2 * GIB
    gib = transfer_lower_bound_s(payload, Bandwidth(50, "GiB/s"))
    gb = transfer_lower_bound_s(payload, Bandwidth(50, "GB/s"))
    assert gb / gib == pytest.approx(GIB / 1e9), f"GB/s is smaller than GiB/s by 1.0737x. {HU}"
    with pytest.raises(TypeError):
        transfer_lower_bound_s(payload, 50)
    with pytest.raises(ValueError):
        transfer_lower_bound_s(payload, Bandwidth(50, "Gbps"))


def test_critical_path_bounds_and_overlap():
    p, bw = 2 * GIB, Bandwidth(50, "GiB/s")
    t = transfer_lower_bound_s(p, bw)
    assert critical_path_s(p, bw) == pytest.approx(t), HSETUP
    assert critical_path_s(p, bw, setup_s=0.005) > t, HSETUP
    assert critical_path_s(p, bw, overlap_fraction=1.0) == 0, "HINT: only the non-overlapped portion is on the critical path."
    assert critical_path_s(p, bw, 0.002, 0.5) < critical_path_s(p, bw, 0.002, 0.0), HSETUP
    with pytest.raises(ValueError):
        critical_path_s(p, bw, overlap_fraction=1.5)


def test_break_small_transfer_is_setup_dominated():
    per_tok = payload_bytes_per_token(**MHA, bits=16)
    bw = Bandwidth(50, "GiB/s")
    small, large = 16 * per_tok, 16384 * per_tok
    assert setup_dominated(small, bw, setup_s=0.001), f"16 tokens at 50 GiB/s vs 1 ms setup. {HSETUP}"
    assert not setup_dominated(large, bw, setup_s=0.001), HSETUP
    eff_small = small / critical_path_s(small, bw, setup_s=0.001)
    assert eff_small < 0.5 * bw.bytes_per_s(), f"effective bandwidth collapses for small transfers. {HSETUP}"


def test_disagg_ttft_adds_non_overlapped_transfer():
    p, bw = 2 * GIB, Bandwidth(50, "GiB/s")
    base = disagg_ttft_s(0.2, p, bw, setup_s=0.0, overlap_fraction=1.0, first_decode_s=0.02)
    assert base == pytest.approx(0.22), "HINT: fully overlapped transfer adds nothing to TTFT."
    worst = disagg_ttft_s(0.2, p, bw, setup_s=0.003, overlap_fraction=0.0, first_decode_s=0.02)
    assert worst == pytest.approx(0.22 + 0.04 + 0.003), HSETUP


def test_quality_gate_semantics():
    assert quality_gate([1, 1, 1, 1], [1, 1, 1, 1], 0.0).passed, HGATE
    g = quality_gate([1, 1, 1, 1], [1, 1, 0, 1], 0.1)
    assert not g.passed and g.metric == pytest.approx(0.25), HGATE
    with pytest.raises(ValueError):
        quality_gate([1, 1], [1], 0.1)


def test_break_lossy_retention_fails_gate_and_kills_capacity_claim():
    data = make_eval.items(200, seed=0)
    base = [s for s, _ in data]
    lossy = retained_fraction_scores(data, keep_fraction=0.5)
    g = quality_gate(base, lossy, max_drop=0.02)
    assert not g.passed, f"keeping half the tokens must fail a 2-point gate here. {HGATE}"
    assert capacity_claim(124, g) is None, HGATE
    keep_all = quality_gate(base, retained_fraction_scores(data, keep_fraction=1.0), max_drop=0.02)
    assert keep_all.passed and capacity_claim(62, keep_all) == 62, HGATE
