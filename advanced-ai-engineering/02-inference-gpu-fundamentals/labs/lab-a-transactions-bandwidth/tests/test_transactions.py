"""LAB A invariants. Behavior and invariants only; no pasteable traffic tables."""
from pathlib import Path

import numpy as np
import pytest
import yaml
from hypothesis import Phase, given, settings
from hypothesis import strategies as st

import transactions as T

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "patterns.yaml"
DET = settings(derandomize=True, max_examples=150, deadline=None, database=None, phases=[Phase.explicit, Phase.generate])
H_BYTES = "HINT: useful vs transferred bytes at a named boundary -- a sector moves whole even if one lane needs 4 bytes of it."
H_UNITS = "HINT: convert to bytes and seconds first, then divide by 2**30 or 10**9 exactly once."


def addresses_for(p, n=32):
    if p["kind"] == "strided":
        return T.strided_addresses(n, p["elem_bytes"], p["stride_elems"], p["offset_elems"])
    if p["kind"] == "aos":
        return T.aos_addresses(n, p["struct_bytes"], p["field_offset"], p["field_bytes"])
    return T.soa_addresses(n, p["field_bytes"])


def access_for(p):
    return p.get("elem_bytes") or p["field_bytes"]


def test_aligned_contiguous_is_fully_efficient():
    for eb in (2, 4, 8, 16):
        t = T.warp_traffic(T.strided_addresses(32, eb), eb)
        assert t["efficiency"] == 1.0, f"{H_BYTES} Aligned stride-1 {eb}-byte lanes should waste nothing."
        assert t["moved_bytes"] == t["useful_bytes"] == 32 * eb, H_BYTES


def test_access_crossing_a_sector_boundary_touches_both_sectors():
    for sector in (32, 64, 128):
        s = T.sectors_touched([sector - 2], 4, sector)
        assert s == {0, 1}, (f"HINT: an access is a byte RANGE [a, a+access_bytes); count every sector it "
                             f"overlaps, not only the sector holding its first byte (sector_bytes={sector}).")


def test_misalignment_adds_moved_bytes_without_adding_useful_bytes():
    aligned = T.warp_traffic(T.strided_addresses(32, 4, 1, 0), 4)
    for off in (1, 3, 7):
        shifted = T.warp_traffic(T.strided_addresses(32, 4, 1, off), 4)
        assert shifted["useful_bytes"] == aligned["useful_bytes"], H_BYTES
        assert shifted["moved_bytes"] > aligned["moved_bytes"], (
            f"{H_BYTES} An offset start straddles one more sector than the aligned case (offset={off}).")


def test_efficiency_non_increasing_with_stride():
    for eb in (2, 4, 8):
        effs = [T.warp_traffic(T.strided_addresses(32, eb, s), eb)["efficiency"] for s in (1, 2, 4, 8, 16, 32, 64)]
        assert all(a >= b for a, b in zip(effs, effs[1:])), (
            f"{H_BYTES} Spreading the same lanes over a larger stride cannot reduce waste ({eb}-byte elements).")


def test_large_stride_saturates_at_one_sector_per_lane():
    for stride in (64, 128, 1024):
        t = T.warp_traffic(T.strided_addresses(32, 4, stride), 4)
        assert t["sectors"] == 32, f"{H_BYTES} When lanes are farther apart than a sector, each lane pays its own sector."


def test_broadcast_counts_useful_bytes_once():
    t = T.warp_traffic([128] * 32, 4)
    assert t["useful_bytes"] == 4 and t["sectors"] == 1, (
        "HINT: useful bytes are DISTINCT requested bytes; 32 lanes reading the same word need 4 bytes, 1 sector.")


def test_aos_wastes_sectors_relative_to_soa():
    aos = T.warp_traffic(T.aos_addresses(32, 16, 0, 4), 4)
    soa = T.warp_traffic(T.soa_addresses(32, 4), 4)
    assert aos["useful_bytes"] == soa["useful_bytes"], H_BYTES
    assert aos["efficiency"] < soa["efficiency"], (
        f"{H_BYTES} Reading one 4-byte field of a 16-byte struct per lane is a stride in disguise.")
    same_as_stride = T.warp_traffic(T.strided_addresses(32, 4, 4), 4)
    assert aos["moved_bytes"] == same_as_stride["moved_bytes"], H_BYTES


def test_pattern_traffic_is_additive_over_warps():
    one = T.warp_traffic(T.strided_addresses(32, 4, 2), 4)
    two = T.pattern_traffic(T.strided_addresses(64, 4, 2), 4)
    assert two["useful_bytes"] == 2 * one["useful_bytes"], "HINT: split lanes into warps of 32, then sum per warp."
    assert two["moved_bytes"] == 2 * one["moved_bytes"], "HINT: split lanes into warps of 32, then sum per warp."
    assert set(two) >= {"useful_bytes", "moved_bytes", "sectors", "efficiency"}, "HINT: keep the documented keys."


@DET
@given(n=st.integers(1, 64), eb=st.sampled_from([1, 2, 4, 8, 16]), stride=st.integers(1, 80),
       off=st.integers(0, 40), sector=st.sampled_from([32, 64, 128]))
def test_moved_bounds(n, eb, stride, off, sector):
    t = T.pattern_traffic(T.strided_addresses(n, eb, stride, off), eb, sector)
    assert t["moved_bytes"] % sector == 0, f"{H_BYTES} Moved bytes are whole sectors."
    assert t["moved_bytes"] >= t["useful_bytes"], f"{H_BYTES} The hardware cannot move fewer bytes than were requested."
    assert 0 < t["efficiency"] <= 1.0, H_BYTES
    per_access = -(-eb // sector) + 1
    assert t["sectors"] <= n * per_access, f"{H_BYTES} One access spans at most ceil(access/sector)+1 sectors."


def test_fixture_patterns_have_equal_useful_bytes():
    pats = yaml.safe_load(FIX.read_text())["patterns"]
    for p in pats:
        t = T.warp_traffic(addresses_for(p), access_for(p))
        assert t["useful_bytes"] == 32 * access_for(p), f"{H_BYTES} pattern {p['name']}: layout must not change useful bytes."


def test_useful_bandwidth_units():
    b, s = 3 * 2**20, 2e-3
    gib, gb, bps = (T.useful_bandwidth(b, s, u) for u in ("GiB/s", "GB/s", "B/s"))
    assert bps == pytest.approx(b / s), H_UNITS
    assert gib * 2**30 == pytest.approx(bps) and gb * 1e9 == pytest.approx(bps), H_UNITS
    assert gib < gb, f"{H_UNITS} A GiB is larger than a GB, so the same rate is a smaller number in GiB/s."
    for bad in (0, -1.0):
        with pytest.raises(ValueError):
            T.useful_bandwidth(b, bad)
    with pytest.raises(ValueError):
        T.useful_bandwidth(b, s, "GiB/ms")


def test_vector_add_matches_oracle_and_checks_shape():
    rng = np.random.default_rng(77)
    a, b = rng.standard_normal(257).astype(np.float32), rng.standard_normal(257).astype(np.float32)
    np.testing.assert_array_equal(T.vector_add(a, b), a + b,
                                  err_msg="HINT: preserve element count and arithmetic while changing layout.")
    with pytest.raises(AssertionError):
        T.vector_add(np.ones(2), np.ones(3))


@pytest.mark.parametrize("m,k,n,tile", [(1, 1, 1, 16), (17, 33, 5, 16), (64, 64, 64, 8), (40, 7, 23, 32)])
def test_tiled_matmul_matches_oracle(m, k, n, tile):
    rng = np.random.default_rng(1234 + m)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    c = T.tiled_matmul(a, b, tile)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    assert c.shape == (m, n), "HINT: shape and bounds assertions -- edge tiles are partial when dims are not multiples of tile."
    np.testing.assert_allclose(c, ref, rtol=1e-4, atol=1e-4 * np.sqrt(k),
                               err_msg="HINT: correctness oracle first; accumulate in at least float32 and cover edge tiles.")


def test_tiled_matmul_rejects_bad_shapes():
    with pytest.raises(AssertionError):
        T.tiled_matmul(np.ones((3, 4)), np.ones((5, 2)))
    with pytest.raises(AssertionError):
        T.tiled_matmul(np.ones(3), np.ones((3, 2)))


def test_break_module_counts_boundary_crossing():
    import bugged_transactions as B
    # byte-misaligned warp (e.g. a packed struct or a sliced buffer); every access is a 4-byte range
    addrs = T.strided_addresses(32, 4, 1, 0, base=2)
    want = len(T.sectors_touched(addrs, 4))
    assert want == 5, f"{H_BYTES} 128 bytes starting at byte 2 span five 32-byte sectors."
    got = B.warp_traffic(addrs, 4)["sectors"]
    assert got == want, ("HINT: BREAK -- bugged_transactions under-counts a byte-misaligned warp. Find out why by "
                         "instrumenting single-lane accesses near a sector boundary; fix it in place and record it in report.md section 6.")
