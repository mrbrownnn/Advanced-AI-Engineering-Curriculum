"""Lab A: contiguous allocator and waste classification (Lesson 3.2)."""
from hypothesis import given, settings, strategies as st

import make_trace
from allocator import ContiguousAllocator, Request, classify, simulate

HLAYER = "HINT: Does Paged KV manage HBM directly or does it ask the framework allocator for memory?"
HCAT = ("HINT: reservation waste = reserved beyond held blocks; internal fragmentation = block "
        "rounding of held tokens; contiguous failure = enough total free space but no single hole.")


def test_readme_reservation_waste_example():
    a = ContiguousAllocator(capacity=8192)
    lengths = {"A": 1000, "B": 100, "C": 1000}
    for rid in lengths:
        assert isinstance(a.reserve(rid, 2048), int), HCAT
    c = classify(a, lengths)
    assert (c["logical"], c["allocated"], c["reservation_waste"]) == (2100, 6144, 4044), \
        f"Lesson 3.2 worked example: got {c}. {HCAT}"
    assert c["internal_fragmentation"] == 0, f"block_size=1 has no rounding slack. {HCAT}"


def test_internal_fragmentation_is_block_tail_slack():
    a = ContiguousAllocator(capacity=1024, block_size=16)
    a.reserve("x", 34)
    c = classify(a, {"x": 34})
    assert c["allocated"] == 48 and c["internal_fragmentation"] == 14 and c["reservation_waste"] == 0, \
        f"B=16, S=34: tail slack is B*ceil(S/B)-S. {HCAT}"


def test_contiguous_failure_vs_capacity_failure():
    a = ContiguousAllocator(capacity=10)
    assert a.reserve("a", 4) == 0 and a.reserve("b", 2) == 4 and a.reserve("c", 4) == 6, \
        "HINT: first-fit places each extent in the lowest hole that fits."
    a.release("a")
    a.release("c")
    assert a.free_total() == 8, HCAT
    assert a.reserve("d", 6) == "contiguous", f"8 free in two 4-slot holes cannot fit 6. {HCAT}"
    assert a.reserve("e", 9) == "capacity", HCAT
    assert "d" not in a.extents and "e" not in a.extents, "HINT: a failed reservation changes nothing."
    assert a.reserve("f", 4) == 0, "HINT: released extents become reusable holes."


def test_break_equal_fixed_lengths_have_zero_fragmentation():
    reqs = [Request(**r) for r in make_trace.equal_fixed(n=8, length=256)]
    out = simulate(reqs, capacity=8 * 256, policy="max_len")
    assert out["failures"] == {"capacity": 0, "contiguous": 0}, HCAT
    assert len(out["completed"]) == 8, HCAT
    for s in out["snapshots"]:
        assert s["internal_fragmentation"] == 0, f"step {s['step']}: {s}. {HCAT}"
        assert s["num_holes"] <= 1, f"equal extents freed together leave no scattered holes. {HCAT}"


def test_exact_policy_has_no_reservation_waste_at_completion():
    reqs = [Request(**r) for r in make_trace.heterogeneous(n=15, seed=1)]
    out = simulate(reqs, capacity=16_384, policy="exact")
    assert len(out["completed"]) == 15, HCAT
    assert all(s["internal_fragmentation"] == 0 for s in out["snapshots"]), HCAT


def test_heterogeneous_trace_shows_all_three_categories():
    reqs = [Request(**r) for r in make_trace.heterogeneous(n=40, seed=0)]
    out = simulate(reqs, capacity=6 * 2048 + 700, policy="max_len", block_size=16)
    snaps = out["snapshots"]
    assert max(s["reservation_waste"] for s in snaps) > 0, HCAT
    assert max(s["internal_fragmentation"] for s in snaps) > 0, HCAT
    assert len(out["completed"]) == 40, "HINT: FIFO admission must eventually drain the queue."
    varied = simulate(reqs, capacity=6000, policy="exact", block_size=16)
    assert varied["failures"]["contiguous"] > 0, \
        f"variable-size extents released at different times should strand free space in holes. {HCAT}"


@settings(max_examples=100, deadline=None)
@given(cap=st.integers(16, 400), bs=st.integers(1, 8),
       ops=st.lists(st.tuples(st.booleans(), st.integers(1, 60)), max_size=40))
def test_property_accounting_identity_and_no_overlap(cap, bs, ops):
    a = ContiguousAllocator(cap, bs)
    lengths = {}
    for i, (do_reserve, n) in enumerate(ops):
        if do_reserve or not a.extents:
            res = a.reserve(i, n)
            if isinstance(res, int):
                lengths[i] = n
            else:
                assert res in ("capacity", "contiguous"), HCAT
                fits = any(h >= -(-n // bs) * bs for _, h in a.holes())
                assert not fits, f"a hole fits but reserve() refused. {HCAT}"
        else:
            rid = sorted(a.extents)[n % len(a.extents)]
            a.release(rid)
            del lengths[rid]
        c = classify(a, lengths)
        assert c["allocated"] == c["logical"] + c["internal_fragmentation"] + c["reservation_waste"], HCAT
        assert c["allocated"] + c["free"] == cap, HLAYER
        spans = sorted(a.extents.values())
        for (s1, n1), (s2, _) in zip(spans, spans[1:]):
            assert s1 + n1 <= s2, "HINT: two extents overlap; contiguous reservations must be disjoint."
        assert sum(h for _, h in a.holes()) == c["free"], "HINT: holes must sum to free space."
