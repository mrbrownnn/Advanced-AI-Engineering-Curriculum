"""Lab B: paging, refcount, CoW, and invariant behavior (Lessons 3.3-3.4)."""
import pytest
from hypothesis import given, settings, strategies as st

from block_manager import BlockManager, InvariantError, OutOfBlocks

H33 = "HINT: 8 tokens filled exactly how many 4-token blocks? What does the 9th token trigger?"
H34 = "HINT: Can a token be inserted into a block that is already full?"
HCOW = ("HINT: copy only when another table still observes the partial tail; the last sole owner "
        "writes in place, and the writer's table must point at the copy.")
HINV = "HINT: a physical block cannot be free and allocated at once; refcount == table references."


def test_new_block_on_ninth_token_not_eighth():
    bm = BlockManager(8, 4)
    bm.allocate("s", list(range(8)))
    assert len(bm.tables["s"]) == 2, f"8 tokens at B=4 -> {len(bm.tables['s'])} blocks. {H33}"
    free_before = bm.num_free()
    assert bm.append_cost("s") == 1, f"appending the 9th token needs a new block. {H33}"
    bm.append("s", 8)
    assert len(bm.tables["s"]) == 3 and bm.num_free() == free_before - 1, H33
    new = bm.tables["s"][2]
    assert new not in bm.free_queue, f"mapped block left in the free pool. {HINV}"
    assert bm.locate("s", 8) == (new, 0), f"token index 8 -> logical 2, offset 0. {H33}"
    bm.check_invariants()


def test_locate_matches_block_table():
    bm = BlockManager(6, 16)
    bm.allocate("s", list(range(34)))
    t = bm.tables["s"]
    assert len(t) == 3, f"ceil(34/16) blocks. {H33}"
    assert bm.locate("s", 33) == (t[2], 1), "HINT: logical_block = floor(i/B), offset = i % B."
    with pytest.raises(IndexError):
        bm.locate("s", 34)


def test_out_of_blocks_leaves_state_unchanged():
    bm = BlockManager(2, 4)
    bm.allocate("a", list(range(8)))
    snapshot = (list(bm.free_queue), list(bm.refcount), dict(bm.lengths))
    with pytest.raises(OutOfBlocks):
        bm.append("a", 9)
    with pytest.raises(OutOfBlocks):
        bm.allocate("b", [1])
    assert (list(bm.free_queue), list(bm.refcount), dict(bm.lengths)) == snapshot, \
        "HINT: check feasibility before mutating; a refused allocation must change nothing."
    assert "b" not in bm.tables, "HINT: a refused allocation must not leave a half-built table."
    bm.check_invariants()


def test_full_shared_block_needs_no_cow():
    bm = BlockManager(8, 16)
    bm.allocate("a", list(range(16)))
    bm.fork("a", "b")
    shared = bm.tables["a"][0]
    assert bm.refcount[shared] == 2, "HINT: refcount = number of active logical references."
    bm.append("a", 100)
    assert bm.cow_events == 0, f"appending past a full shared block copied it. {H34}"
    assert bm.refcount[shared] == 2 and bm.tables["a"][0] == shared, H34
    bm.check_invariants()


def test_partial_shared_tail_cow_isolates():
    bm = BlockManager(8, 16)
    bm.allocate("a", list(range(14)))
    bm.fork("a", "b")
    orig = bm.tables["a"][0]
    bm.append("a", 99)
    assert bm.cow_events == 1, f"shared partial tail must be copied before writing. {HCOW}"
    assert bm.tables["a"][0] != orig and bm.tables["b"][0] == orig, HCOW
    assert bm.refcount[orig] == 1, f"original refcount should drop to 1. {HCOW}"
    assert bm.tokens("b") == list(range(14)), f"writer corrupted the other sequence. {HCOW}"
    assert bm.tokens("a") == list(range(14)) + [99], HCOW
    bm.check_invariants()


def test_three_samples_cow_refcount_trace():
    bm = BlockManager(8, 16)
    bm.allocate("s1", list(range(14)))
    bm.fork("s1", "s2")
    bm.fork("s1", "s3")
    orig = bm.tables["s1"][0]
    assert bm.refcount[orig] == 3, HCOW
    bm.append("s1", 1)
    assert bm.refcount[orig] == 2, HCOW
    bm.append("s2", 2)
    assert bm.refcount[orig] == 1, HCOW
    bm.append("s3", 3)
    assert bm.cow_events == 2, f"{bm.cow_events} copies for 3 writers; the last sole owner should not copy. {HCOW}"
    assert bm.tables["s3"][0] == orig, HCOW
    assert len({bm.tables[s][0] for s in ("s1", "s2", "s3")}) == 3, HCOW
    assert all(bm.refcount[bm.tables[s][0]] == 1 for s in ("s1", "s2", "s3")), HCOW
    bm.check_invariants()


def test_free_returns_blocks_and_forgets_sequence():
    bm = BlockManager(4, 4)
    bm.allocate("a", list(range(6)))
    bm.fork("a", "b")
    bm.free("a")
    assert bm.num_free() == 2, "HINT: shared blocks stay allocated while another table references them."
    bm.free("b")
    assert bm.num_free() == 4 and not bm.tables, "HINT: refcount 0 returns a block to the free pool."
    bm.check_invariants()


def test_touch_revives_zero_ref_block():
    bm = BlockManager(4, 4)
    bm.allocate("a", [1, 2, 3, 4])
    b = bm.tables["a"][0]
    bm.free("a")
    assert b in bm.free_queue, HINV
    bm.allocate("x", [1, 2, 3, 4, 5], prefix_blocks=[b])
    assert b not in bm.free_queue and bm.refcount[b] == 1, \
        "HINT: touching a zero-ref block must remove it from the free pool (Lesson 3.6)."
    bm.check_invariants()


# ---- BREAK: each fault must raise, not corrupt silently -------------------

def test_break_double_free_raises():
    bm = BlockManager(4, 4)
    bm.allocate("a", [1, 2])
    bm.free("a")
    with pytest.raises(InvariantError):
        bm.free("a")
    b = 0
    with pytest.raises(InvariantError):
        bm._release(b)
    bm.check_invariants()


def test_break_write_through_shared_tail_raises():
    bm = BlockManager(4, 16)
    bm.allocate("a", list(range(14)))
    bm.fork("a", "b")
    tail = bm.tables["a"][-1]
    with pytest.raises(InvariantError):
        bm._write(tail, 14, 99)  # bypassing CoW must be refused by the guard
    assert bm.tokens("b") == list(range(14)), HCOW


def test_break_stale_table_entry_detected():
    bm = BlockManager(4, 4)
    bm.allocate("a", [1, 2, 3, 4, 5])
    stale = bm.tables["a"][0]
    bm._release(stale)  # refcount dropped, table not redirected: a stale mapping
    with pytest.raises(InvariantError):
        bm.check_invariants()


def test_break_free_and_allocated_detected():
    bm = BlockManager(4, 4)
    bm.allocate("a", [1, 2, 3, 4, 5])
    bm.free_queue.append(bm.tables["a"][0])
    with pytest.raises(InvariantError):
        bm.check_invariants()


def test_break_leak_detected():
    bm = BlockManager(4, 4)
    bm.allocate("a", [1])
    del bm.tables["a"], bm.lengths["a"]  # forgot the table without releasing
    with pytest.raises(InvariantError):
        bm.check_invariants()


# ---- property test --------------------------------------------------------

ops = st.lists(st.tuples(st.sampled_from(["new", "append", "fork", "free"]),
                         st.integers(0, 5), st.integers(1, 20)), max_size=60)


@settings(max_examples=150, deadline=None)
@given(block_size=st.integers(1, 5), num_blocks=st.integers(1, 12), script=ops)
def test_property_invariants_and_isolation(block_size, num_blocks, script):
    bm = BlockManager(num_blocks, block_size)
    model: dict[str, list[int]] = {}  # expected token content per sequence
    n = 0
    for op, i, k in script:
        names = sorted(model)
        try:
            if op == "new":
                toks = list(range(n, n + k))
                bm.allocate(f"s{n}", toks)
                model[f"s{n}"] = toks
            elif op == "append" and names:
                s = names[i % len(names)]
                bm.append(s, n)
                model[s] = model[s] + [n]
            elif op == "fork" and names:
                s = names[i % len(names)]
                bm.fork(s, f"s{n}")
                model[f"s{n}"] = list(model[s])
            elif op == "free" and names:
                s = names[i % len(names)]
                bm.free(s)
                del model[s]
        except OutOfBlocks:
            pass
        n += 1
        bm.check_invariants()
        for s, toks in model.items():
            assert bm.tokens(s) == toks, f"sequence {s} content changed by another sequence. {HCOW}"
    assert bm.num_free() + len({b for t in bm.tables.values() for b in t}) == num_blocks, HINV
