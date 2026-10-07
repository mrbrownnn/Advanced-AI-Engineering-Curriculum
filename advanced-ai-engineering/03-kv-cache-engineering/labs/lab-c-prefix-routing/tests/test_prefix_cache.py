"""Lab C: cache identity, residency vs ownership, LRU eviction (Lesson 3.6)."""
from hypothesis import given, settings, strategies as st

from prefix_cache import Identity, PrefixCachingBlockManager, block_hashes

HID = "HINT: What if the exact same prompt is passed to two completely different model architectures or checkpoints?"
HFULL = "HINT: only full blocks are hashed; a partial tail is never indexed."
HRES = ("HINT: refcount == 0 means no active owner, not 'empty': a cached zero-ref block stays "
        "reusable until it is popped for reuse.")
M = Identity("m1")


def test_hash_chain_counts_full_blocks_only():
    assert len(block_hashes(list(range(47)), 16, M)) == 2, HFULL
    assert len(block_hashes(list(range(48)), 16, M)) == 3, HFULL
    assert block_hashes(list(range(32)), 16, M) == block_hashes(list(range(32)), 16, M), \
        "HINT: use a stable hash; Python's hash() is randomized per process."


def test_chain_depends_on_parent():
    a = block_hashes([1] * 16 + [2] * 16, 16, M)
    b = block_hashes([3] * 16 + [2] * 16, 16, M)
    assert a[1] != b[1], "HINT: identical block tokens after different prefixes must hash differently (parent chain)."


def test_extra_keys_separate_identities():
    toks = list(range(64))
    base = block_hashes(toks, 16, M)
    for other in (Identity("m2"), Identity("m1", adapter="lora-a"), Identity("m1", salt="tenant-b")):
        h = block_hashes(toks, 16, other)
        assert all(x != y for x, y in zip(base, h)), f"{other} shares a block hash with {M}. {HID}"


def test_identity_mismatch_misses_in_cache():
    bm = PrefixCachingBlockManager(16, 16)
    toks = list(range(70))
    bm.allocate_cached("a", toks, M)
    bm.free("a")
    assert bm.allocate_cached("b", toks, Identity("m1", adapter="lora-x")) == 0, HID
    assert bm.allocate_cached("c", toks, Identity("m1", salt="t2")) == 0, HID
    assert bm.allocate_cached("d", toks, M) == 64, HRES
    bm.check_invariants()


def test_zero_ref_cached_blocks_are_resident_and_reusable():
    bm = PrefixCachingBlockManager(16, 16)
    toks = list(range(100))  # 6 full blocks + partial tail
    bm.allocate_cached("r1", toks, M)
    bm.free("r1")
    assert bm.num_free() == 16, "HINT: released blocks return to the free queue."
    assert bm.num_cached_free() == 6, f"{bm.num_cached_free()} cached free blocks. {HRES} {HFULL}"
    used_before = bm.allocations
    hit = bm.allocate_cached("r2", toks, M)
    assert hit == 96, f"Lesson 3.6 example: 96-token full-block prefix should hit, got {hit}. {HRES}"
    assert bm.allocations - used_before == 1, "HINT: only the partial tail needs a fresh block."
    assert all(bm.refcount[b] == 1 for b in bm.tables["r2"][:6]), HRES
    bm.check_invariants()


def test_hit_never_covers_whole_prompt():
    bm = PrefixCachingBlockManager(8, 16)
    toks = list(range(32))
    bm.allocate_cached("a", toks, M)
    bm.free("a")
    assert bm.allocate_cached("b", toks, M) == 16, \
        "HINT: at least one token must be computed to produce logits; a fully cached prompt still recomputes its last block."


def test_shared_hit_while_active_bumps_refcount():
    bm = PrefixCachingBlockManager(16, 16)
    toks = list(range(40))
    bm.allocate_cached("a", toks, M)
    bm.allocate_cached("b", toks[:32] + [999] * 8, M)
    assert bm.tables["a"][:2] == bm.tables["b"][:2], HRES
    assert all(bm.refcount[x] == 2 for x in bm.tables["a"][:2]), "HINT: a hit adds an active reference."
    bm.check_invariants()


def test_lru_evicts_oldest_released_first_and_never_active():
    bm = PrefixCachingBlockManager(6, 4)
    bm.allocate_cached("old", [1] * 8 + [0], M)      # 2 full + tail
    bm.free("old")
    bm.allocate_cached("live", [2] * 8 + [0], M)     # stays active
    bm.allocate_cached("new", [3] * 4 + [0], M)
    bm.free("new")
    bm.allocate_cached("filler", [4] * 4 + [0], M)   # needs blocks: must evict
    assert bm.evictions >= 1, "HINT: popping an indexed free block is an eviction; count it."
    assert bm.allocate_cached("probe-live", [2] * 8 + [7], M) == 8, \
        f"an active sequence's cached blocks were evicted. {HRES}"
    bm.check_invariants()


def test_lru_order():
    bm = PrefixCachingBlockManager(4, 4)
    bm.allocate_cached("x", [1] * 4 + [0], M)
    bm.free("x")
    bm.allocate_cached("y", [2] * 4 + [0], M)
    bm.free("y")
    bm.allocate_cached("z", [5] * 5, M)           # 2 blocks: reuses the 2 least recently released
    assert bm.allocate_cached("py", [2] * 4 + [9], M) == 4, \
        "HINT: LRU: the most recently released cached block should survive; reuse the front of the free queue."


ops = st.lists(st.tuples(st.sampled_from(["req", "free"]), st.integers(0, 3), st.integers(1, 30)), max_size=40)


@settings(max_examples=100, deadline=None)
@given(script=ops)
def test_property_cache_never_corrupts_active_sequences(script):
    bm = PrefixCachingBlockManager(10, 4)
    live: dict[int, list[int]] = {}
    for i, (op, p, k) in enumerate(script):
        if op == "req":
            toks = [p] * 8 + list(range(100 + i, 100 + i + k % 6))
            try:
                bm.allocate_cached(i, toks, M)
                live[i] = toks
            except Exception as e:  # only capacity refusals are acceptable
                assert type(e).__name__ == "OutOfBlocks", f"unexpected {e!r}. {HRES}"
        elif live:
            s = sorted(live)[k % len(live)]
            bm.free(s)
            del live[s]
        bm.check_invariants()
        for s, toks in live.items():
            assert bm.tokens(s) == toks, f"sequence {s} content changed. {HRES}"
