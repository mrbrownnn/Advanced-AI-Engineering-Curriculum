"""BREAK: starter/bugged_block_manager.py has one real lifecycle bug.

Every invariant check passes and no sequence is corrupted, yet it does extra
work. Instrument it (cow_events, allocations, refcount traces) to find the bug,
then fix it in place. This test passes once the bug is gone.
"""
import bugged_block_manager as bugged

HINT = ("HINT: Trace the refcount of a partial block shared by 3 parallel sampling sequences. "
        "Count copies per writer and compare with the Lesson 3.4 feedback contract.")


def _scenario(n_samples):
    bm = bugged.BlockManager(16, 16)
    bm.allocate("s0", list(range(14)))
    for i in range(1, n_samples):
        bm.fork("s0", f"s{i}")
    for i in range(n_samples):
        bm.append(f"s{i}", 100 + i)
    bm.check_invariants()
    for i in range(n_samples):
        assert bm.tokens(f"s{i}") == list(range(14)) + [100 + i], HINT
    return bm


def test_bugged_manager_copy_count():
    for n in (2, 3, 4):
        bm = _scenario(n)
        assert bm.cow_events == n - 1, f"{n} writers on one shared partial tail made {bm.cow_events} copies. {HINT}"
