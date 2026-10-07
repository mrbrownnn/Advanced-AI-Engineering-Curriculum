"""Lab B: replay the Lesson 3.5 worked example through the block manager.

The expected rows are the README's published table (Lesson 3.5, Step 5), not new answers.
"""
import json
import pathlib

import pytest

from timeline import k1, k2, lesson35, my_cost, replay

HINT = ("HINT: Replay the next 32 iterations. When does each request need a block, "
        "and when does A release its blocks?")
TABLE = json.loads((pathlib.Path(__file__).parents[1] / "fixtures" / "lesson35_table.json").read_text())
COSTS = {"K1": k1, "K2": k2}


@pytest.mark.parametrize("row", TABLE, ids=lambda r: f"{r['timeline']}-{r['cost']}-{r['policy']}")
def test_reproduces_lesson35_row(row):
    running, waiting = lesson35(row["timeline"])
    out = replay(running, waiting, COSTS[row["cost"]],
                 delay_until="A" if row["policy"] == "delay" else None)
    q = out["requests"]
    got = {"preemptions": out["preemptions"], "a_gap": round(q["A"]["mean_gap_ms"], 2),
           "b_gap": round(q["B"]["mean_gap_ms"], 2), "c_ttft": q["C"]["ttft_ms"], "c_done": q["C"]["done_ms"]}
    want = {k: row[k] for k in got}
    assert got == want, f"{row['timeline']}/{row['cost']}/{row['policy']}: got {got}. {HINT}"


def test_every_request_finishes_with_all_tokens():
    running, waiting = lesson35("T1")
    out = replay(running, waiting, k1)
    assert {n: r["tokens"] for n, r in out["requests"].items()} == {"A": 64, "B": 128, "C": 32}, \
        "HINT: a preempted request keeps its emitted tokens; recomputation re-prefills them."


def test_preemption_frees_victim_blocks_and_trace_is_consistent():
    running, waiting = lesson35("T1")
    out = replay(running, waiting, k1)
    tr = out["trace"]
    assert all(r["free_blocks"] >= 0 for r in tr), HINT
    assert [r["iteration"] for r in tr] == list(range(1, len(tr) + 1)), "HINT: one trace row per iteration."
    assert abs(sum(r["ms"] for r in tr) - tr[-1]["t_ms"]) < 1e-9, "HINT: clock = sum of iteration durations."
    pre = [e for r in tr for e in r["events"] if "preempt" in e]
    assert pre and pre[0]["preempt"] == "C", "HINT: the shortfall policy preempts the youngest request."
    assert all(r["free_blocks"] == 10 + 14 for r in tr[-1:]), \
        "HINT: after every request completes, all blocks must be back in the free pool."


def test_my_cost_model_is_well_formed():
    vals = [my_cost(b, p) for b in (1, 2, 3) for p in (0, 16, 96, 112)]
    assert all(v > 0 for v in vals), "HINT: a step cost must be positive."
    assert all(my_cost(b, 112) >= my_cost(b, 0) for b in (1, 2, 3)), \
        "HINT: more prefill tokens in an iteration should not make it cheaper."
    running, waiting = lesson35("T1")
    out = replay(running, waiting, my_cost)
    assert out["requests"]["C"]["done_ms"] > 0, HINT
