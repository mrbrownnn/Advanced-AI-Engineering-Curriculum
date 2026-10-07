"""Lab B experiment: replay Lesson 3.5 (T1/T2 x K1/K2/your cost x admit/delay).

Writes results/<stamp>-timeline/{raw.jsonl,result.yaml}; raw.jsonl holds the
per-iteration trace for every scenario so you can see whose gaps moved.
    python experiments/run_timeline.py
    LAB_IMPL=solution python experiments/run_timeline.py
"""
import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "_shared"))
import labkit  # noqa: E402

labkit.use_impl(LAB)
from timeline import k1, k2, lesson35, my_cost, replay  # noqa: E402

rows, summary = [], []
for T in ("T1", "T2"):
    for cname, cost in (("K1", k1), ("K2", k2), ("mine", my_cost)):
        for policy in ("admit", "delay"):
            running, waiting = lesson35(T)
            out = replay(running, waiting, cost, delay_until="A" if policy == "delay" else None)
            key = {"timeline": T, "cost": cname, "policy": policy}
            for it in out["trace"]:
                rows.append({**key, **it})
            s = {**key, "preemptions": out["preemptions"]}
            for n, r in out["requests"].items():
                s.update({f"{n}_mean_gap_ms": r["mean_gap_ms"], f"{n}_ttft_ms": r["ttft_ms"], f"{n}_done_ms": r["done_ms"]})
            summary.append(s)
            print(s)

path = labkit.write_run(LAB, "timeline", rows, {
    "question": "Which conclusions about running requests' mean gaps depend on the completion "
                "timeline and the step-cost model?",
    "workload": {"configuration": "Lesson 3.5 worked example; free_blocks=10, block_size=16"},
    "aggregated_metrics": {"scenarios": summary},
    "known_limitations": ["synthetic cost models, not measurements", "no chunked prefill"],
})
print(f"wrote {len(rows)} rows to {path}")
