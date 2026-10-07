"""Lab A experiment: waste categories vs capacity, policy, block size.

Writes results/<stamp>-waste-sweep/{raw.jsonl,result.yaml}. Run from the lab dir:
    python experiments/run_waste_sweep.py            # your starter
    LAB_IMPL=solution python experiments/run_waste_sweep.py
`failures` counts blocked admission attempts (one per step the queue head is refused).
"""
import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "_shared"))
sys.path.insert(0, str(LAB / "fixtures"))
import labkit  # noqa: E402

labkit.use_impl(LAB)
import make_trace  # noqa: E402
from allocator import Request, simulate  # noqa: E402

rows = []
for seed in range(5):
    trace = make_trace.heterogeneous(n=40, seed=seed)
    for capacity in (4096, 6000, 8192, 16384):
        for policy in ("max_len", "exact"):
            for bs in (1, 16):
                out = simulate([Request(**r) for r in trace], capacity, policy, bs)
                s = out["snapshots"]
                alloc = sum(x["allocated"] for x in s) or 1
                rows.append({
                    "seed": seed, "capacity": capacity, "policy": policy, "block_size": bs,
                    "steps": len(s), "blocked_capacity": out["failures"]["capacity"],
                    "blocked_contiguous": out["failures"]["contiguous"],
                    "share_logical": sum(x["logical"] for x in s) / alloc,
                    "share_internal": sum(x["internal_fragmentation"] for x in s) / alloc,
                    "share_reservation": sum(x["reservation_waste"] for x in s) / alloc,
                    "mean_running": sum(x["running"] for x in s) / len(s),
                })

path = labkit.write_run(LAB, "waste-sweep", rows, {
    "question": "How do reservation waste, internal fragmentation and contiguous failure separate "
                "across capacity, reservation policy and block size?",
    "workload": {"dataset": "fixtures/make_trace.heterogeneous n=40 seeds 0-4", "unit": "token slots"},
    "known_limitations": ["synthetic trace; token-slot address space, not bytes",
                          "one token per step; no prefill cost model"],
})
print(f"wrote {len(rows)} rows to {path}")
