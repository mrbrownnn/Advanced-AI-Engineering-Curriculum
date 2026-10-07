"""Lab D experiment: sweep representation, retained tokens, bandwidth, setup, overlap.

Writes results/<stamp>-bounds/{raw.jsonl,result.yaml}. Every number is a screening
bound under declared assumptions, not a measurement.
    python experiments/run_bounds.py
    LAB_IMPL=solution python experiments/run_bounds.py
"""
import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "_shared"))
sys.path.insert(0, str(LAB / "fixtures"))
import labkit  # noqa: E402

labkit.use_impl(LAB)
import make_eval  # noqa: E402
from bounds import (GIB, Bandwidth, capacity_claim, concurrency_bound, critical_path_s,  # noqa: E402
                    payload_bytes_per_token, quality_gate, retained_fraction_scores, scale_bytes_per_token)

MHA = dict(layers=32, kv_heads=32, head_dim=128)
BUDGET, TOKENS = 63 * GIB, 4096
data = make_eval.items(200, seed=0)
base = [s for s, _ in data]
rows = []

for bits, group in ((16, None), (8, 16), (8, 64), (4, 16), (4, 1)):
    per_tok = payload_bytes_per_token(**MHA, bits=bits) + (scale_bytes_per_token(32, 32, 4, group) if group else 0)
    for keep in (1.0, 0.75, 0.5):
        g = quality_gate(base, retained_fraction_scores(data, keep), max_drop=0.02)
        bound = concurrency_bound(BUDGET, int(TOKENS * keep), per_tok)
        rows.append({"kind": "capacity", "bits": bits, "scale_group_tokens": group, "keep_fraction": keep,
                     "bytes_per_token": per_tok, "bound_requests": bound, "gate_passed": g.passed,
                     "gate_mean_drop": g.metric, "surviving_claim": capacity_claim(bound, g)})

per_tok16 = payload_bytes_per_token(**MHA, bits=16)
for tokens in (16, 256, 4096, 16384):
    for bw in (Bandwidth(50, "GiB/s"), Bandwidth(25, "GB/s")):
        for setup in (0.0, 0.0005, 0.002):
            for overlap in (0.0, 0.5, 0.9):
                p = tokens * per_tok16
                rows.append({"kind": "transfer", "tokens": tokens, "bandwidth": f"{bw.value} {bw.unit}",
                             "setup_s": setup, "overlap_fraction": overlap,
                             "critical_path_ms": 1000 * critical_path_s(p, bw, setup, overlap)})

path = labkit.write_run(LAB, "bounds", rows, {
    "question": "When does a nominal payload reduction survive metadata, quality, setup and overlap?",
    "workload": {"configuration": "Lesson 3.1 MHA config; 63 GiB KV budget; 4096-token requests",
                 "dataset": "fixtures/make_eval.items(200, seed=0) synthetic gate stub"},
    "known_limitations": ["analytical bounds only; no kernel, dequantization or contention measurement",
                          "quality gate uses a synthetic retention stub, not a model eval"],
})
print(f"wrote {len(rows)} rows to {path}")
