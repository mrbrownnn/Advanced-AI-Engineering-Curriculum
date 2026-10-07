"""CPU MHA/GQA/MQA timing + geometry + RoPE boundary sweep. Append-only output."""
import json
import pathlib
import statistics
import sys
import time

LAB = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "_shared"))
import labenv
labenv.select_impl(LAB)

import torch
import yaml
import attention as A


def median_ms(fn, warmup, iterations):
    with torch.no_grad():
        for _ in range(warmup): fn()
        vals = []
        for _ in range(iterations):
            t = time.perf_counter_ns(); fn(); vals.append((time.perf_counter_ns() - t) / 1e6)
    return vals


def main():
    cfg = yaml.safe_load((LAB / "fixtures" / "sweep.yaml").read_text())
    torch.manual_seed(cfg["seed"]); torch.set_num_threads(1)
    rows = []
    total0 = time.perf_counter()
    for b in cfg["batch_sizes"]:
        for s in cfg["sequence_lengths"]:
            x = torch.randn(b, s, cfg["d"])
            for nkv in cfg["kv_heads"]:
                m = A.GQAAttention(cfg["d"], cfg["query_heads"], nkv).eval()
                vals = median_ms(lambda: m(x), cfg["warmup"], cfg["iterations"])
                rows.append({"event": "timing", "batch": b, "sequence": s, "n_kv": nkv,
                             "geometry": A.geometry(cfg["d"], cfg["query_heads"], nkv, b, s),
                             "latencies_ms": vals, "median_ms": statistics.median(vals)})
    q = torch.randn(1, 1, 1, cfg["d"] // cfg["query_heads"], dtype=torch.float64)
    k = torch.randn_like(q)
    for off in cfg["rope"]["offsets"]:
        base = A.relative_score(q, k, torch.tensor([0]), torch.tensor([off])).item()
        for shift in cfg["rope"]["shifts"]:
            score = A.relative_score(q, k, torch.tensor([shift]), torch.tensor([shift + off])).item()
            rows.append({"event": "rope", "offset": off, "shift": shift, "score": score, "shift_error": abs(score - base)})
    elapsed = time.perf_counter() - total0
    print(json.dumps(rows, indent=2))
    all_lat = [v for r in rows if r["event"] == "timing" for v in r["latencies_ms"]]
    labenv.write_result(LAB, "gqa-rope", rows, {
        "question": "Do fewer KV heads shrink declared geometry, and does CPU latency follow proportionally?",
        "workload": {"model": "synthetic GQAAttention", "dataset": "seeded random tensors", "configuration": cfg},
        "execution": {"warmup_iterations": cfg["warmup"], "measurement_iterations": cfg["iterations"], "total_duration_seconds": elapsed},
        "aggregated_metrics": {"latency_median_ms_all_shapes": statistics.median(all_lat)},
        "confidence": {"method": "raw repeated timings; descriptive only", "confidence_level": None, "intervals": {}},
        "known_limitations": ["CPU PyTorch backend only", "Random tiny weights; no model quality claim", "Timing order not randomized"],
        "notes": "Fewer logical KV elements need not produce proportional end-to-end latency.",
    })


if __name__ == "__main__": main()
