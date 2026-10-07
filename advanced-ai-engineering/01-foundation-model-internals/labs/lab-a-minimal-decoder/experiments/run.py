"""Run solution/starter decoder invariants and sampling; append raw JSONL + result.yaml."""
import json
import pathlib
import sys
import time

LAB = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "_shared"))
import labenv
labenv.select_impl(LAB)

import torch
import yaml
import decoder as D
import sampling as SM


def main():
    fixture = yaml.safe_load((LAB / "fixtures" / "tiny.yaml").read_text())
    torch.manual_seed(fixture["seed"])
    model = D.TinyDecoder(D.Config(**fixture["config"])).eval().cpu()
    ids = torch.randint(0, model.cfg.vocab, (fixture["inputs"]["batch"], fixture["inputs"]["sequence"]))
    t0 = time.perf_counter()
    with torch.no_grad():
        full, _ = model(ids)
        prefix, cache = model(ids[:, :4])
        parts = [prefix]
        for i in range(4, ids.shape[1]):
            out, cache = model(ids[:, i:i + 1], cache)
            parts.append(out)
    elapsed = time.perf_counter() - t0
    inc = torch.cat(parts, 1)
    rows = [{"event": "parity", "max_abs_cache_error": (full - inc).abs().max().item(),
             "finite_logits": bool(torch.isfinite(full).all()), "elapsed_seconds": elapsed}]
    contract = SM.sampling_contract(do_sample=True, **fixture["sampling"])
    generator = torch.Generator(device="cpu").manual_seed(contract["seed"])
    generated = SM.generate(model, ids[:1], contract, generator)
    rows.append({"event": "sampling", "contract": contract, "new_token_ids": generated})
    print(json.dumps(rows, indent=2))
    labenv.write_result(LAB, "decoder", rows, {
        "question": "Does the transparent decoder preserve full-vs-cached parity and its explicit sampling contract?",
        "workload": {"model": "synthetic TinyDecoder", "dataset": "seeded token IDs", "configuration": fixture},
        "execution": {"warmup_iterations": 0, "measurement_iterations": 1, "total_duration_seconds": elapsed},
        "aggregated_metrics": {"max_abs_cache_error": rows[0]["max_abs_cache_error"]},
        "confidence": {"method": "deterministic invariant", "confidence_level": None, "intervals": {}},
        "known_limitations": ["Tiny random model; no quality claim", "One CPU process and one shape"],
        "notes": "FMA and wall-time performance are outside Lab A.",
    })


if __name__ == "__main__":
    main()
