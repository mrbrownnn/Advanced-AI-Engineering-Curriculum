import json, math, pathlib, sys
LAB=pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0,str(LAB.parent/"_shared")); import labkit
sys.path[:0]=[str(labkit.impl_dir(LAB)),str(labkit.impl_dir(LAB,"lab-a-static-vs-continuous"))]
import sim,chunked
cfg=json.loads((LAB/"fixtures/mixed_workload.json").read_text()); w=sim.make_workload(cfg["n"],seed=cfg["seed"],rate_per_s=cfg["rate_per_s"],n_in_mean=cfg["n_in_mean"],n_in_sd=cfg["n_in_sd"],n_out_mean=cfg["n_out_mean"],n_out_sd=cfg["n_out_sd"]); cost=sim.LinearCost(base=10,per_decode=.1,per_prefill=.02,per_chunk=1); rows=[]
for raw in cfg["chunks"]:
 C=math.inf if raw=="infinity" else raw; r=chunked.run_chunked(w,16,cost,chunk=C); row={"chunk":raw,"steps":len(r.steps),"prefill_cost_ms":sum(x["dt"] for x in r.steps if x["n_prefill"]),"decode_cost_ms":sum(x["dt"] for x in r.steps if x["n_decode"])}; row.update(sim.summarize(r,slo_ttft_ms=2000,slo_tpot_ms=50)); rows.append(row)
out=labkit.write_run(LAB,"chunk-sweep",rows,{"question":"How does chunk size trade decode ITL tail against prefill TTFT and overhead?","workload":cfg,"execution":{"warmup_iterations":0,"measurement_iterations":len(rows),"total_duration_seconds":None},"raw_metrics":{"rows":len(rows)},"aggregated_metrics":{},"confidence":{"method":"deterministic seeded sweep","confidence_level":None,"intervals":{}},"known_limitations":["Synthetic step-cost model; no kernel overlap."],"notes":"Chunk 32 is the BREAK case; infinity is unchunked."}); print(out)
