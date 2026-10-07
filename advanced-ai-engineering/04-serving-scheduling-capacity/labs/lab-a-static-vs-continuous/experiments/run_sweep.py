import json, pathlib, sys
LAB = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "_shared")); import labkit
sys.path.insert(0, str(labkit.impl_dir(LAB))); import sim
cfg = json.loads((LAB / "fixtures/workload.json").read_text())
rows=[]
for sd in cfg["generation_sd"]:
  w=sim.make_workload(cfg["n"],seed=cfg["seed"]+sd,n_out_mean=200,n_out_sd=sd)
  for B in cfg["batch_sizes"]:
    for name,fn in (("static",sim.run_static),("continuous",sim.run_continuous)):
      r=fn(w,B,sim.LinearCost()); row={"policy":name,"batch":B,"n_out_sd":sd}
      row.update(sim.summarize(r,slo_ttft_ms=cfg["slo_ttft_ms"],slo_tpot_ms=cfg["slo_tpot_ms"])); rows.append(row)
out=labkit.write_run(LAB,"batch-sweep",rows,{"question":"How do batching policy, B, and output variance affect bubbles, latency, throughput, and goodput?","workload":cfg,"execution":{"warmup_iterations":0,"measurement_iterations":len(rows),"total_duration_seconds":None},"raw_metrics":{"rows":len(rows)},"aggregated_metrics":{},"confidence":{"method":"deterministic seeded sweep","confidence_level":None,"intervals":{}},"known_limitations":["Synthetic linear step cost; not a GPU measurement."],"notes":"Nearest-rank empirical quantiles."})
print(out)
