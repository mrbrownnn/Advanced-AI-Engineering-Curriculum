import json,pathlib,sys
LAB=pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0,str(LAB.parent/"_shared")); import labkit
sys.path[:0]=[str(labkit.impl_dir(LAB)),str(labkit.impl_dir(LAB,"lab-a-static-vs-continuous"))]; import sim,kvsched as k
cfg=json.loads((LAB/"fixtures/pressure.json").read_text()); w=sim.make_workload(cfg["n"],seed=cfg["seed"],n_in_mean=256,n_in_sd=64,n_out_mean=400,n_out_sd=100); cost=sim.LinearCost(base=10,per_decode=.1,per_prefill=.2); rows=[]
for blocks in cfg["pool_blocks"]:
 for mode in cfg["modes"]:
  for victim in cfg["victims"]:
   for admission in cfg["admission"]:
    pol=k.Policy(n_blocks=blocks,B=32,mode=mode,victim=victim,admission=admission,queue_limit=8 if admission=="reserve" else float("inf")); states,steps=k.run_kv(w,cost,pol); o=k.outcomes(states,steps,slo_ttft_ms=2000); base=1.5; rows.append({"blocks":blocks,"mode":mode,"victim":victim,"admission":admission,**o,"thrash_windows":len(k.detect_thrashing(steps,window_ms=2000,baseline_rate=base,drop=.7))})
out=labkit.write_run(LAB,"pressure-sweep",rows,{"question":"When do KV pressure and preemption reduce useful progress, and what utility does bounded admission preserve?","workload":cfg,"execution":{"warmup_iterations":0,"measurement_iterations":len(rows),"total_duration_seconds":None},"raw_metrics":{"rows":len(rows)},"aggregated_metrics":{},"confidence":{"method":"deterministic seeded sweep","confidence_level":None,"intervals":{}},"known_limitations":["Counter-only KV model; synthetic transfer and recompute costs; unbounded host swap pool."],"notes":"Module 03 Lab B block manager can replace counter."}); print(out)
