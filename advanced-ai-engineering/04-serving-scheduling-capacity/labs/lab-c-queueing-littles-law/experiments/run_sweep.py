import json,pathlib,sys
import numpy as np
LAB=pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0,str(LAB.parent/"_shared")); import labkit
sys.path.insert(0,str(labkit.impl_dir(LAB))); import queueing as q
cfg=json.loads((LAB/"fixtures/sweep.json").read_text()); rows=[]
for rho in cfg["rho"]:
 for cv in cfg["cv"]:
  for seed in cfg["seeds"]:
   lam=rho*cfg["mu"]; a=q.poisson_arrivals(lam,cfg["horizon_s"],seed=seed); s=q.gamma_services(len(a),1/cfg["mu"],cv,seed=1000+seed); st,d=q.fifo_single_server(a,s); e=q.steady_estimates(a,st,d,cfg["horizon_s"],cfg["warmup_fraction"]); rows.append({"rho":rho,"cv":cv,"seed":seed,**e,"pk_Wq":q.pk_wq(lam,1/cfg["mu"],cv/cfg["mu"]) if rho<1 else None,"backlog_end":q.backlog(a,d,cfg["horizon_s"]-1e-9)})
means={str(r):float(np.mean([x["Wq"] for x in rows if x["rho"]==r])) for r in cfg["rho"]}
out=labkit.write_run(LAB,"utilization-sweep",rows,{"question":"Where does mean queue delay bend upward, and do matched finite-window boundaries satisfy Little's Law?","workload":cfg,"execution":{"warmup_iterations":cfg["warmup_fraction"],"measurement_iterations":len(rows),"total_duration_seconds":None},"raw_metrics":{"rows":len(rows)},"aggregated_metrics":{"mean_Wq_by_rho":means},"confidence":{"method":"independent-seed standard error; compute intervals in report","confidence_level":.95,"intervals":{}},"known_limitations":["M/G/1 single-server model; rho>=1 is transient and not a steady-state estimate."],"notes":"In-window residence is clipped at both measurement edges."}); print(out)
