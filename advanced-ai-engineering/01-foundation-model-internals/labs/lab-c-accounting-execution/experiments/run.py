"""Reconcile analytical counts, profiler FLOPs/shapes, KV tensors, and CPU timing."""
import json,pathlib,statistics,sys,time
LAB=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LAB.parent/'_shared'))
import labenv; labenv.select_impl(LAB)
import torch,yaml
from torch.profiler import profile,ProfilerActivity
from torch.utils.flop_counter import FlopCounterMode
import accounting as A


def main():
    spec=yaml.safe_load((LAB/'fixtures'/'sweep.yaml').read_text()); torch.manual_seed(spec['seed']); torch.set_num_threads(1)
    rows=[]; total0=time.perf_counter()
    for ci,raw in enumerate(spec['configs']):
        c=A.Config(**raw); m=A.TinyModel(c).eval()
        rows.append({'event':'parameters','config_id':ci,'analytical':A.analytical_parameter_count(c),'actual_unique':A.actual_parameter_count(m),
                     'actual_named_with_aliases':A.actual_parameter_count(m,False),'manifest':A.parameter_manifest(c)})
        for work in spec['workloads']:
            ids=torch.randint(0,c.vocab,(work['batch'],work['sequence']))
            with FlopCounterMode(display=False) as fc,torch.no_grad(): m(ids)
            with profile(activities=[ProfilerActivity.CPU],record_shapes=True,profile_memory=True) as p:
                with torch.no_grad(): m(ids)
            for _ in range(spec['warmup']): m(ids)
            vals=[]
            for _ in range(spec['iterations']):
                t=time.perf_counter_ns(); m(ids); vals.append((time.perf_counter_ns()-t)/1e6)
            events=[{'key':e.key,'input_shapes':e.input_shapes,'self_cpu_time_us':e.self_cpu_time_total,'self_cpu_memory_bytes':e.self_cpu_memory_usage} for e in p.key_averages(group_by_input_shape=True) if e.self_cpu_time_total>0]
            rows.append({'event':'execution','config_id':ci,**work,'logical':A.logical_flops(c,**work),'profiler_flops':fc.get_total_flops(),
                         'logical_kv_bytes_float32':A.kv_bytes(c,**work,dtype_bytes=4),'latencies_ms':vals,'median_ms':statistics.median(vals),'profiler_events':events})
    mla=spec['mla_pinned_config']; rows.append({'event':'mla','source_revision':mla['revision'],**A.mla_retained_elements(mla['num_hidden_layers'],mla['kv_lora_rank'],mla['qk_rope_head_dim']),
      'rejected_gqa_elements':2*mla['num_hidden_layers']*mla['num_key_value_heads']*mla['v_head_dim']})
    elapsed=time.perf_counter()-total0; print(json.dumps(rows,indent=2))
    lats=[v for r in rows if r['event']=='execution' for v in r['latencies_ms']]
    labenv.write_result(LAB,'accounting',rows,{
      'question':'Where do exact architecture counts agree with instantiated tensors, and where does CPU execution depart from logical work?',
      'workload':{'model':'synthetic TinyModel','dataset':'seeded token IDs','configuration':spec},
      'execution':{'warmup_iterations':spec['warmup'],'measurement_iterations':spec['iterations'],'total_duration_seconds':elapsed},
      'aggregated_metrics':{'latency_mean_ms':statistics.mean(lats),'latency_std_ms':statistics.stdev(lats) if len(lats)>1 else 0},
      'confidence':{'method':'raw repeated timings; descriptive only','confidence_level':None,'intervals':{}},
      'known_limitations':['CPU RSS/allocator peak is not reported by torch.profiler','Profiler shapes are operator-level, not fused-kernel contracts','Tiny synthetic configurations'],
      'notes':'Logical FLOPs use FMA=2. Profiler records real operator shapes and self CPU memory, not reserved allocator memory.'})
if __name__=='__main__': main()
