"""Emit offline static trace evidence. Does not install/import Transformers or access network."""
import json,pathlib,sys,time
LAB=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LAB.parent/'_shared'))
import labenv
import yaml

def main():
    t=time.perf_counter(); data=yaml.safe_load((LAB/'fixtures'/'pinned-trace.yaml').read_text())
    rows=[{'event':'source','repository':data['repository'],'commit':data['commit'],'verification_date':data['verification_date'],'files':data['files'],'symbols':data['symbols']},
          {'event':'static_facts',**data['static_facts']},{'event':'execution_status',**data['execution_status']}]
    elapsed=time.perf_counter()-t; print(json.dumps(rows,indent=2))
    labenv.write_result(LAB,'static-trace',rows,{
      'question':'What does pinned Transformers source prove statically, and which backend/processor claims still require execution?',
      'workload':{'model':'none (source trace)','dataset':'pinned raw source observations','configuration':{'commit':data['commit']}},
      'execution':{'warmup_iterations':0,'measurement_iterations':1,'total_duration_seconds':elapsed},
      'aggregated_metrics':{'files_traced':len(data['files']),'symbols_traced':len(data['symbols'])},
      'confidence':{'method':'manual static verification against raw immutable GitHub URLs','confidence_level':None,'intervals':{}},
      'known_limitations':['Pinned Transformers package not executed','Selected backend and effective processor instances remain TODO_VERIFY'],
      'notes':data['execution_status']['note']})
if __name__=='__main__': main()
