"""Durable bounded experiment runner. Each later pilot keeps its own gate.

The plan is read between commands so pending experiments can be refined without
touching a running scientific job. No shell, mass-production or sibling edits.
"""
import json,os,subprocess,sys,time
from pathlib import Path
from rind3d.pack import atomic_json
from .audit import OUT

def run():
    plan=OUT/'campaign-plan.json';path=OUT/'campaign-status.json';state=json.loads(path.read_text()) if path.exists() else {'stages':[]};done={s['name'] for s in state['stages'] if s.get('passed')}
    while True:
        remaining=[s for s in json.loads(plan.read_text()) if s['name'] not in done]
        if not remaining:state['complete']=True;atomic_json(path,state);return
        job=remaining[0];log=OUT/(job['name']+'.campaign.log');start=time.perf_counter();state.update(complete=False,running=job['name'],started_unix=time.time());atomic_json(path,state)
        with log.open('ab',buffering=0) as f:r=subprocess.run([sys.executable,*job['args']],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
        stage={'name':job['name'],'args':job['args'],'returncode':r.returncode,'wall_seconds':time.perf_counter()-start,'passed':r.returncode==0,'log':str(log)}
        state['stages'].append(stage);state.pop('running',None);atomic_json(path,state)
        if r.returncode:raise RuntimeError('Campaign gate failed: '+job['name'])
        done.add(job['name'])

if __name__=='__main__':run()
