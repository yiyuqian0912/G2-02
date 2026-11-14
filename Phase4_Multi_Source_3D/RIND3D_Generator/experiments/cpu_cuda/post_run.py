"""Run the two extra attribution controls after performance/regression jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from rind3d.pack import atomic_json

ROOT=Path('outputs/cpu-cuda-efficiency')


def main():
    started=time.time()
    while time.time()-started<3600:
        state=json.loads((ROOT/'run-status.json').read_text())
        if state.get('failed'):
            atomic_json(ROOT/'post-status.json',{'complete':False,'blocked_by_job_failure':state['failed']});return
        if state.get('complete'):break
        time.sleep(2)
    else:raise RuntimeError('Main coordinator wait budget exhausted')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    tasks=[('fair-io','experiments.cpu_cuda.fair_io',180),('native-ablation','experiments.cpu_cuda.reuse_ablation',180)]
    rows=[]
    for name,module,limit in tasks:
        atomic_json(ROOT/'post-status.json',{'complete':False,'active':name,'completed':rows})
        start=time.time()
        with (ROOT/(name+'-job.log')).open('a') as log:
            p=subprocess.run([sys.executable,'-m',module],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=limit)
        rows.append({'name':name,'exit_code':p.returncode,'wall_seconds':time.time()-start})
        if p.returncode:atomic_json(ROOT/'post-status.json',{'complete':False,'failed':name,'completed':rows});return
    atomic_json(ROOT/'post-status.json',{'complete':True,'completed':rows})


if __name__=='__main__':main()
