"""Bounded durable coordinator, sequential performance jobs and regression logs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from rind3d.pack import atomic_json
from .benchmark import ROOT


def main():
    tasks=[('full',['-m','experiments.cpu_cuda.evaluate','full'],1500),
           ('reuse-1024-cpu',['-m','experiments.cpu_cuda.evaluate','reuse','--method','CPU_clip'],360),
           ('reuse-1024-gpu',['-m','experiments.cpu_cuda.evaluate','reuse','--method','GPU'],360),
           ('reuse-poly16-cpu',['-m','experiments.cpu_cuda.evaluate','reuse','--case','poly16','--resolution','128','--method','CPU_clip'],180),
           ('reuse-poly16-gpu',['-m','experiments.cpu_cuda.evaluate','reuse','--case','poly16','--resolution','128','--method','GPU'],180),
           ('lite-cpu',['-m','experiments.cpu_cuda.lite_end_to_end','--method','CPU_clip'],180),
           ('lite-gpu',['-m','experiments.cpu_cuda.lite_end_to_end','--method','GPU'],180),
           ('canonical-targets',['-m','experiments.cpu_cuda.canonical_audit','targeted'],180),
           ('canonical-full-attempt',['-m','experiments.cpu_cuda.canonical_audit','exhaustive'],90),
           ('regression',['-m','pytest','-q','--junitxml='+str(ROOT/'regression.xml')],900)]
    completed=[];manifest=ROOT/'run-status.json';before=time.time()
    with (ROOT/'gpu-hardware-final.txt').open('w') as f:subprocess.run(['nvidia-smi','-q','-d','POWER,CLOCK,MEMORY,PERFORMANCE'],stdout=f,check=True)
    with (ROOT/'gpu-utilization-durable.csv').open('w') as gpu_log:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem','--format=csv','-lms','200'],stdout=gpu_log)
        try:
            for name,args,limit in tasks:
                old=ROOT/(name+'-job.json')
                if old.exists() and json.loads(old.read_text()).get('exit_code')==0:
                    completed.append(json.loads(old.read_text()));continue
                row={'name':name,'args':args,'start_unix':time.time(),'budget_seconds':limit,'complete':False}
                atomic_json(manifest,{'pid':os.getpid(),'completed':completed,'active':row,'complete':False,'started_unix':before})
                print('START',name,flush=True)
                with (ROOT/(name+'-job.log')).open('a') as log:
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',RIND_TEST_CUDA='1')
                    try:result=subprocess.run([sys.executable,*args],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=limit);code=result.returncode
                    except subprocess.TimeoutExpired:code='TIME_BUDGET'
                row.update(end_unix=time.time(),exit_code=code,complete=True);completed.append(row);atomic_json(old,row)
                print('END',name,code,flush=True)
                if code!=0:
                    atomic_json(manifest,{'pid':os.getpid(),'completed':completed,'active':None,'complete':False,'failed':name});return
        finally:monitor.terminate();monitor.wait()
    atomic_json(manifest,{'pid':os.getpid(),'completed':completed,'active':None,'complete':True,'started_unix':before,'ended_unix':time.time()})


if __name__=='__main__':main()
