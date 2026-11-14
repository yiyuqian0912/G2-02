"""Final difficult fields and full regression after cache-integrity checks."""
import os
import subprocess
import sys
import time
from rind3d.pack import atomic_json
from .benchmark import ROOT


def main():
    jobs=[('fallback-study',['-m','experiments.cpu_cuda.fallback_study'],300),
          ('regression-final',['-m','pytest','-q','--junitxml='+str(ROOT/'regression-final.xml')],900)]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',RIND_TEST_CUDA='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    done=[]
    for name,args,limit in jobs:
        atomic_json(ROOT/'final-status.json',{'complete':False,'active':name,'completed':done})
        start=time.time()
        with (ROOT/(name+'.log')).open('a') as log:
            try:code=subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=limit).returncode
            except subprocess.TimeoutExpired:code='TIME_BUDGET'
        done.append({'name':name,'exit_code':code,'wall_seconds':time.time()-start})
        if code:atomic_json(ROOT/'final-status.json',{'complete':False,'failed':name,'completed':done});return
    atomic_json(ROOT/'final-status.json',{'complete':True,'completed':done})


if __name__=='__main__':main()
