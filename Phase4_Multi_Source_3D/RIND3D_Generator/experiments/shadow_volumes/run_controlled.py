"""Finite follow-up matrix. No production generation or READY publication."""
import json,os,subprocess,sys,time
from pathlib import Path
from rind3d.pack import atomic_json
from .benchmark import ROOT


def main():
    jobs=[
        ['stratified','--case','poly8'],['stratified','--case','poly32'],
        ['full','--case','poly16'],['full','--case','multiscale4'],
        ['stratified','--case','corridor'],['full','--case','poly8'],['full','--case','poly32'],
        ['full','--case','corridor'],['full','--case','corridor','--method','B'],
        ['full','--case','corridor','--method','B_native'],['full','--case','box','--method','A'],
        ['reuse','--case','rotated_box'],['reuse','--case','poly16','--resolution','128'],
        ['reuse','--case','multiscale4','--resolution','128'],
    ]
    logs=ROOT/'controlled-jobs';logs.mkdir(exist_ok=False);running=[];done=[];start=time.perf_counter();index=0
    while jobs or running:
        while jobs and len(running)<2:
            args=jobs.pop(0);index+=1;handle=(logs/f'{index:02d}-{args[0]}-{args[2]}.log').open('w')
            process=subprocess.Popen([sys.executable,'-m','experiments.shadow_volumes.evaluate',*args],stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            running.append((args,process,handle,time.perf_counter()))
        for row in running[:]:
            args,p,handle,t=row;code=p.poll();aborted=False
            if code is None and time.perf_counter()-t>600:p.kill();code=p.wait();aborted=True
            if code is None:continue
            handle.close();running.remove(row);done.append({'args':args,'exit_code':code,'external_budget_abort':aborted,'wall_seconds':time.perf_counter()-t});print(done[-1],flush=True)
        atomic_json(ROOT/'controlled-progress.json',{'done':done,'remaining':jobs,'running':[r[0] for r in running],'elapsed_seconds':time.perf_counter()-start})
        if running:time.sleep(.5)


if __name__=='__main__':main()
