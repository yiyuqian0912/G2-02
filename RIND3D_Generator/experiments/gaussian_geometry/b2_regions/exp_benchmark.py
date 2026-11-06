"""Matched numerical exp experiment; this is not a physical-ray throughput test."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import gmpy2 as mp
import numpy as np
from ..study import atomic_json
from .api import extension
from .build import sources


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--points',type=int,default=65536);p.add_argument('--repeats',type=int,default=5);args=p.parse_args()
    rng=np.random.default_rng(20261106);n=args.points
    x=-np.r_[2.**rng.uniform(-12,10,n//2),rng.uniform(0,745,n-n//2)]
    x[:12]=[0,-np.nextafter(0.,1.),-1.,-.5,-32.,-700.,-744.,-745.,-746.,-1074.,-10000.,-1e308]
    module=extension();args.output.mkdir(parents=True,exist_ok=True)
    identity={'sources':sources(),'seed':20261106,'points':n,'repeats':args.repeats,
              'scope':'synthetic exact-binary64 nonpositive arguments, not physical ray speed',
              'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'arguments_SHA256':hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()}
    rows={};bounds={}
    for name,fast in [('MPFR53_enclosure',False),('polynomial_enclosure',True)]:
        times=[];CPUs=[]
        for _ in range(args.repeats):
            c=time.process_time();t=time.perf_counter();a,work=module.exponents(x,fast);times.append(time.perf_counter()-t);CPUs.append(time.process_time()-c)
        bounds[name]=a
        rows[name]={'median_wall_seconds':statistics.median(times),'median_CPU_seconds':statistics.median(CPUs),'wall_repeats':times,'work':work,
                    'relative_width_quantiles':np.percentile((a[:,1]-a[:,0])/np.maximum(a[:,1],np.nextafter(0.,1.)),[0,50,95,99,100]).tolist()}
    a=bounds['MPFR53_enclosure'];b=bounds['polynomial_enclosure']
    checked=np.unique(np.r_[np.arange(12),np.linspace(0,n-1,min(2048,n),dtype=int)])
    for i in checked:
        with mp.context(precision=256,round=mp.RoundDown):lo=mp.exp(mp.mpfr(float(x[i])))
        with mp.context(precision=256,round=mp.RoundUp):hi=mp.exp(mp.mpfr(float(x[i])))
        assert mp.mpfr(float(b[i,0]))<=lo<=hi<=mp.mpfr(float(b[i,1])),(i,x[i],b[i])
    middle=a[:,0]*.5+a[:,1]*.5;tau=np.maximum(middle*.8,np.nextafter(0.,1.));near=np.arange(n)%5==0
    tau[near]=np.maximum(middle[near],np.nextafter(0.,1.))
    statuses={}
    for name,fast,fallback in [('MPFR53_only',False,False),('polynomial_only',True,False),('MPFR53_fallback',False,True),('polynomial_fallback',True,True)]:
        times=[];CPUs=[]
        for _ in range(args.repeats):
            c=time.process_time();t=time.perf_counter();labels,work=module.exp_classes(x,tau,fast,fallback);times.append(time.perf_counter()-t);CPUs.append(time.process_time()-c)
        statuses[name]=np.array(labels);rows[name]={'median_wall_seconds':statistics.median(times),'median_CPU_seconds':statistics.median(CPUs),
                  'wall_repeats':times,'statuses':dict(Counter(labels)),'work':work}
    assert np.array_equal(statuses['MPFR53_fallback'],statuses['polynomial_fallback'])
    for name,labels in statuses.items():
        common=labels>=0
        assert np.array_equal(labels[common],statuses['MPFR53_fallback'][common]),name
    np.savez_compressed(args.output/'arguments.npz',arguments=x,thresholds=tau)
    atomic_json(args.output/'results.json',{'identity':identity,'modes':rows,'independent_256_bit_enclosures_checked':len(checked),
         'fallback_disagreements':0,'note':'Finite checks support the analytic remainder proof; do not establish a universal libm ULP bound.'})
    print(json.dumps({k:{kk:v for kk,v in r.items() if kk in ('median_wall_seconds','statuses')} for k,r in rows.items()}),flush=True)


if __name__=='__main__':main()
