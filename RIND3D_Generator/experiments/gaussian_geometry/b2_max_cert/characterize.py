"""Freeze trace-stratified rays and profile the optimized unchanged baseline.

Floating-point extrema below are selection/diagnostic heuristics ONLY. Every
physical expected label comes from a complete previously certified field.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar

from ..models import load,record
from ..b2_native.api import NativeOracle
from ..study import atomic_json
from .profile_baseline import extension


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def approximate(scene,source,receivers,steps=513):
    t=np.linspace(0,1,steps);result=[]
    tau=scene.threshold.value if scene.threshold.kind=='binary64' else np.exp(scene.threshold.value)
    for receiver in receivers:
        d=receiver-source
        u=np.array([k.approx_inverse@(source-k.center) for k in scene.kernels]);v=np.array([k.approx_inverse@d for k in scene.kernels])
        a=np.sum(v*v,axis=1);b=2*np.sum(u*v,axis=1);c=np.sum(u*u,axis=1);w=np.array([k.weight for k in scene.kernels])
        def value(x):return float(np.sum(w*np.exp(-np.maximum(0,(a*x+b)*x+c)/2)))
        f=np.sum(w[:,None]*np.exp(-np.maximum(0,(a[:,None]*t+b[:,None])*t+c[:,None])/2),axis=0)
        peaks=np.flatnonzero((f[1:-1]>f[:-2])&(f[1:-1]>=f[2:]))+1
        maximum=max(float(f[0]),float(f[-1]));at=float(t[np.argmax(f)])
        for j in peaks:
            opt=minimize_scalar(lambda x:-value(x),bounds=(t[j-1],t[j+1]),method='bounded',options={'xatol':1e-14})
            if -opt.fun>maximum:maximum=-float(opt.fun);at=float(opt.x)
        active=w*np.exp(-np.maximum(0,(a*at+b)*at+c)/2)
        result.append({'approx_max':maximum,'approx_tau_margin':float(tau-maximum),'approx_argmax':at,
                       'approx_local_peaks':len(peaks),'approx_active_kernels_1percent':int(np.sum(active>=max(active)*.01)),
                       'selection_values_are_not_certificates':True})
    return result


def freeze(output,inputs,root):
    rows=[];sources={};identities=[]
    for ordinal in (14,7,0,21):
        case=f'gaussian_sum-{ordinal:03d}';scene=load(inputs/case/'geometry.json')
        source=np.array(json.loads((inputs/case/'result.json').read_text())['source_xyz_synthetic'])
        n=128 if ordinal in (14,7) else 64
        directory=root/('depth7-scales-v3' if n==128 else 'four64-v1')
        method='A_fixed7' if n==128 else 'A_fixed6'
        archive=directory/f'{case}-{n}-{method}-independent.npz';arrays=np.load(archive)
        path=arrays['evaluation_path'].ravel();cost=arrays['ray_nanoseconds'].ravel();shape=(n,n,n)
        chosen={}
        def add(indices,group):
            for i in indices:chosen.setdefault(int(i),[]).append(group)
        miss=np.flatnonzero(path==6);order=miss[np.argsort(cost[miss],kind='stable')]
        add(order[-24:],'expensive_MISS')
        mid=len(order)//2;add(order[max(0,mid-12):mid+12],'median_MISS');add(order[:16],'cheap_MISS')
        pool=miss[np.linspace(0,len(miss)-1,min(512,len(miss)),dtype=int)]
        xx=[(np.array(np.unravel_index(int(i),shape))[::-1]+.5)/n for i in pool]
        values=approximate(scene,source,xx)
        margin=np.array([v['approx_tau_margin'] for v in values]);add(pool[np.argsort(np.abs(margin),kind='stable')[:16]],'near_threshold_MISS_heuristic')
        hit=np.flatnonzero(path==7);horder=hit[np.argsort(cost[hit],kind='stable')]
        add(horder[-8:],'expensive_HIT');add(horder[np.linspace(0,len(horder)-1,min(8,len(horder)),dtype=int)],'ordinary_HIT')
        selected=sorted(chosen);receivers=[(np.array(np.unravel_index(i,shape))[::-1]+.5)/n for i in selected]
        approx=approximate(scene,source,receivers)
        for i,x,est in zip(selected,receivers,approx):
            identity=hashlib.sha256(np.array([*source,*x],dtype='<f8').tobytes()+record(scene)['geometry_id'].encode()).hexdigest()
            rows.append({'ray_id':identity,'case':case,'resolution':n,'linear_ZYX_index':i,'source_xyz':source.tolist(),'receiver_xyz':x.tolist(),
                         'source_hex':[float(v).hex() for v in source],'receiver_hex':[float(v).hex() for v in x],
                         'expected':'MISS' if path[i]==6 else 'HIT','strata':chosen[i],'saved_ray_nanoseconds':int(cost[i]),
                         'split':'development' if int(identity[-8:],16)%3==0 else 'evaluation',**est})
        sources[case]=source.tolist();identities.append({'case':case,'geometry_id':record(scene)['geometry_id'],'geometry_file_SHA256':sha(inputs/case/'geometry.json'),
                                                       'source_file_SHA256':sha(inputs/case/'result.json'),'trace_archive':str(archive),'trace_archive_SHA256':sha(archive),
                                                       'kernels':len(scene.kernels),'kernel_scales_diagnostic':[float(np.linalg.norm(k.matrix)) for k in scene.kernels]})
    atomic_json(output/'frozen-rays.json',{'schema':'b2-max-ray-fixtures-v1','selection':'Trace cost strata plus approximate-margin stratum; frozen before solver implementation','scenes':identities,'rays':rows})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('outputs/gaussian-b2-max-cert-v1'))
    p.add_argument('--input',type=Path,default=Path('outputs/gaussian-geometry-v1/distribution-v2'))
    p.add_argument('--traces',type=Path,default=Path('outputs/gaussian-b2-angular-v1'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    frozen=a.output/'frozen-rays.json';rows=json.loads(frozen.read_text())['rays'] if frozen.exists() else freeze(a.output,a.input,a.traces)
    m=extension(a.output/'profile-baseline-v2');results=[]
    for case in sorted({r['case'] for r in rows}):
        scene=load(a.input/case/'geometry.json');params=([k.center.tolist() for k in scene.kernels],[k.matrix.ravel().tolist() for k in scene.kernels],[k.weight for k in scene.kernels],scene.threshold.value,scene.threshold.kind=='exp')
        prof=m.Independent(*params);options=m.Options();options.max_nodes=16384
        old=NativeOracle(scene,independent=True,max_nodes=16384);primary=NativeOracle(scene,max_nodes=16384)
        for row in (r for r in rows if r['case']==case):
            s,x=row['source_xyz'],row['receiver_xyz'];t=time.perf_counter();c=time.process_time();b=old.segment(s,x);wall=time.perf_counter()-t;cpu=time.process_time()-c
            t=time.perf_counter();pr=primary.segment(s,x);pwall=time.perf_counter()-t
            t=time.perf_counter();decisions,stats=prof.rays(s,np.array([x]),options);profile_wall=time.perf_counter()-t;d=decisions[0]
            label={0:'MISS',1:'HIT',-1:'UNKNOWN',-2:'INVALID'}[d['status']]
            assert b.status==label==pr.status==row['expected'],(row,b,label,pr)
            trace=np.array(m.visits(),dtype=float).reshape(-1,4)
            np.savez_compressed(a.output/f'visits-{row["ray_id"]}.npz',visits=trace)
            result={'ray_id':row['ray_id'],'case':case,'split':row['split'],'expected':row['expected'],'strata':row['strata'],
                    'independent_seconds':wall,'independent_CPU_seconds':cpu,'primary_seconds':pwall,'profile_seconds':profile_wall,
                    'baseline_nodes':b.nodes,'baseline_depth':b.depth,'profile_work':dict(stats),
                    'natural_free_width_histogram':dict(Counter(str(int(round(-np.log2(v[1]-v[0])))) for v in trace if v[3]==0 and v[2]==0 and v[1]>v[0])),
                    'profile_matches_original':True}
            results.append(result)
        print(json.dumps({'case':case,'queries':sum(r['case']==case for r in rows),'baseline_seconds':sum(r['independent_seconds'] for r in results if r['case']==case)}),flush=True)
    atomic_json(a.output/'baseline-characterization.json',{'frozen_fixture_SHA256':sha(frozen),'results':results,'scope':'Timing-only baseline copy, unchanged physical expressions; diagnostic scalar traces, not new labels'})


if __name__=='__main__':main()
