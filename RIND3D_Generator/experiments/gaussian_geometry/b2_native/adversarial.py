"""Matched B2 query stresses; approximate points only propose exact queries."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..models import Kernel,KernelScene,Threshold,save
from ..queries import FieldOracle,Limits
from ..study import atomic_json
from .api import NativeOracle
from .controls import bridge
from .build import sources


def fixtures():
    rng=np.random.default_rng(8301)
    yield 'fused_branch',bridge(),[.03125]*3,None
    yield 'thin_branch',bridge(.125),[.03125]*3,None
    dense=KernelScene('gaussian_sum',[Kernel(c,np.eye(3)*rng.uniform(.08,.14),float(w)) for c,w in zip(rng.uniform(.37,.63,(48,3)),rng.uniform(.5,1.5,48))],Threshold('binary64',8.))
    yield 'dense_overlap',dense,[.03125]*3,None
    disconnected=KernelScene('gaussian_sum',[Kernel(c,np.eye(3)*.016) for c in rng.uniform(.18,.82,(32,3))],Threshold('binary64',.65))
    yield 'dispersed_32',disconnected,[.03125]*3,None
    m=np.array([[.11,.1099999,0],[0,2**-20,.000001],[0,0,.04]])
    thin=KernelScene('gaussian_sum',[Kernel([.45,.5,.5],m),Kernel([.55,.5,.5],m*.65)],Threshold('binary64',.45))
    yield 'thin_rotated_sheared',thin,[.03125]*3,None
    scales=KernelScene('gaussian_sum',[Kernel([.5,.5,.5],np.diag([.12,.08,.04]),1.),
                Kernel([.57,.5,.5],np.diag([2**-18,.02,.01]),.7),Kernel([.42,.53,.5],np.eye(3)*.03,2**-80)],Threshold('binary64',.5))
    yield 'multiscale_and_small_amplitude',scales,[.03125]*3,None
    tangent=KernelScene('gaussian_sum',[Kernel([x,.5,.5],np.eye(3)*.125,.5) for x in (.4375,.5625)],Threshold('exp',-.25))
    # Separate sources keep each source/receiver pair under identical y.
    for name,y in [('exact',.5625),('outside',np.nextafter(.5625,np.inf)),('inside',np.nextafter(.5625,-np.inf))]:
        yield 'tangent_'+name,tangent,[.5,y,.03125],np.array([[.5,y,.96875]])


def receivers(scene,seed):
    rng=np.random.default_rng(seed);old=FieldOracle(scene);points=[]
    for p in rng.random((64,3)):
        if old.membership(p).status=='FREE':points.append(p)
        if len(points)>=32:break
    # Approximate bisection merely generates difficult proposed receivers;
    # every retained receiver then gets an exact FREE membership decision.
    for i in range(16):
        inside=scene.kernels[i%len(scene.kernels)].center.copy();outside=rng.random(3)
        if scene.approx_values([inside])[0]<=0 or scene.approx_values([outside])[0]>=0:continue
        for _ in range(24):
            middle=(inside+outside)/2
            if scene.approx_values([middle])[0]>=0:inside=middle
            else:outside=middle
        for eps in (2**-14,2**-22):
            p=outside+eps*(outside-inside)/max(np.linalg.norm(outside-inside),1e-30)
            if np.all((p>=0)&(p<=1)) and old.membership(p).status=='FREE':points.append(p)
    return np.array(points)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    identity={'native_sources':sources(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'nodes':512,'seconds_per_query':.5,'depth':40,'source_distribution':'declared deterministic valid synthetic sources'}
    atomic_json(args.output/'manifest.json',identity);rows=[]
    for ordinal,(name,scene,source,points) in enumerate(fixtures()):
        path=args.output/name;path.mkdir(exist_ok=True);save(path/'geometry.json',scene)
        if points is None:points=receivers(scene,8301+ordinal)
        old=FieldOracle(scene,Limits(max_nodes=512,seconds=.5))
        assert old.source_clearance(source).status=='FREE'
        assert all(old.membership(p).status=='FREE' for p in points)
        np.savez_compressed(path/'queries.npz',source=source,receivers=points)
        engines={'python_Arb':old,'native_without_Taylor':NativeOracle(scene,taylor=False,exact_depth=51,max_nodes=512,query_seconds=.5),
                 'native_Taylor':NativeOracle(scene,max_nodes=512,query_seconds=.5),
                 'native_groups':NativeOracle(scene,groups=True,max_nodes=512,query_seconds=.5),
                 'independent_plain':NativeOracle(scene,independent=True,taylor=False,tail_bounds=False,max_nodes=512,query_seconds=.5),
                 'independent_optimized':NativeOracle(scene,independent=True,max_nodes=512,query_seconds=.5)}
        row={'fixture':name,'rays':len(points),'kernel_count':len(scene.kernels),'backends':{},'query_fixture_only':True};outputs={}
        for backend,e in engines.items():
            e.work.clear();timings=[];ds=[]
            for point in points:
                t=time.perf_counter();d=e.segment(source,point);timings.append(time.perf_counter()-t);ds.append(d)
            row['backends'][backend]={'wall_seconds':sum(timings),'scalar_seconds':dict(zip(('p50','p95','p99','max'),map(float,np.percentile(timings,[50,95,99,100])))),
                  'statuses':dict(Counter(d.status for d in ds)),'reasons':dict(Counter(d.reason for d in ds)),'work':dict(e.work),
                  'unknown_examples':[{'receiver_xyz':p.tolist(),'decision':asdict(d)} for p,d in zip(points,ds) if not d.certified][:4]}
            outputs[backend]=ds
        disagree=0
        for i in range(len(points)):
            labels={d[i].status for d in outputs.values() if d[i].certified}
            if len(labels)>1:disagree+=1
        if name=='tangent_exact':
            # Original v1 names budget exhaustion separately; it is still a
            # non-physical result, not a MISS or a contradictory classification.
            assert all(d[0].status in ('HIT','UNKNOWN','BUDGET_EXHAUSTED') for d in outputs.values())
        if disagree:raise AssertionError('Adversarial physical disagreement: '+name)
        row['certified_disagreements']=disagree;atomic_json(path/'result.json',row);rows.append(row)
        print(json.dumps({'fixture':name,'rays':len(points),'unknowns':{k:v['statuses'].get('UNKNOWN',0)+v['statuses'].get('BUDGET_EXHAUSTED',0) for k,v in row['backends'].items()}}),flush=True)
    atomic_json(args.output/'summary.json',{'cases':rows,'identity':identity,'production':False})


if __name__=='__main__':main()
