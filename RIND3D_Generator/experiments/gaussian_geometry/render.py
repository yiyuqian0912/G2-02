"""Bounded complete SMALL fields. No production renderer/RINDPack changes."""
from itertools import product
import time

import numpy as np

from .queries import ConvexOracle,point


def small_field(oracle,source,resolution,seconds=45.,*,independent=None):
    if resolution not in (8,16,32,64,128):
        raise ValueError('Research gate: small fields only; no 1024³ execution')
    n=resolution;began=time.perf_counter();cpu=time.process_time()
    before=dict(oracle.work)
    if isinstance(oracle,ConvexOracle):
        renderer=oracle.context.bind(source,independent=bool(independent))
        native_before=dict(renderer.work)
        y=np.zeros((n,)*3,dtype=bool);m=np.zeros_like(y);valid=np.zeros_like(y);covered=0
        side=min(64,n)
        for z,j,i in product(range(0,n,side),repeat=3):
            if time.perf_counter()-began>seconds:break
            # Native API is a <=64³ brick renderer, with an independent world
            # resolution. Assemble the full small field without changing it.
            origin=(i,j,z);a,b,count=renderer.render(origin,side,n)
            region=(slice(z,z+side),slice(j,j+side),slice(i,i+side))
            y[region]=a;m[region]=b;valid[region]=True;covered+=count
        complete=bool(valid.all())
        return {'response':y,'solid':m,'valid':valid,
                'complete':complete,'status':'COMPLETE' if complete else 'TIME_BUDGET_EXHAUSTED','wall_seconds':time.perf_counter()-began,
                'cpu_seconds':time.process_time()-cpu,'certificate_coverage':covered/n**3,
                'valid_voxels':int(valid.sum()),'native_max_brick_side':64,
                'work':{k:v-native_before.get(k,0) for k,v in renderer.work.items()},'independent':bool(independent)}
    y=np.zeros((n,)*3,dtype=bool);mask=np.zeros_like(y);valid=np.zeros_like(y)
    stack=[((0,0,0),n,-1,-1)];filled=0;status='COMPLETE';unresolved=[]
    while stack:
        if time.perf_counter()-began>seconds:
            status='TIME_BUDGET_EXHAUSTED';break
        origin,side,known_y,known_m=stack.pop();o=np.array(origin)
        lo=(o+.5)/n;hi=(o+side-.5)/n
        if independent is None:
            cy,cm=oracle.certificate(source,lo,hi)
            if known_y>=0:cy=known_y
            if known_m>=0:cm=known_m
        else:
            # Reference intentionally uses no primary support/region/proxy.
            cy,cm=-1,-1
        x,zx,zz=origin;region=(slice(zz,zz+side),slice(zx,zx+side),slice(x,x+side))
        if cy>=0 and cm>=0:
            y[region]=bool(cy);mask[region]=bool(cm);valid[region]=True;filled+=side**3;continue
        if side>2 and independent is None:
            half=side//2
            for offset in product((0,half),repeat=3):
                stack.append((tuple(a+b for a,b in zip(origin,offset)),half,cy,cm))
            continue
        for z in range(origin[2],origin[2]+side):
            for j in range(origin[1],origin[1]+side):
                for i in range(origin[0],origin[0]+side):
                    if time.perf_counter()-began>seconds:
                        status='TIME_BUDGET_EXHAUSTED';break
                    p=(np.array([i,j,z])+.5)/n
                    judge=oracle if independent is None else independent
                    decision=judge.membership(p) if cm<0 else None
                    m=cm if cm>=0 else 1 if decision.status=='SOLID' else 0 if decision.status=='FREE' else -1
                    if m<0:
                        unresolved.append({'voxel':[i,j,z],'stage':'membership','status':decision.status,'reason':decision.reason});continue
                    if m:
                        y[z,j,i]=False;mask[z,j,i]=True;valid[z,j,i]=True;continue
                    decision=judge.segment(source,p) if cy<0 else None
                    v=cy if cy>=0 else 1 if decision.status=='MISS' else 0 if decision.status=='HIT' else -1
                    if v<0:
                        unresolved.append({'voxel':[i,j,z],'stage':'visibility','status':decision.status,'reason':decision.reason});continue
                    y[z,j,i]=bool(v);valid[z,j,i]=True
        if status!='COMPLETE':break
    if unresolved and status=='COMPLETE':status='NUMERICALLY_INCOMPLETE'
    complete=bool(valid.all()) and status=='COMPLETE'
    assert not np.any(y&mask)
    return {'response':y,'solid':mask,'valid':valid,'complete':complete,'status':status,
            'wall_seconds':time.perf_counter()-began,'cpu_seconds':time.process_time()-cpu,
            'certificate_coverage':filled/n**3,'valid_voxels':int(valid.sum()),
            'unresolved_count':len(unresolved),'unresolved_examples':unresolved[:8],
            'work':{k:v-before.get(k,0) for k,v in oracle.work.items()},'independent':independent is not None}
