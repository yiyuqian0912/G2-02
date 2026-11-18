"""Rerender a source derived solely from each certified inner region using native ray clipping."""
import argparse
from fractions import Fraction
from hashlib import sha256
import io
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from rind3d.geometry import load_geometry
from rind3d.rendering.context import GeometryContext
from rind3d_posterior.fastlabel.inputs import POLICY, legal_source
from research import read_obs
from query import load_records
from poly import Poly, primitive, form


def interior_point(p):
    # A positive convex combination of every vertex is strictly interior.
    # Equal Euclidean weights avoid bias to the vertex with the largest
    # homogeneous denominator (that bias can round a thin interior onto a face).
    barycenter = [sum(Fraction(v[k], v[3]) for v in p.vertices)/len(p.vertices) for k in range(3)]
    from math import lcm
    common=lcm(*(v.denominator for v in barycenter))
    total=tuple(int(v*common) for v in barycenter)+(common,)
    coords = list(map(float,barycenter))
    den = max(v.as_integer_ratio()[1] for v in coords)
    dyadic = tuple(int(Fraction(v)*den) for v in coords) + (den,)
    # Do not let binary64 conversion move the point onto/outside a supporting face.
    for face in p.faces:
        a=face[0]
        def direction(b):
            return tuple(b[i]*a[3]-a[i]*b[3] for i in range(3))
        u=direction(face[1])
        for b in face[2:]:
            v=direction(b)
            n=tuple(u[(i+1)%3]*v[(i+2)%3]-u[(i+2)%3]*v[(i+1)%3] for i in range(3))
            if any(n): break
        else:
            # Redundant collinear face has no supporting plane of its own.
            continue
        plane=primitive((*(x*a[3] for x in n),-sum(n[i]*a[i] for i in range(3))))
        if form(plane,total)*form(plane,dyadic)<=0: return None
    return np.array(coords, dtype='<f8')


def verify(root, record):
    start=perf_counter()
    task=record['task']
    obs,digest=read_obs(root,task)
    if digest!=record['geometry_sha256']: raise ValueError('geometry identity changed')
    pieces=[Poly([tuple(tuple(x) for x in f) for f in p]) for p in record['inner_pieces']]
    point=next((s for p in sorted(pieces,key=lambda p:p.volume,reverse=True)
                if (s:=interior_point(p)) is not None), None)
    if point is None:
        return dict(key=task['key'],geometry_sha256=digest,voxels=0,
                    status='no_binary64_interior_witness_found',wall_s=perf_counter()-start)
    slot=task['slot']
    with np.load(root/'scenes'/f'{slot//1000:04d}'/f'{slot:06d}.npz',allow_pickle=False) as z:
        geom=load_geometry(io.BytesIO(z['geometry'].tobytes()))
    context=GeometryContext(geom,POLICY)
    proof=legal_source(context,point)
    if proof is None: raise ArithmeticError('native source clearance failed')
    n=obs.denom//2
    if all(o<=v*n<=o+obs.side for o,v in zip(obs.origin,point)):
        raise ArithmeticError('witness belongs to observation window')
    model=context.bind(point)
    response_errors=solid_errors=0
    for loc in product(range(0,obs.side,64),repeat=3):
        origin=tuple(a+b for a,b in zip(obs.origin,loc))
        y,m,_=model.render(origin,64,n)
        observed=obs.cls[tuple(slice(a,a+64) for a in loc)].T
        response_errors+=int(np.count_nonzero(y!=(observed==1)))
        solid_errors+=int(np.count_nonzero(m!=(observed==2)))
    if response_errors or solid_errors: raise ArithmeticError((task, response_errors,solid_errors))
    return dict(key=task['key'], geometry_sha256=digest,
                observation_xyz_sha256=sha256(obs.cls.tobytes()).hexdigest(),
                origin=obs.origin, voxels=obs.side**3, clearance=proof['status'],status='verified',
                response_errors=response_errors, solid_errors=solid_errors,
                backend=model.identity,wall_s=perf_counter()-start)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('root','records','out'): ap.add_argument(name,type=Path)
    args=ap.parse_args()
    with args.out.open('x') as dest:
        for row in load_records(args.records):
            if row['positive']:
                result=verify(args.root,row)
                dest.write(json.dumps(result)+'\n')
                dest.flush()
                print(result['key'],result['voxels'],flush=True)
