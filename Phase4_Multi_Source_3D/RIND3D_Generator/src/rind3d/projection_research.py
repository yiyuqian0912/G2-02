"""Isolated independent verifier B2: exact silhouette facets + interval/GMP.

No source/receiver clipping from A is used. Production B remains untouched.
Only bounded, full-dimensional convex unions with the original geometry proof
are supported. This is an experimental backend, not automatic authorization to
publish; its separate correctness certificate is required by the new pilot.
"""
from collections import Counter
from fractions import Fraction as Q
import hashlib,json,time
from pathlib import Path
import numpy as np
from . import _research
from .rendering.polyhedra import normalize,rational_rows,shadow_planes

VERSION='rational-silhouette-projection-v3-analytic-enclosures'

def value(row,p):return sum(a*x for a,x in zip(row[:3],p))+row[3]

def topology(context):
    """Use the existing exact *complete* triple-plane vertex enumeration.

    Check every active equality with Fraction, including positive face area.
    Boundedness and strict interior/world duals are independently checked by
    the existing geometry acceptance. Source-independent topology is reusable.
    """
    from .lite_acceptance import verify_geometry_proof
    proof=context.geometry_proof();verify_geometry_proof(context.bundle,proof)
    result=[]
    for body,record in zip(context.bodies,proof['components']):
        vertices=[tuple(map(Q,v)) for v in record['vertices']];active=[];facets=[]
        for i,row in enumerate(body):
            ids={j for j,v in enumerate(vertices) if value(row,v)==0};active.append(ids)
            points=[vertices[j] for j in sorted(ids)];area=False
            for a in points[1:]:
                for b in points[1:]:
                    u=[a[k]-points[0][k] for k in range(3)];v=[b[k]-points[0][k] for k in range(3)]
                    if any(u[k]*v[(k+1)%3]-u[(k+1)%3]*v[k] for k in range(3)):area=True;break
                if area:break
            if area:facets.append(i)
        edges={(i,j) for i in facets for j in facets if i<j and len(active[i]&active[j])>=2}
        if len(facets)<4 or len(edges)<6:raise ValueError('Certified full-dimensional convex topology unavailable')
        result.append({'facets':tuple(facets),'edges':frozenset(edges),'vertices':len(vertices)})
    return tuple(result)

def silhouette_planes(body,source,topo):
    p=tuple(Q(float(v)) for v in source);c=[-value(row,p) for row in body]
    negative=[i for i in topo['facets'] if c[i]<0];positive=[i for i in topo['facets'] if c[i]>0]
    if not negative:raise ValueError('Source in closed convex component')
    rows=[body[i] for i in negative]
    rows.extend(body[i] for i in topo['facets'] if c[i]==0)
    for i in negative:
        for j in positive:
            if (min(i,j),max(i,j)) not in topo['edges']:continue
            n=tuple(c[j]*body[i][k]-c[i]*body[j][k] for k in range(3))
            rows.append((*n,-sum(a*x for a,x in zip(n,p))))
    return tuple(dict.fromkeys(normalize(row) for row in rows if any(row)))

def implementation_identity():
    paths=[Path(__file__),Path(_research.__file__),Path(__file__).parents[2]/'native/research_projection.hpp']
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}
    return {'kind':'exact-projected-shadow','scope':'bounded-convex-top-union-v1','revision':VERSION,
            'implementation_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()}

class ProjectionSource:
    def __init__(self,context,source,*,topo=None,reduced=True,leaf_side=4):
        start=time.perf_counter();self.context=context;self.source=np.asarray(source,dtype='<f8').copy();self.leaf_side=leaf_side
        if self.source.shape!=(3,) or not np.isfinite(self.source).all() or np.any(self.source<=0) or np.any(self.source>=1):raise ValueError('Interior world source required')
        self.topology=topo if topo is not None else topology(context)
        self.planes=tuple(silhouette_planes(b,self.source,t) if reduced else shadow_planes(b,self.source) for b,t in zip(context.bodies,self.topology))
        self.native=_research.Projection(rational_rows(context.bodies),rational_rows(self.planes));self.work=Counter()
        self.identity=implementation_identity();self.setup_seconds=time.perf_counter()-start
        self.source_id=hashlib.sha256(context.geometry.content_hash+self.source.tobytes()).hexdigest()
        self.plane_report={'original_fme_planes':[len(shadow_planes(b,self.source)) for b in context.bodies],
                           'retained_planes':[len(p) for p in self.planes],'actual_faces':[len(t['facets']) for t in self.topology],
                           'proof':'bounded full-dimensional convex shadow facets: front caps, source-coplanar facets, silhouette edges'}
    def response(self,source,points,*,budget=None):
        if not np.array_equal(source,self.source):raise ValueError('Bound source differs')
        p=np.ascontiguousarray(points,dtype='<f8').reshape(-1,3)
        if budget:budget.check(len(p))
        y,_,work=self.native.points(p);self.work.update(work);return y.astype(bool)
    def packed_batch(self,origins,side,resolution,*,budget=None):
        if budget:budget.check()
        pairs,work=self.native.batch([list(map(int,o)) for o in origins],side,resolution,self.leaf_side)
        depth=work.pop('max_subdivision_depth');self.work.update(work);self.work['max_subdivision_depth']=max(self.work['max_subdivision_depth'],depth)
        if budget:budget.check(work['point_rays']+work['solid_points'])
        return pairs
    def render(self,origin,size,resolution,*,budget=None):
        pair=self.packed_batch([origin],size,resolution,budget=budget)[0]
        a=[np.unpackbits(np.frombuffer(p,dtype=np.uint8),bitorder='little').reshape((64,)*3)[:size,:size,:size].astype(bool) for p in pair]
        return *a,0
    def certificate(self,source,lo,hi,*,budget=None):
        if not np.array_equal(source,self.source):raise ValueError('Bound source differs')
        if budget:budget.check()
        return self.native.certificate(list(lo),list(hi))
