"""Immutable canonical geometry, exact native predicates and bounded source cache."""
from collections import Counter,OrderedDict
from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import threading
import time
import numpy as np
from rind3d.geometry import Oracle,physics_provenance,Unsupported
from rind3d import _convex
from .polyhedra import decode_convex,rational_rows,shadow_planes

SCOPE='bounded-convex-top-union-v1'

@lru_cache(maxsize=8)
def backend_identity(kind):
    paths=[Path(_convex.__file__),Path(__file__),Path(__file__).with_name('polyhedra.py')]
    values={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return {'kind':kind,'scope':SCOPE,'implementation_sha256':hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()}

@dataclass(frozen=True)
class PredicateCapabilities:
    scope:str
    exact_membership:bool=True
    exact_closed_segments:bool=True
    independent_projection:bool=True
    publication_acceptance_required:bool=True


class GeometryContext:
    def __init__(self,bundle,policy=None,*,source_cache_size=1024):
        start=time.perf_counter();self.bundle=bundle;self.geometry=bundle.geometries[0]
        self.policy=dict(policy or {});self.provenance=physics_provenance(self.policy)
        self.policy_hash=hashlib.sha256(json.dumps(self.provenance,sort_keys=True).encode()).hexdigest()
        self.components=decode_convex(bundle);self.bodies=tuple(c.rows for c in self.components)
        self.native=_convex.Geometry(rational_rows(self.bodies));self.geometry_id=self.geometry.content_hash.hex()
        self.capabilities=PredicateCapabilities(SCOPE);self._canonical=None;self.canonical_init_seconds=0.
        self._proof=None;self._cache=OrderedDict();self.source_cache_size=int(source_cache_size)
        if self.source_cache_size<1:raise ValueError('Positive source cache capacity required')
        self.lock=threading.RLock();self.work=Counter();self.evaluated_sources=set()
        self.setup_seconds=time.perf_counter()-start

    @property
    def canonical(self):
        with self.lock:
            if self._canonical is None:
                start=time.perf_counter();self._canonical=Oracle(self.bundle,self.policy)
                if self._canonical.policy_hash!=self.policy_hash:raise AssertionError('Canonical numerical-policy mismatch')
                self.canonical_init_seconds+=time.perf_counter()-start
            return self._canonical

    def contains(self,points,*,budget=None):
        p=np.ascontiguousarray(points,dtype='<f8').reshape(-1,3)
        if budget:budget.check(len(p))
        self.work['membership_points']+=len(p)
        return self.native.contains(p).astype(bool)

    def membership(self,points):
        p=np.ascontiguousarray(points,dtype='<f8').reshape(-1,3);values=self.native.membership(p)
        uncertain=np.flatnonzero(values<0)
        if len(uncertain):
            self.work['canonical_topology_queries']+=len(uncertain)
            values[uncertain]=self.canonical._values(self.canonical.reference.classify_points(p[uncertain]))
        return values

    def box_class(self,lo,hi):
        self.geometry_proof()
        value=self.native.box_class(list(lo),list(hi))
        if value<0:
            self.work['canonical_box_queries']+=1
            value=int(self.canonical._values(self.canonical.reference.classify_box([lo,hi]))[0])
        return value

    def geometry_proof(self):
        if self._proof is None:
            start=time.perf_counter();proof=dict(self.native.proof([[float(v) for v in c.anchor] for c in self.components]))
            proof.update(protocol='convex-interior-and-world-dual-v1',scope=SCOPE,geometry_id=self.geometry_id,
                         canonical_roundtrip=True,proof_seconds=time.perf_counter()-start)
            for component,record in zip(self.components,proof['components']):
                record.update(leaf_id=component.leaf_id,codec=component.codec,anchor=[str(v) for v in component.anchor])
            self._proof=proof
        return self._proof

    def clearance(self,point,radius):
        """GT proof by a strict separating plane for each convex component.

        This is sufficient, not a LE test. An inconclusive separator delegates
        to the original exact clearance service and verifies any LE witness.
        """
        p=np.asarray(point,dtype='<f8').reshape(3)
        if np.any(p<=0) or np.any(p>=1) or not np.isfinite(p).all():raise ValueError('Source outside open world')
        selected=self.native.clearance_planes(p.tolist(),float(radius))
        if len(selected)==len(self.bodies):
            return {'status':'GT','proof':'strict-separating-planes','planes':selected,'radius_hex':float(radius).hex()}
        from rind.geometry import Clearance
        self.work['canonical_clearance_queries']+=1
        oracle=self.canonical;result=oracle.reference.clearance_decide(p,float(radius));value=int(oracle._values(result)[0])
        if value==int(Clearance.GT):return {'status':'GT','proof':'canonical-clearance','radius_hex':float(radius).hex()}
        witness=result.evidence_ref[0]
        if value!=int(Clearance.LE) or witness is None or not oracle.reference.verify_clearance_witness(p,float(radius),witness):
            raise RuntimeError('Canonical clearance rejection lacked a certified witness')
        return {'status':'LE','proof':'canonical-clearance-witness','radius_hex':float(radius).hex()}

    def bind(self,source,*,independent=False):
        p=np.asarray(source,dtype='<f8').reshape(3);key=(p.tobytes(),bool(independent))
        with self.lock:
            if key in self._cache:
                item=self._cache.pop(key);self._cache[key]=item;return item
            item=NativeSource(self,p,independent=independent);self._cache[key]=item
            while len(self._cache)>self.source_cache_size:self._cache.popitem(last=False)
            return item


class NativeSource:
    def __init__(self,context,source,*,independent=False):
        start=time.perf_counter();self.context=context;self.source=np.array(source,dtype='<f8',copy=True).reshape(3);self.source.setflags(write=False)
        self.independent=bool(independent);self.backend='cpu_native_certified';self.work=Counter()
        planes=[shadow_planes(b,self.source) for b in context.bodies] if independent else []
        self.native=_convex.Model(context.native,self.source.tolist(),rational_rows(planes))
        self.source_id=hashlib.sha256(context.geometry.content_hash+self.source.tobytes()).hexdigest()
        self.identity=backend_identity('exact-projected-shadow' if independent else 'cached-parametric-clipping')
        self.setup_seconds=time.perf_counter()-start

    def __getattr__(self,name):return getattr(self.context,name)
    @property
    def metrics(self):return dict(self.work)
    def validate_source(self,source,budget=None):
        if budget:budget.check()
        if not np.array_equal(np.asarray(source).reshape(3),self.source):self.context.bind(source)
    def contains(self,points,*,budget=None):return self.context.contains(points,budget=budget)
    def response(self,source,points,*,budget=None):
        s=np.ascontiguousarray(source,dtype='<f8').reshape(-1,3);p=np.ascontiguousarray(points,dtype='<f8').reshape(-1,3)
        if len(s)==1 and np.array_equal(s[0],self.source):
            if budget:budget.check(len(p))
            self.context.evaluated_sources.add(self.source.tobytes());y,_,stats=self.native.points(p,self.independent);self.work.update(stats)
            return y.astype(bool)
        if len(s)==1:return self.context.bind(s[0],independent=self.independent).response(s[0],p,budget=budget)
        if len(s) not in (1,len(p)) and len(p)!=1:raise ValueError('Paired or single-source queries required')
        n=max(len(s),len(p));result=np.empty(n,dtype=bool)
        for i in range(n):
            a=self.context.bind(s[0 if len(s)==1 else i],independent=self.independent)
            result[i]=a.response(a.source,p[0 if len(p)==1 else i][None],budget=budget)[0]
        return result
    def certificate(self,source,lo,hi,*,budget=None):
        if budget:budget.check()
        active=self if np.array_equal(np.asarray(source).reshape(-1),self.source) else self.context.bind(source,independent=self.independent)
        return active.native.certificate(list(lo),list(hi),active.independent)
    def render(self,origin,size,resolution,*,budget=None,leaf_side=8):
        if budget:budget.check()
        y,m,stats=self.native.raster(list(origin),int(size),int(resolution),int(leaf_side),self.independent,False)
        self.work.update(stats)
        if budget:budget.check(stats['point_rays']+stats['solid_points'])
        if np.any(y&m):raise AssertionError('Solid receiver lit')
        return y.astype(bool),m.astype(bool),stats['certificate_covered_voxels']
    def packed_batch(self,origins,side,resolution,*,budget=None):
        result=[]
        for origin in origins:
            y,m,_=self.render(origin,side,resolution,budget=budget);pair=[]
            for a in (y,m):
                padded=np.zeros((64,)*3,dtype=bool);padded[:side,:side,:side]=a;pair.append(np.packbits(padded,bitorder='little').tobytes())
            result.append(tuple(pair))
        return result
    def report(self):
        return {'backend':self.backend,'identity':self.identity,'scope':SCOPE,'source_id':self.source_id,
                'geometry_id':self.context.geometry_id,'source_setup_seconds':self.setup_seconds,
                'canonical_initialization_seconds':self.context.canonical_init_seconds,'work':dict(self.work),
                'selection_reason':getattr(self,'selection_reason',None)}
    def close(self):pass
