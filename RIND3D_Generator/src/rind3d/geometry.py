"""Adapters to the existing canonical codecs and certified geometry oracles."""
from __future__ import annotations
from dataclasses import dataclass,field
import hashlib
import json
from pathlib import Path
import time
import threading

import numpy as np
from rind.contracts import CSGNode,CSGOpcode,CodecID,Geometry,Leaf,RootKind,make_leaf
from rind.storage import CanonicalBundle,pack_arrays,_decode_bundle
from rind.geometry import GeometryView,Membership,QueryStatus
from rind.expanded_geometry import ExpandedGeometryView
from .pack import atomic_json

POLICY={'version':'closed-solid-closed-segment-v1','coordinates':'exact binary64 dyadics',
        'array_order':'zyx','source_precision':'float64','solid_response':0,
        'native_filter':'outward-rounded binary64 intervals; unknown -> original oracle',
        'transcendental':'existing Arb precision/subdivision policy; unresolved is fatal'}
POLICY_HASH=hashlib.sha256(json.dumps(POLICY,sort_keys=True).encode()).hexdigest()

def physics_provenance(policy=None):
    """One implementation identity for all mathematically equivalent renderers."""
    from rind.geometry import backend_build_id
    from rind import transcendental,expanded_geometry
    return {'physics':POLICY,'oracle_limits':dict(policy or {}),'canonical_backend':backend_build_id(3),
            'function_source_sha256':{Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (transcendental,expanded_geometry)},
            'filter_policy':'outward-interval-convex-union-v1'}

class NumericalUnresolved(RuntimeError):pass
class Aborted(RuntimeError):pass
class BudgetExhausted(Aborted):
    """Our declared work limit, distinct from an aborted canonical predicate."""
    def __init__(self,kind):
        self.kind=kind
        super().__init__('Declared '+kind+' budget exhausted')
class Unsupported(RuntimeError):pass
class Incomplete(RuntimeError):pass

@dataclass
class Budget:
    seconds:float=30
    max_points:int=50_000_000
    started:float=field(default_factory=time.perf_counter)
    points:int=0
    lock:object=field(default_factory=threading.Lock,repr=False)

    def check(self,n=0):
        if self.seconds<=0 or self.max_points<0:raise ValueError('Invalid work budget')
        with self.lock:
            if time.perf_counter()-self.started>self.seconds:raise BudgetExhausted('time')
            if self.points+n>self.max_points:raise BudgetExhausted('point')
            self.points+=n

def keyed_rng(seed,scene,stream):return np.random.default_rng(np.random.SeedSequence([int(seed),int(scene),int(stream)]))

def generate_geometry(seed,scene_id,families):
    """Offline family v1. Function coefficients, not meshes, are canonical."""
    rng=keyed_rng(seed,scene_id,1);bundle=CanonicalBundle(3);roots=[]
    def leaf(codec,**fields):
        item=make_leaf(codec,3,**fields);bundle.add_leaf(item);return item.content_hash
    def box(center,half):return leaf('affine_box_v1',center=center,matrix=np.diag(half))
    for family in families:
        if family=='slabs':
            for j in range(3):
                half=rng.uniform(.12,.23,3);half[j%3]=rng.uniform(.018,.04)
                roots.append(box(rng.uniform(.28,.72,3),half))
        elif family=='csg':
            for _ in range(3):roots.append(box(rng.uniform(.25,.75,3),rng.uniform(.06,.15,3)))
        elif family=='smooth_blobs':
            for _ in range(3):roots.append(leaf('affine_ball_v1',center=rng.uniform(.25,.75,3),matrix=np.diag(rng.uniform(.06,.16,3))))
        elif family=='random_field':
            roots.append(leaf('cosine_field_v1',amplitude=[1.,.35,.2],frequency=[[9.,1.,0.],[0.,7.,1.],[1.,0.,6.]],phase=rng.uniform(-3,3,3),threshold=.93))
        else:raise ValueError(f'Unknown generator family {family}')
    if len(roots)>1:
        node=CSGNode(3,CSGOpcode.UNION,tuple(roots));bundle.add_node(node);root=node.content_hash
    elif roots:root=roots[0]
    else:root=None
    geometry=Geometry(3,RootKind.REF,root) if root is not None else Geometry.empty(3)
    bundle.add_geometry(geometry)
    # A canonical lossless codec round trip is required before geometry is used.
    return _decode_bundle(pack_arrays(bundle),3)

def save_geometry(path,bundle):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**pack_arrays(bundle))
    tmp.replace(path)

def load_geometry(path):
    with np.load(path,allow_pickle=False) as z:return _decode_bundle({k:z[k] for k in z.files},3)

def filter_primitives(bundle):
    refs={x.content_hash:x for x in bundle.references};pieces=[]
    def walk(h):
        x=refs[h]
        if isinstance(x,CSGNode):
            if x.opcode!=CSGOpcode.UNION:raise Unsupported('Filter only accelerates top-level convex unions')
            for child in x.children:walk(child)
            return
        if not isinstance(x,Leaf) or x.codec not in (CodecID.AFFINE_BOX,CodecID.AFFINE_BALL):raise Unsupported('Function/mesh requires the canonical fallback')
        center=x.field('center').as_array();matrix=x.field('matrix').as_array()
        if not np.array_equal(matrix,np.diag(np.diag(matrix))) or np.any(np.diag(matrix)<=0):raise Unsupported('Non-diagonal transform requires canonical fallback')
        pieces.append((0 if x.codec==CodecID.AFFINE_BOX else 1,center.tolist(),np.diag(matrix).tolist()))
    geometry=bundle.geometries[0]
    if geometry.root_kind==RootKind.REF:walk(geometry.root_hash)
    return pieces

class Oracle:
    def __init__(self,bundle,policy=None):
        self.bundle=bundle;self.geometry=bundle.geometries[0]
        self.policy=dict(policy or {});self.metrics={'interval_points':0,'fallback_points':0,'certificate_calls':0,'unresolved':0}
        self.lock=threading.RLock();self.evaluated_sources=set()
        has_function=any(isinstance(x,Leaf) and x.codec in (CodecID.COSINE_FIELD,CodecID.RADIAL_FOURIER_SPHERE) for x in bundle.references)
        self.reference=(ExpandedGeometryView if has_function else GeometryView)(self.geometry.content_hash,bundle,self.policy)
        self.capability=self.reference.capability()
        if not self.capability.get('supported'):raise Unsupported(str(self.capability))
        try:
            from ._native import IntervalFilter
            self.primitives=filter_primitives(bundle);self.filter=IntervalFilter(self.primitives)
        except Unsupported:self.primitives=None;self.filter=None
        self.provenance=physics_provenance(self.policy)
        self.policy_hash=hashlib.sha256(json.dumps(self.provenance,sort_keys=True).encode()).hexdigest()

    def _values(self,result):
        if not np.all(result.valid):
            self.metrics['unresolved']+=int((~result.valid).sum())
            if np.any(result.status==int(QueryStatus.ABORTED)):raise Aborted('Canonical oracle aborted')
            raise NumericalUnresolved('Canonical oracle did not certify a query: '+str(result.reason_code))
        return result.value

    def contains(self,points,*,budget=None):
        p=np.ascontiguousarray(points,dtype=np.float64).reshape(-1,3)
        if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):raise ValueError('Invalid world coordinates')
        if budget:budget.check(len(p))
        out=self.filter.contains(p) if self.filter else np.full(len(p),-1,dtype=np.int8)
        unknown=np.flatnonzero(out<0);self.metrics['interval_points']+=len(p)-len(unknown)
        for i in unknown:
            if budget:budget.check()
            with self.lock:v=self._values(self.reference.classify_points(p[i]))[0]
            out[i]=int(v!=int(Membership.FREE));self.metrics['fallback_points']+=1
        return out.astype(bool)

    def response(self,source,points,*,budget=None):
        p=np.ascontiguousarray(points,dtype=np.float64).reshape(-1,3)
        s=np.ascontiguousarray(source,dtype=np.float64).reshape(-1,3)
        if not np.isfinite(p).all() or not np.isfinite(s).all() or np.any(p<0) or np.any(p>1) or np.any(s<0) or np.any(s>1):raise ValueError('Invalid world coordinates')
        if s.shape[0] not in (1,len(p)) and len(p)!=1:raise ValueError('Paired or single-source queries only')
        n=max(len(p),len(s))
        if budget:budget.check(n)
        out=self.filter.visibility(s,p) if self.filter else np.full(n,-1,dtype=np.int8)
        if self.filter:self.evaluated_sources.update(a.astype('<f8').tobytes() for a in s)
        unknown=np.flatnonzero(out<0);self.metrics['interval_points']+=n-len(unknown)
        for i in unknown:
            if budget:budget.check()
            a=s[0 if len(s)==1 else i];b=p[0 if len(p)==1 else i]
            self.evaluated_sources.add(a.astype('<f8').tobytes())
            with self.lock:out[i]=1-int(self._values(self.reference.segment_intersect(a,b))[0]);self.metrics['fallback_points']+=1
        return out.astype(bool)

    def certificate(self,source,lo,hi,*,budget=None):
        if budget:budget.check()
        self.evaluated_sources.add(np.asarray(source,dtype='<f8').tobytes())
        self.metrics['certificate_calls']+=1
        if self.filter:return tuple(self.filter.certificate(list(source),list(lo),list(hi)))
        # This is an optional acceleration proof. A mixed/unknown box does not
        # decide a scientific label; leaf points still use the reference oracle.
        with self.lock:result=self.reference.classify_box([lo,hi])
        solid=int(result.value[0]) if bool(result.valid[0]) else -1
        if solid==1:return 0,1
        hull=[np.minimum(source,lo),np.maximum(source,hi)]
        with self.lock:free=self.reference.classify_box(hull)
        if bool(free.valid[0]) and int(free.value[0])==0:return 1,0
        return -1,0 if solid==0 else -1

    def validate_source(self,source,budget=None):
        p=np.asarray(source,dtype=np.float64)
        if p.shape!=(3,) or np.any(p<=0) or np.any(p>=1) or self.contains(p,budget=budget)[0]:raise ValueError('Source must be certified free and strictly inside the world')

def centres(origin_xyz,size,resolution):
    x,y,z=origin_xyz
    sx,sy,sz=(size,)*3 if np.isscalar(size) else size
    zz,yy,xx=np.meshgrid((np.arange(z,z+sz)+.5)/resolution,(np.arange(y,y+sy)+.5)/resolution,(np.arange(x,x+sx)+.5)/resolution,indexing='ij')
    return np.stack([xx,yy,zz],axis=-1).reshape(-1,3)

def render_region(oracle,source,origin,size,resolution,*,budget=None,leaf_side=8):
    """Bounded adaptive sampling, with proofs for fills and oracle leaf values."""
    response=np.empty((size,)*3,dtype=bool);solid=np.empty_like(response)
    pending=[(tuple(map(int,origin)),(int(size),)*3)];certified_voxels=0
    while pending:
        pos,lengths=pending.pop()
        lo=(np.asarray(pos)+.5)/resolution;hi=(np.asarray(pos)+lengths-.5)/resolution
        yv,mv=oracle.certificate(source,lo,hi,budget=budget)
        x,y,z=np.asarray(pos)-origin;lx,ly,lz=lengths;sl=(slice(z,z+lz),slice(y,y+ly),slice(x,x+lx))
        if yv>=0 and mv>=0:
            response[sl]=bool(yv);solid[sl]=bool(mv);certified_voxels+=lx*ly*lz;continue
        if max(lengths)<=leaf_side:
            points=centres(pos,lengths,resolution)
            response[sl]=oracle.response(source,points,budget=budget).reshape((lz,ly,lx)) if yv<0 else bool(yv)
            solid[sl]=oracle.contains(points,budget=budget).reshape((lz,ly,lx)) if mv<0 else bool(mv)
        else:
            cuts=[[(0,n)] if n<=leaf_side else [(0,n//2),(n//2,n-n//2)] for n in lengths]
            for dz,nz in cuts[2]:
                for dy,ny in cuts[1]:
                    for dx,nx in cuts[0]:pending.append(((pos[0]+dx,pos[1]+dy,pos[2]+dz),(nx,ny,nz)))
    if np.any(response&solid):raise AssertionError('Solid receivers must have response zero')
    return response,solid,certified_voxels
