"""Exact Fourier–Motzkin shadow construction and isolated stream adapters."""
from collections import Counter
from contextlib import contextmanager
from fractions import Fraction as F
import hashlib,time,threading
import numpy as np
from rind.contracts import CSGNode,CSGOpcode,CodecID,Leaf,RootKind
from rind3d.geometry import Unsupported,centres
from . import _shadow_native as native


def inverse(matrix):
    a=[[F(float(v)) for v in row]+[F(int(i==j)) for j in range(3)] for i,row in enumerate(matrix)]
    for j in range(3):
        pivot=next((i for i in range(j,3) if a[i][j]),None)
        if pivot is None:raise Unsupported('Singular canonical affine matrix')
        a[j],a[pivot]=a[pivot],a[j];v=a[j][j];a[j]=[x/v for x in a[j]]
        for i in range(3):
            if i!=j:
                v=a[i][j];a[i]=[x-v*y for x,y in zip(a[i],a[j])]
    return [row[3:] for row in a]


def normalize(row):
    scale=max(map(abs,row))
    return tuple(v/scale for v in row) if scale else tuple(row)


def extract(bundle):
    refs={x.content_hash:x for x in bundle.references};bodies=[];old=[]
    def walk(key):
        item=refs[key]
        if isinstance(item,CSGNode):
            if item.opcode!=CSGOpcode.UNION:raise Unsupported('Experimental shadows require top-level unions')
            for child in item.children:walk(child)
            return
        if not isinstance(item,Leaf) or item.codec not in (CodecID.AFFINE_BOX,CodecID.CONVEX_HALFSPACES):raise Unsupported('Only canonical affine boxes and convex halfspaces supported')
        c=[F(float(v)) for v in item.field('center').as_array()];matrix=item.field('matrix').as_array();inv=inverse(matrix)
        if item.codec==CodecID.AFFINE_BOX:
            normals=[];offsets=[]
            for axis in range(3):
                for sign in (-1,1):
                    normals.append([F(sign*int(k==axis)) for k in range(3)]);offsets.append(F(1))
            if np.array_equal(matrix,np.diag(np.diag(matrix))) and np.all(np.diag(matrix)>0):old.append([0,*map(float,c),*np.diag(matrix)])
        else:
            normals=[[F(float(v)) for v in row] for row in item.field('normal').as_array()];offsets=[F(float(v)) for v in item.field('plane_offset').as_array()]
        # The canonical backend regularizes intersection with Ω. A strict
        # interior witness in int(Ω) proves that regularization cannot remove
        # a face-only component. Other domain configurations use the oracle.
        if any(v<=0 or v>=1 for v in c) or any(b<=0 for b in offsets):raise Unsupported('A strict canonical center witness inside the world is required')
        planes=[]
        for a,b in zip(normals,offsets):
            n=[sum(a[i]*inv[i][j] for i in range(3)) for j in range(3)];d=-b-sum(x*y for x,y in zip(n,c))
            if not any(n):
                if d>0:raise Unsupported('Empty halfspace constraint')
                continue
            planes.append(normalize((*n,d)))
        bodies.append(tuple(dict.fromkeys(planes)))
    root=bundle.geometries[0]
    if root.root_kind!=RootKind.REF:raise Unsupported('No bounded convex body')
    walk(root.root_hash)
    return bodies,old if len(old)==len(bodies) else []


def shadow_planes(body,source):
    p=tuple(F(float(v)) for v in source);c=[-row[3]-sum(a*x for a,x in zip(row[:3],p)) for row in body]
    negative=[i for i,v in enumerate(c) if v<0];positive=[i for i,v in enumerate(c) if v>0]
    if not negative:raise ValueError('Source must be outside every closed obstacle')
    rows=[body[i] for i in negative]
    for i,v in enumerate(c):
        if not v:rows.append((*body[i][:3],-sum(a*x for a,x in zip(body[i][:3],p))))
    for i in negative:
        for j in positive:
            n=tuple(c[j]*body[i][k]-c[i]*body[j][k] for k in range(3));d=-sum(a*x for a,x in zip(n,p))
            row=(*n,d)
            if any(row):rows.append(row)
    return tuple(dict.fromkeys(normalize(row) for row in rows))


def strings(bodies):return [[[str(v) for v in row] for row in body] for body in bodies]


def geometry_split(geometry_id,seed=0):
    k=int.from_bytes(hashlib.sha256(f'{seed}/{geometry_id}/geometry-group-v1'.encode()).digest()[:8],'little')%100
    return 'train' if k<80 else 'val' if k<90 else 'test'


class SharedGeometry:
    def __init__(self,base):
        t=time.perf_counter();self.base=base;self.bodies,self.old=extract(base.bundle)
        self.native=native.Geometry(strings(self.bodies),self.old);self.setup_seconds=time.perf_counter()-t
        self.geometry_id=base.geometry.content_hash.hex()
    def bind(self,source,*,solid=None):return ShadowOracle(self,source,solid=solid)


class SharedSolid:
    """A geometry/policy-checked, read-only, bounded RINDPack mask cache."""
    def __init__(self,shared,source,path):
        from rind3d.pack import RINDPack
        from rind3d.stream import scene_identity
        self.geometry_id=shared.geometry_id;self.policy_hash=shared.base.policy_hash;self.lock=threading.RLock()
        self.pack=RINDPack(path,verify_file=True)
        if self.pack.identity!=scene_identity(shared.base,source) or self.pack.policy!=self.policy_hash:
            self.pack.close();raise ValueError('Shared mask geometry/source/policy identity mismatch')
        self.shape=self.pack.shape
    def read_window(self,*args):
        with self.lock:return self.pack.read_window(*args)
    def close(self):self.pack.close()


class ShadowOracle:
    def __init__(self,shared,source,*,solid=None):
        self.shared=shared;self.base=shared.base;self.source=np.array(source,dtype='<f8',copy=True).reshape(3);self.source.setflags(write=False);self.work=Counter();self.solid=solid
        if solid is not None and (solid.geometry_id!=shared.geometry_id or solid.policy_hash!=self.base.policy_hash):raise ValueError('Cross-geometry mask reuse forbidden')
        self.base.validate_source(self.source);t=time.perf_counter();self.planes=[shadow_planes(b,self.source) for b in shared.bodies]
        self.native=native.Model(shared.native,strings(self.planes),self.source.tolist());self.setup_seconds=time.perf_counter()-t
        self.source_id=hashlib.sha256(bytes.fromhex(shared.geometry_id)+self.source.tobytes()).hexdigest()
    def __getattr__(self,name):return getattr(self.base,name)
    @property
    def metrics(self):return {**self.base.metrics,**{f'shadow_{k}':v for k,v in self.work.items()}}
    def report(self):return {'version':'exact-polyhedral-shadow-experiment-v1','geometry_id':self.shared.geometry_id,'source_id':self.source_id,
        'geometry_setup_seconds':self.shared.setup_seconds,'shadow_setup_seconds':self.setup_seconds,'obstacles':len(self.planes),
        'geometry_plane_counts':[len(b) for b in self.shared.bodies],'shadow_plane_counts':[len(b) for b in self.planes],
        'work':dict(self.work),'production_ready':False,'solid_mask_reused':self.solid is not None}
    def _same(self,source):return np.array_equal(np.asarray(source).reshape(-1),self.source)
    def certificate(self,source,lo,hi,*,budget=None):
        if not self._same(source):return self.base.certificate(source,lo,hi,budget=budget)
        if budget:budget.check()
        self.work['certificate_calls']+=1
        return tuple(self.native.certificate(list(lo),list(hi)))
    def _query(self,points,which,budget):
        p=np.ascontiguousarray(points,dtype=np.float64).reshape(-1,3)
        if budget:budget.check(len(p))
        values=self.native.points(p,False)[which];unknown=np.flatnonzero(values<0)
        if len(unknown):
            self.work['canonical_fallback_rays' if which==0 else 'canonical_fallback_points']+=len(unknown)
            values[unknown]=(self.base.response(self.source,p[unknown],budget=budget) if which==0 else self.base.contains(p[unknown],budget=budget)).astype(np.int8)
        return values.astype(bool)
    def contains(self,points,*,budget=None):return self._query(points,1,budget)
    def response(self,source,points,*,budget=None):
        return self._query(points,0,budget) if self._same(source) else self.base.response(source,points,budget=budget)
    def render(self,source,origin,size,n,*,budget=None,leaf_side=8,old_mode=False):
        if not self._same(source):raise ValueError('Source-dependent shadow cache key mismatch')
        if budget:budget.check()
        args=(list(origin),size,n,leaf_side,old_mode)
        y,m,stats=self.native.raster(*args,True) if self.solid is not None else self.native.raster(*args)
        self.work.update(stats)
        if budget:budget.check(stats['point_rays']+stats['solid_points'])
        uy=np.flatnonzero(y.ravel()<0);um=np.flatnonzero(m.ravel()<0)
        if len(uy) or len(um):
            p=centres(origin,size,n)
            if len(uy):
                self.work['canonical_fallback_rays']+=len(uy);y.ravel()[uy]=self.base.response(self.source,p[uy],budget=budget)
            if len(um):
                self.work['canonical_fallback_points']+=len(um);m.ravel()[um]=self.base.contains(p[um],budget=budget)
        if self.solid is not None:
            if self.solid.shape!=(n,)*3:raise ValueError('Shared mask resolution mismatch')
            m=self.solid.read_window(*origin,size).astype(np.int8);self.work['solid_bricks_reused']+=1
        y=y.astype(bool);m=m.astype(bool)
        if np.any(y&m):raise AssertionError('Solid receiver incorrectly lit')
        return y,m,stats['certificate_covered_voxels']


@contextmanager
def native_stream(*,old_mode=False):
    """Subprocess-scoped experiment hook; production stream source is untouched."""
    import rind3d.stream as stream
    original=stream.render_region
    def render(oracle,*args,**kwargs):return oracle.render(*args,**kwargs,old_mode=old_mode)
    stream.render_region=render
    try:yield
    finally:stream.render_region=original


def optional_adapter(base,source):
    try:return SharedGeometry(base).bind(source)
    except Unsupported:return base
