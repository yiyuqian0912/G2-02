"""Lossless finite continuous models; ordinary NumPy evaluation is DISPLAY ONLY.

Every stored float is interpreted as its exact binary64 value. A kernel uses
q(x)=||M^{-1}(x-c)||² with the inverse taken over the rationals, not a rounded
inverse. Thus A=M^{-T}M^{-1} is exactly SPD whenever det(M)!=0.
"""
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path

from flint import fmpq
import numpy as np

SCHEMA = 'rind3d-experimental-gaussian-parameters-v1'


def rational(v):
    if isinstance(v, fmpq):
        return v
    if isinstance(v, (int, np.integer)):
        return fmpq(int(v))
    if not math.isfinite(float(v)):
        raise ValueError('Nonfinite canonical parameter')
    return fmpq(*float(v).as_integer_ratio())


def inverse(matrix):
    a = [[rational(x) for x in row] + [fmpq(i == j) for j in range(3)]
         for i, row in enumerate(matrix)]
    for k in range(3):
        pivot = next((i for i in range(k, 3) if a[i][k]), None)
        if pivot is None:
            raise ValueError('Singular canonical kernel transform')
        a[k], a[pivot] = a[pivot], a[k]
        d = a[k][k]
        a[k] = [v/d for v in a[k]]
        for i in range(3):
            if i != k:
                d = a[i][k]
                a[i] = [v-d*w for v, w in zip(a[i], a[k])]
    return tuple(tuple(row[3:]) for row in a)


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), fmpq(0))


@dataclass
class Kernel:
    center: object
    matrix: object
    weight: float = 1.
    cluster: int = 0
    scale_class: str = 'base'
    inv: object = field(init=False, repr=False)

    def __post_init__(self):
        self.center = np.array(self.center, dtype='<f8', copy=True)
        self.matrix = np.array(self.matrix, dtype='<f8', copy=True)
        if self.center.shape != (3,) or self.matrix.shape != (3, 3):
            raise ValueError('Expected XYZ center and 3×3 transform')
        if not np.isfinite(self.center).all() or not np.isfinite(self.matrix).all() or not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError('Finite parameters and positive amplitudes required')
        self.inv = inverse(self.matrix)
        self.qcenter = tuple(map(rational, self.center))
        self.row_norms = tuple(dot(tuple(map(rational, row)), tuple(map(rational, row))) for row in self.matrix)
        self.approx_inverse = np.linalg.inv(self.matrix)  # Never used by a certified predicate.
        self.center.setflags(write=False); self.matrix.setflags(write=False)

    def coordinates(self, p):
        d = tuple(rational(x)-c for x, c in zip(p, self.qcenter))
        return tuple(dot(row, d) for row in self.inv)

    def quadratic(self, p):
        d = self.coordinates(p)
        return dot(d, d)

    def coefficients(self, source, receiver):
        u = self.coordinates(source)
        d = tuple(rational(b)-rational(a) for a, b in zip(source, receiver))
        v = tuple(dot(row, d) for row in self.inv)
        return dot(v, v), 2*dot(u, v), dot(u, u)

    def parameters(self):
        return dict(center=self.center, matrix=self.matrix, weight=self.weight,
                    cluster=self.cluster, scale_class=self.scale_class)


@dataclass(frozen=True)
class Threshold:
    kind: str = 'binary64'
    value: float = .5

    def __post_init__(self):
        if self.kind not in ('binary64', 'exp') or not math.isfinite(self.value):
            raise ValueError('Unsupported threshold representation')
        if self.kind == 'binary64' and self.value <= 0:
            raise ValueError('Positive threshold required for bounded positive fields')

    def approximate(self):
        return self.value if self.kind == 'binary64' else math.exp(self.value)


@dataclass
class KernelScene:
    kind: str
    kernels: list[Kernel]
    threshold: Threshold = Threshold('exp',-.5)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in ('ellipsoid_union', 'gaussian_sum') or not self.kernels:
            raise ValueError('Unsupported or empty kernel scene')
        if self.kind=='ellipsoid_union' and self.threshold!=Threshold('exp',-.5):
            raise ValueError('Ellipsoid union uses explicit per-kernel normalized exp(-1/2) levels')

    def approx_values(self, points):
        """Visualization/statistical estimates only, never physical evidence."""
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        values = np.full(len(p), -np.inf) if self.kind == 'ellipsoid_union' else np.zeros(len(p))
        for k in self.kernels:
            local = (p-k.center) @ k.approx_inverse.T
            q = np.einsum('ij,ij->i', local, local)
            if self.kind == 'ellipsoid_union':
                values = np.maximum(values, 1-q)
            else:
                values += k.weight*np.exp(-q/2)
        return values if self.kind == 'ellipsoid_union' else values-self.threshold.approximate()

    def parameters(self):
        return dict(kernels=[k.parameters() for k in self.kernels],
                    threshold={'kind':self.threshold.kind, 'value':self.threshold.value},
                    metadata=self.metadata)


@dataclass
class SpectralScene:
    frequencies: object
    cosine: object
    sine: object
    center: object
    radius: object
    threshold: Threshold
    bias: float = .5
    metadata: dict = field(default_factory=dict)
    kind: str = field(default='spectral_grf', init=False)

    def __post_init__(self):
        for name in ('frequencies', 'cosine', 'sine', 'center', 'radius'):
            value = np.array(getattr(self, name), dtype='<f8', copy=True)
            if not np.isfinite(value).all(): raise ValueError('Nonfinite spectral parameter')
            value.setflags(write=False); setattr(self, name, value)
        n = len(self.cosine)
        if self.frequencies.shape != (n,3) or self.sine.shape != (n,) or self.center.shape != (3,) or self.radius.shape != (3,) or np.any(self.radius <= 0) or not math.isfinite(self.bias):
            raise ValueError('Invalid finite spectral field')
        # This compact taper is part of the continuous Gaussian model,
        # not clipping a previously generated solid to the world.

    def approx_values(self, points):
        p = np.asarray(points).reshape(-1,3)
        taper = np.prod(np.maximum(0, 1-((p-self.center)/self.radius)**2)**2, axis=1)
        value = np.full(len(p), self.bias)
        for f,a,b in zip(self.frequencies,self.cosine,self.sine):
            phase = 2*np.pi*(p @ f)
            value += a*np.cos(phase)+b*np.sin(phase)
        return taper*value-self.threshold.approximate()

    def parameters(self):
        return {name:getattr(self,name) for name in ('frequencies','cosine','sine','center','radius','bias','metadata')} | {
            'threshold':{'kind':self.threshold.kind,'value':self.threshold.value}}


@dataclass
class ConvexScene:
    objects: list[dict]
    metadata: dict = field(default_factory=dict)
    kind: str = field(default='gaussian_convex', init=False)

    def parameters(self):
        return dict(objects=self.objects, metadata=self.metadata)

    def bundle(self):
        from rind.contracts import CSGNode,CSGOpcode,Geometry,RootKind,make_leaf
        from rind.storage import CanonicalBundle,pack_arrays,_decode_bundle
        b=CanonicalBundle(3);roots=[]
        for o in self.objects:
            params={name:o[name] for name in ('center','matrix')}
            if 'normal' in o: params.update(normal=o['normal'],plane_offset=o['plane_offset'])
            leaf=make_leaf('affine_box_v1' if 'normal' not in o else 'convex_halfspaces_v1',3,**params)
            b.add_leaf(leaf);roots.append(leaf.content_hash)
        if len(roots)>1:
            node=CSGNode(3,CSGOpcode.UNION,tuple(roots));b.add_node(node);root=node.content_hash
        else: root=roots[0]
        b.add_geometry(Geometry(3,RootKind.REF,root))
        return _decode_bundle(pack_arrays(b),3)

    def approx_values(self,points):
        p=np.asarray(points).reshape(-1,3);values=np.full(len(p),-np.inf)
        for o in self.objects:
            local=(p-o['center'])@np.linalg.inv(o['matrix']).T
            if 'normal' in o:
                value=np.min(np.asarray(o['plane_offset'])[:,None]-np.asarray(o['normal'])@local.T,axis=0)
            else: value=1-np.max(np.abs(local),axis=1)
            values=np.maximum(values,value)
        return values


def encode(value):
    if isinstance(value,np.ndarray):return encode(value.tolist())
    if isinstance(value,(float,np.floating)):return {'binary64':float(value).hex()}
    if isinstance(value,dict):return {k:encode(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [encode(v) for v in value]
    if isinstance(value,np.integer):return int(value)
    return value


def decode(value):
    if isinstance(value,dict):
        if set(value)=={'binary64'}:return float.fromhex(value['binary64'])
        return {k:decode(v) for k,v in value.items()}
    if isinstance(value,list):return [decode(v) for v in value]
    return value


def record(scene):
    payload={'schema':SCHEMA,'model':scene.kind,'parameters':encode(scene.parameters()),
             'boundary':'closed superlevel / closed convex union','ready_capable':False,
             'publication':'EXPERIMENTAL_ONLY'}
    text=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False)
    return payload | {'geometry_id':hashlib.sha256(text.encode()).hexdigest()}


def save(path,scene):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record(scene),sort_keys=True,indent=2)+'\n')


def load(path):
    raw=json.loads(Path(path).read_text());expected=raw.pop('geometry_id')
    if raw['schema']!=SCHEMA or raw['ready_capable'] is not False:
        raise ValueError('Experimental schema/publication gate mismatch')
    if hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=expected:
        raise ValueError('Experimental parameter record checksum mismatch')
    p=decode(raw['parameters']);kind=raw['model']
    if kind in ('gaussian_sum','ellipsoid_union'):
        p['kernels']=[Kernel(**v) for v in p['kernels']];p['threshold']=Threshold(**p['threshold'])
        return KernelScene(kind,**p)
    if kind=='spectral_grf':p['threshold']=Threshold(**p['threshold']);return SpectralScene(**p)
    if kind=='gaussian_convex':
        for o in p['objects']:
            for key in ('center','matrix','normal','plane_offset'):
                if key in o:o[key]=np.asarray(o[key])
        return ConvexScene(**p)
    raise ValueError('Unknown experimental geometry model')
