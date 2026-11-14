"""Canonical convex decoding and exact source-dependent shadow construction.

No mesh or sampled representation is used to define geometry. All binary64
inputs are interpreted as exact rational numbers before inverse/plane algebra.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
import numpy as np
from rind.contracts import CSGNode,CSGOpcode,CodecID,Leaf,RootKind
from rind3d.geometry import Unsupported

def exact_inverse(matrix):
    a=[[Q(float(v)) for v in row]+[Q(int(i==j)) for j in range(3)] for i,row in enumerate(matrix)]
    for j in range(3):
        pivot=next((i for i in range(j,3) if a[i][j]),None)
        if pivot is None:raise Unsupported('Singular canonical affine matrix')
        a[j],a[pivot]=a[pivot],a[j];v=a[j][j];a[j]=[x/v for x in a[j]]
        for i in range(3):
            if i!=j:
                v=a[i][j];a[i]=[x-v*y for x,y in zip(a[i],a[j])]
    result=[row[3:] for row in a]
    for i in range(3):
        for j in range(3):
            if sum(Q(float(matrix[i,k]))*result[k][j] for k in range(3))!=int(i==j):raise AssertionError('Inverse certificate failed')
    return result

def normalize(row):
    scale=max(map(abs,row))
    return tuple(v/scale for v in row) if scale else tuple(row)

@dataclass(frozen=True)
class ConvexComponent:
    leaf_id:str
    codec:str
    anchor:tuple
    rows:tuple

def decode_convex(bundle):
    refs={x.content_hash:x for x in bundle.references};components=[]
    def walk(key):
        item=refs[key]
        if isinstance(item,CSGNode):
            if item.opcode!=CSGOpcode.UNION:raise Unsupported('Native scope requires top-level convex unions')
            for child in item.children:walk(child)
            return
        if not isinstance(item,Leaf) or item.codec not in (CodecID.AFFINE_BOX,CodecID.CONVEX_HALFSPACES):
            raise Unsupported('Native scope supports affine boxes and convex halfspaces only')
        c=tuple(Q(float(v)) for v in item.field('center').as_array());matrix=item.field('matrix').as_array();inv=exact_inverse(matrix)
        if item.codec==CodecID.AFFINE_BOX:
            normals=[[Q(sign*int(k==axis)) for k in range(3)] for axis in range(3) for sign in (-1,1)];offsets=[Q(1)]*6
        else:
            normals=[[Q(float(v)) for v in row] for row in item.field('normal').as_array()]
            offsets=[Q(float(v)) for v in item.field('plane_offset').as_array()]
        if any(v<=0 or v>=1 for v in c) or any(b<=0 for b in offsets):
            raise Unsupported('Native scope requires strict component-anchor interior witnesses in the world')
        if len(normals)>64:raise Unsupported('Native resource capability: at most 64 planes/component')
        rows=[]
        for a,b in zip(normals,offsets):
            n=[sum(a[i]*inv[i][j] for i in range(3)) for j in range(3)]
            d=-b-sum(x*y for x,y in zip(n,c))
            if any(n):rows.append(normalize((*n,d)))
        components.append(ConvexComponent(item.content_hash.hex(),item.codec.name,c,tuple(dict.fromkeys(rows))))
    root=bundle.geometries[0]
    if root.root_kind!=RootKind.REF:raise Unsupported('Native scope requires nonempty bounded convex components')
    walk(root.root_hash)
    if not 1<=len(components)<=64:raise Unsupported('Native resource capability: 1..64 components')
    return tuple(components)

def shadow_planes(body,source):
    p=tuple(Q(float(v)) for v in source);c=[-row[3]-sum(a*x for a,x in zip(row[:3],p)) for row in body]
    negative=[i for i,v in enumerate(c) if v<0];positive=[i for i,v in enumerate(c) if v>0]
    if not negative:raise ValueError('Source must be outside every closed component')
    rows=[body[i] for i in negative]
    for i,v in enumerate(c):
        if not v:rows.append((*body[i][:3],-sum(a*x for a,x in zip(body[i][:3],p))))
    for i in negative:
        for j in positive:
            n=tuple(c[j]*body[i][k]-c[i]*body[j][k] for k in range(3));d=-sum(a*x for a,x in zip(n,p));row=(*n,d)
            if any(row):rows.append(row)
    return tuple(dict.fromkeys(normalize(row) for row in rows))

def rational_rows(bodies):return [[[str(v) for v in row] for row in body] for body in bodies]

def geometry_split(geometry_id,seed=0):
    import hashlib
    value=int.from_bytes(hashlib.sha256(f'{seed}/{geometry_id}/geometry-group-v1'.encode()).digest()[:8],'little')%100
    return 'train' if value<80 else 'val' if value<90 else 'test'
