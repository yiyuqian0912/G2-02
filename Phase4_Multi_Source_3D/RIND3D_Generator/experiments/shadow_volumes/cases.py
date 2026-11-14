"""Frozen experimental families using existing canonical lossless codecs."""
import numpy as np
from rind.contracts import CSGNode,CSGOpcode,Geometry,RootKind,make_leaf
from rind.storage import CanonicalBundle,pack_arrays,_decode_bundle

NAMES=('box','rotated_box','thin_slab','prism','wedge','poly8','poly16','poly32','multiscale4','multiscale12','corridor','cluster')
SOURCES=np.array([[.071234567,.083456789,.052345678],[.93234567,.17345678,.77345678],[.45345678,.94345678,.09345678]],dtype='<f8')


def rotation(rng):
    q,r=np.linalg.qr(rng.normal(size=(3,3)));q=q@np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q)<0:q[:,0]*=-1
    return q


def make_case(name,seed=20261031):
    rng=np.random.default_rng(np.random.SeedSequence([seed,NAMES.index(name)]));bundle=CanonicalBundle(3);roots=[];objects=[]
    def add(center,matrix,normals=None,offsets=None):
        fields={'center':np.asarray(center),'matrix':np.asarray(matrix)}
        if normals is not None:fields.update(normal=normals,plane_offset=offsets)
        leaf=make_leaf('affine_box_v1' if normals is None else 'convex_halfspaces_v1',3,**fields);bundle.add_leaf(leaf);roots.append(leaf.content_hash)
        objects.append({'center':np.asarray(center).tolist(),'matrix':np.asarray(matrix).tolist(),'declared_planes':6 if normals is None else len(normals),'codec':leaf.codec.name})
    if name=='box':add([.5]*3,np.diag([.125]*3))
    elif name=='rotated_box':add([.5]*3,rotation(rng)@np.diag([.17,.12,.09]))
    elif name=='thin_slab':add([.5]*3,rotation(rng)@np.diag([.23,.19,2**-18]))
    elif name=='prism':
        angles=np.arange(6)*2*np.pi/6
        normals=[[np.cos(t),np.sin(t),0] for t in angles]+[[0,0,-1],[0,0,1]]
        add([.5]*3,rotation(rng)@np.diag([.16,.14,.2]),normals,[1]*8)
    elif name=='wedge':add([.5]*3,rotation(rng)@np.diag([.18,.16,.2]),[[-1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[1,0,1]],[1,1,1,1,.4])
    elif name.startswith('poly'):
        count=int(name[4:]);normals=np.r_[np.eye(3),-np.eye(3)];offsets=[1.]*6
        extra=rng.normal(size=(count-6,3));extra/=np.linalg.norm(extra,axis=1)[:,None]
        normals=np.r_[normals,extra];offsets += [.82]*(count-6)
        add([.5]*3,rotation(rng)@np.diag([.2,.16,.14]),normals,offsets)
    elif name.startswith('multiscale'):
        count=int(name[10:])
        for i in range(count):
            size=(.018,.035,.075,.14)[i%4]
            add(rng.uniform(.25,.75,3),rotation(rng)@np.diag(rng.uniform(.7,1.2,3)*size))
    elif name=='corridor':
        for c,h in [([.35,.5,.5],[.015,.28,.28]),([.65,.5,.5],[.015,.28,.28]),([.5,.75,.5],[.15,.015,.28]),([.5,.43,.34],[.065,.02,.1]),([.56,.6,.67],[.03,.08,.055])]:add(c,np.diag(h))
    elif name=='cluster':
        for i in range(8):add(np.array([.5,.55,.5])+rng.uniform(-.15,.15,3),rotation(rng)@np.diag(rng.uniform(.025,.08,3)))
    node=CSGNode(3,CSGOpcode.UNION,tuple(roots)) if len(roots)>1 else None
    if node:bundle.add_node(node)
    bundle.add_geometry(Geometry(3,RootKind.REF,node.content_hash if node else roots[0]))
    return _decode_bundle(pack_arrays(bundle),3),{'family':name,'seed':seed,'objects':objects,'sources':SOURCES.tolist(),'experimental_family':True,'production_ready':False}
