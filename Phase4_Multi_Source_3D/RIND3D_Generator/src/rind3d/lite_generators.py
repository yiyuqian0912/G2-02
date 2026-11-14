"""Versioned randomized canonical convex recipes; no rendering-based selection."""
import numpy as np
from rind.contracts import CSGNode,CSGOpcode,Geometry,RootKind,make_leaf
from rind.storage import CanonicalBundle,pack_arrays,_decode_bundle
from .geometry import keyed_rng

VERSION='lite-convex-recipes-v0.1'
FAMILIES=('rotated_boxes','thin_slabs','prisms','wedges','convex_polyhedra',
          'multiscale_convex','corridors','barriers','occluder_clusters')
DISTRIBUTION={'version':VERSION,'world_inset_proposal':.01,'orientations':'uniform unit-quaternion SO(3), canonical binary64 result',
              'centres':'uniform in conservative world-fitting intervals; structured recipes use a shared random frame',
              'thin_half_thickness_log2':[-20,-5],'polyhedron_planes':[8,12,16,24,32],
              'object_count_ranges':{'rotated_boxes':[2,10],'thin_slabs':[2,8],'prisms':[2,6],'wedges':[2,6],
                                     'convex_polyhedra':[1,5],'multiscale_convex':[4,24],'corridors':[4,8],'barriers':[4,12],'occluder_clusters':[4,16]},
              'selection':'no renderer timing or candidate-count selection; bounded attempts recorded',
              'world_scope':'all components strictly inside world; not unrestricted world-clipped geometry'}

def rotation(rng):
    u,v,w=rng.random(3);x=np.sqrt(1-u)*np.sin(2*np.pi*v);y=np.sqrt(1-u)*np.cos(2*np.pi*v)
    z=np.sqrt(u)*np.sin(2*np.pi*w);q=np.sqrt(u)*np.cos(2*np.pi*w)
    return np.array([[1-2*(y*y+z*z),2*(x*y-q*z),2*(x*z+q*y)],
                     [2*(x*y+q*z),1-2*(x*x+z*z),2*(y*z-q*x)],
                     [2*(x*z-q*y),2*(y*z+q*x),1-2*(x*x+y*y)]],dtype='<f8')

def log_uniform(rng,lo,hi,size=None):return np.exp(rng.uniform(np.log(lo),np.log(hi),size))

def generate(seed,slot,family,attempt=0):
    if family not in FAMILIES:raise ValueError('Uncertified Lite generator family: '+str(family))
    rng=keyed_rng(seed,slot,8101+attempt*97);bundle=CanonicalBundle(3);roots=[];objects=[];proposal_failures=[]
    lo,hi=DISTRIBUTION['object_count_ranges'][family];count=int(rng.integers(lo,hi+1));frame=rotation(rng)
    anchor=rng.uniform(.4,.6,3);layout_scale=float(rng.uniform(.18,.29))
    def add(kind,index):
        rot=rotation(rng);normals=offsets=None;radius=np.ones(3)
        if kind=='thin':half=np.r_[log_uniform(rng,.07,.24,2),2.**rng.uniform(-20,-5)]
        elif kind=='prism':
            sides=int(rng.integers(3,11));angles=np.arange(sides)*2*np.pi/sides
            normals=np.r_[np.c_[np.cos(angles),np.sin(angles),np.zeros(sides)],[[0,0,-1],[0,0,1]]]
            offsets=np.ones(sides+2);half=log_uniform(rng,.02,.12,3);radius=np.array([2.1,2.1,1.])
        elif kind=='wedge':
            normals=np.array([[-1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[1,0,1]],dtype=float)
            offsets=np.array([1,1,1,1,.4]);half=log_uniform(rng,.02,.16,3);radius=np.full(3,1.5)
        elif kind=='poly':
            planes=int(rng.choice([8,12,16,24,32]));extra=rng.normal(size=(planes-6,3));extra/=np.linalg.norm(extra,axis=1)[:,None]
            normals=np.r_[np.eye(3),-np.eye(3),extra];offsets=np.r_[np.ones(6),rng.uniform(.58,1.05,planes-6)]
            half=log_uniform(rng,.035,.22,3)
        elif kind=='small':half=log_uniform(rng,.008,.04,3)
        elif kind=='large':half=log_uniform(rng,.07,.24,3)
        else:half=log_uniform(rng,.025,.22,3)
        if family in ('corridors','barriers'):
            # Walls and baffles in a shared randomly oriented local frame.
            rot=frame if family=='corridors' else frame@rotation(rng)
            half=np.array([rng.uniform(.002,.014),layout_scale*rng.uniform(.55,1.),layout_scale*rng.uniform(.55,1.)])
        matrix=np.asarray(rot@np.diag(half),dtype='<f8');extent=np.abs(matrix)@radius
        if np.any(extent>=.49):
            centre=np.full(3,.5);proposal_failures.append({'object':index,'reason':'conservative_world_fitting_interval_empty'})
        else:
            centre=rng.uniform(extent+.01,1-extent-.01)
            if family=='occluder_clusters':centre=np.clip(anchor+rng.normal(0,.09,3),extent+.01,1-extent-.01)
            elif family in ('corridors','barriers'):
                local=np.array([(-1 if index%2 else 1)*layout_scale*.65,rng.uniform(-.12,.12),rng.uniform(-.1,.1)])
                if family=='barriers':local[0]=(index/(count-1)-.5)*.5
                centre=np.clip(anchor+frame@local,extent+.01,1-extent-.01)
        fields={'center':np.asarray(centre,dtype='<f8'),'matrix':matrix}
        if normals is not None:fields.update(normal=np.asarray(normals,dtype='<f8'),plane_offset=np.asarray(offsets,dtype='<f8'))
        item=make_leaf('affine_box_v1' if normals is None else 'convex_halfspaces_v1',3,**fields)
        bundle.add_leaf(item);roots.append(item.content_hash)
        objects.append({'kind':kind,'leaf_id':item.content_hash.hex(),'center':centre.tolist(),'matrix':matrix.tolist(),
                        'half_parameters':np.asarray(half).tolist(),'declared_planes':6 if normals is None else len(normals),
                        'orientation':rot.tolist(),'conservative_extent':extent.tolist()})
    for index in range(count):
        if family=='thin_slabs':kind='thin'
        elif family=='prisms':kind='prism'
        elif family=='wedges':kind='wedge'
        elif family=='convex_polyhedra':kind='poly'
        elif family=='multiscale_convex':kind=str(rng.choice(['small','large','prism','poly'],p=[.4,.35,.15,.1]))
        elif family=='occluder_clusters':kind=str(rng.choice(['small','box','thin'],p=[.5,.25,.25]))
        else:kind='box'
        add(kind,index)
    if len(roots)>1:
        node=CSGNode(3,CSGOpcode.UNION,tuple(roots));bundle.add_node(node);root=node.content_hash
    else:root=roots[0]
    bundle.add_geometry(Geometry(3,RootKind.REF,root));bundle=_decode_bundle(pack_arrays(bundle),3)
    return bundle,{'version':VERSION,'family':family,'seed':int(seed),'slot':int(slot),'attempt':int(attempt),
                   'geometry_id':bundle.geometries[0].content_hash.hex(),'objects':objects,'proposal_failures':proposal_failures,
                   'distribution':DISTRIBUTION,'source_selection':'independent uniform world proposals with exact FREE/clearance rejection'}
