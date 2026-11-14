"""Per-scene mathematical acceptance, independently checkable certificates.

Renderer capability is insufficient for publication. This module enforces the
canonical structural/round-trip, topology, probe, source and physics obligations.
"""
from fractions import Fraction as Q
import hashlib,json
from pathlib import Path
import time
import numpy as np
from rind.contracts import make_physics,make_scene
from rind.storage import pack_arrays,_decode_bundle
from rind.generation import DRIVER_CLEARANCE
from rind.geometry import Membership
from .geometry import NumericalUnresolved,Unsupported,keyed_rng
from .lite_generators import FAMILIES
from .rendering.context import GeometryContext

PROTOCOL='lite-convex-acceptance-v0.1'
DRIVER_RADIUS=float(DRIVER_CLEARANCE)

def verify_geometry_proof(bundle,proof):
    from .rendering.polyhedra import decode_convex
    geometry=bundle.geometries[0]
    if proof.get('protocol')!='convex-interior-and-world-dual-v1' or proof.get('geometry_id')!=geometry.content_hash.hex():raise ValueError('Geometry proof identity/protocol mismatch')
    roundtrip=_decode_bundle(pack_arrays(bundle),3)
    original=pack_arrays(bundle);restored=pack_arrays(roundtrip)
    if original.keys()!=restored.keys() or geometry.content_hash!=roundtrip.geometries[0].content_hash or any(original[k].dtype!=restored[k].dtype or original[k].shape!=restored[k].shape or original[k].tobytes()!=restored[k].tobytes() for k in original):raise ValueError('Canonical roundtrip failed')
    components=decode_convex(bundle)
    if len(components)!=len(proof['components']):raise ValueError('Component certificate count mismatch')
    for component,report in zip(components,proof['components']):
        if component.leaf_id!=report['leaf_id']:raise ValueError('Component identity mismatch')
        c=component.anchor;r=Q(report['interior_radius'])
        if r<=0:raise ValueError('No positive interior certificate')
        for row in component.rows:
            v=sum(a*x for a,x in zip(row[:3],c))+row[3];norm=sum(a*a for a in row[:3])
            if v>=0 or v*v<=r*r*norm:raise ValueError('Invalid interior-ball proof')
        vertices=[tuple(Q(q) for q in v) for v in report['vertices']]
        if len(set(vertices))!=len(vertices) or len(vertices)<4:raise ValueError('Invalid exact vertex certificate')
        for v in vertices:
            if any(x<=0 or x>=1 for x in v):raise ValueError('Lite v0.1 requires a strict world-face gap')
            for row in component.rows:
                if sum(a*x for a,x in zip(row[:3],v))+row[3]>0:raise ValueError('Vertex outside canonical component')
        # Six independently checkable Farkas/LP duals prove global bounds for
        # EVERY point of the canonical polytope, not merely listed vertices.
        duals=report.get('bound_duals',[])
        if {(d['axis'],d['sign']) for d in duals}!={(a,s) for a in range(3) for s in (-1,1)} or len(duals)!=6:raise ValueError('Missing world-bound dual certificates')
        for dual in duals:
            axis,sign=dual['axis'],dual['sign'];normal=[Q(0)]*3;bound=Q(0)
            if len(dual['planes'])!=len(dual['weights']) or not 1<=len(dual['planes'])<=3:raise ValueError('Invalid dual support')
            for index,weight in zip(dual['planes'],dual['weights']):
                if not 0<=index<len(component.rows):raise ValueError('Invalid dual plane')
                w=Q(weight);row=component.rows[index]
                if w<0:raise ValueError('Negative bound dual multiplier')
                for a in range(3):normal[a]+=w*row[a]
                bound-=w*row[3]
            if normal!=[Q(sign if a==axis else 0) for a in range(3)] or bound!=Q(dual['bound']):raise ValueError('Invalid bound dual equality')
            expected=Q(report['upper'][axis]) if sign==1 else -Q(report['lower'][axis])
            if bound!=expected or (bound>=1 if sign==1 else bound>=0):raise ValueError('World containment not proved')
        gap=min([Q(x) for x in report['lower']]+[1-Q(x) for x in report['upper']])
        if Q(report['world_gap'])!=gap or gap<=0 or not report['strictly_inside_world']:raise ValueError('World-face condition not proved')
    return True

def verify_clearance(context,point,proof):
    p=tuple(Q(float(v)) for v in point);r=Q(DRIVER_RADIUS)
    if any(x<=0 or x>=1 for x in p) or proof.get('status')!='GT' or proof.get('radius_hex')!=DRIVER_RADIUS.hex():raise ValueError('Invalid accepted source clearance')
    if proof['proof']=='strict-separating-planes':
        if len(proof['planes'])!=len(context.bodies):raise ValueError('Clearance component coverage mismatch')
        for body,index in zip(context.bodies,proof['planes']):
            if not 0<=int(index)<len(body):raise ValueError('Invalid clearance plane')
            row=body[int(index)];value=sum(a*x for a,x in zip(row[:3],p))+row[3]
            if value<=0 or value*value<=r*r*sum(a*a for a in row[:3]):raise ValueError('Invalid strict clearance separator')
    elif proof['proof']=='canonical-clearance':
        if context.clearance(point,DRIVER_RADIUS)['status']!='GT':raise ValueError('Canonical clearance recheck failed')
    else:raise ValueError('Unknown clearance proof')
    return True

def geometry_acceptance(context,family,seed,slot,attempt=0):
    start=time.perf_counter()
    if family not in FAMILIES:raise Unsupported('Family has no Lite acceptance contract')
    proof=context.geometry_proof();verify_geometry_proof(context.bundle,proof)
    rng=keyed_rng(seed,slot,9011+attempt*17)
    from itertools import product
    fixed=np.array(list(product((0.,1.),repeat=3))+[(.5,.5,.5)])
    random=rng.random((16,3));points=np.r_[fixed,random];membership=context.membership(points)
    if np.any((membership<0)|(membership>2)):raise NumericalUnresolved('Acceptance membership unresolved')
    if np.any(membership[:8]!=int(Membership.FREE)):raise ValueError('World-face certificate contradicts corner probes')
    # A fixed world-near-corner source is proven free by the world-gap profile.
    gap=min(Q(c['world_gap']) for c in proof['components']);corner=float(min(gap/4,Q(1,1024)))
    if not corner>0:raise NumericalUnresolved('No representable anchor for acceptance rays')
    source=np.full(3,corner);model=context.bind(source);independent=context.bind(source,independent=True)
    starts=np.r_[random[::2],random[1::2],random[:4]];ends=np.r_[random[1::2],random[::2],random[:4]]
    decisions=[]
    for a,b in zip(starts,ends):
        # Ray sources in obstacles are still legal closed-segment queries for
        # this gate; a solid endpoint immediately proves HIT.
        if context.contains(np.array([a,b])).any():decisions.append(1)
        else:decisions.append(int(not context.bind(a).response(a,b[None])[0]))
    if decisions[:8]!=decisions[8:16]:raise AssertionError('Acceptance segment reversal failed')
    boxes=[[[0,0,0],[1,1,1]],[[0,0,0],[.5,1,1]],[[.5,0,0],[1,1,1]]]
    classes=[int(context.box_class(*box)) for box in boxes]
    y=model.response(source,random);other=independent.response(source,random)
    if not np.array_equal(y,other) or np.any(y&context.contains(random)):raise AssertionError('Acceptance physical probes disagree')
    return {'protocol':PROTOCOL,'family':family,'geometry_id':context.geometry_id,'geometry_proof':proof,
            'canonical_codecs_valid':True,'geometry_roundtrip':True,'transforms_nonsingular':True,
            'topology_proof':'finite union of bounded full-dimensional convex closed components strictly inside Ω',
            'domain_faces':'entire structure has certified positive gap from every Ω face',
            'empty':False,'full':False,'point_inputs':points.tolist(),'point_membership':membership.tolist(),
            'segment_starts':starts.tolist(),'segment_ends':ends.tolist(),'segment_decisions':decisions,
            'box_bounds':boxes,'box_classes':classes,'physical_probe_source':source.tolist(),
            'physical_probe_response':y.tolist(),'passed':True,'seconds':time.perf_counter()-start,
            'canonical_fallback_work':dict(context.work)}

def sample_source(context,seed,slot,attempt=0,limit=256):
    rng=keyed_rng(seed,slot,10007+attempt*31);rejections=[]
    for index in range(limit):
        p=rng.random(3)
        if not np.all((p>0)&(p<1)):rejections.append('outside_open_world');continue
        if context.contains(p[None])[0]:rejections.append('solid');continue
        proof=context.clearance(p,DRIVER_RADIUS)
        if proof['status']!='GT':rejections.append('clearance_LE');continue
        verify_clearance(context,p,proof)
        scene=make_scene(context.geometry.content_hash,3,make_physics(0),p[None])
        if scene.driver_position.as_array().shape!=(1,3):raise ValueError('Single source physics contract failed')
        return p,{'position':p.tolist(),'clearance':proof,'proposals':index+1,'rejections':rejections,
                  'canonical_scene_id':scene.content_hash.hex(),'physics':'binary visibility / unit source'}
    raise NumericalUnresolved('Declared source-proposal budget exhausted; no certified source')
