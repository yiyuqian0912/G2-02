"""Independent check of a continuous source-space OUTER cover.

Uses the UNCHANGED full-FME projection verifier with its original interval
arithmetic; ambiguous boxes are checked by exact Fraction extrema. The octree
partition is independently reconstructed from terminal cells. This can prove
absence at a separation scale, never continuous uniqueness by candidate count.
"""
from fractions import Fraction as Q
import hashlib,json,time
import numpy as np
from .rendering.polyhedra import shadow_planes
from .inverse_research import max_distance_sq

def exact_union(bodies,lo,hi):
    uncertain=False
    for body in bodies:
        lower=[];upper=[]
        for row in body:
            lower.append(row[3]+sum(a*(lo[k] if a>=0 else hi[k]) for k,a in enumerate(row[:3])))
            upper.append(row[3]+sum(a*(hi[k] if a>=0 else lo[k]) for k,a in enumerate(row[:3])))
        if all(v<=0 for v in upper):return 1
        uncertain|=not any(v>0 for v in lower)
    return -1 if uncertain else 0

def partition(cells):
    trie={}
    for cell in cells:
        depth=cell['depth']
        if not isinstance(depth,int) or not 0<=depth<=16:raise ValueError('Invalid dyadic depth')
        lo=tuple(Q(float(v)) for v in cell['lower']);hi=tuple(Q(float(v)) for v in cell['upper']);step=Q(1,2**depth)
        if len(lo)!=3 or len(hi)!=3 or any(a<0 or b>1 or b-a!=step or (a*2**depth).denominator!=1 for a,b in zip(lo,hi)):raise ValueError('Invalid dyadic cell')
        ids=[int(a*2**depth) for a in lo];node=trie
        for level in range(depth):
            if 'leaf' in node:raise ValueError('Overlapping ancestor/descendant source cells')
            bit=depth-level-1;octant=sum(((ids[k]>>bit)&1)<<k for k in range(3));node=node.setdefault(octant,{})
        if node:raise ValueError('Duplicate/overlapping source cell')
        node['leaf']=True
    def complete(node):
        if 'leaf' in node:return len(node)==1
        return set(node)==set(range(8)) and all(complete(node[k]) for k in range(8))
    if not complete(trie):raise ValueError('Source-space cover has a hole')
    return True

def verify(context,points,labels,truth,window,report,*,separation=0):
    start=time.perf_counter();points=np.asarray(points,dtype='<f8');labels=np.asarray(labels,dtype=bool);truth=np.asarray(truth,dtype='<f8')
    if len(points)!=len(labels) or np.any(context.contains(points)):raise ValueError('Invalid independent inverse observations')
    from .windows import excludes
    from .lite_acceptance import verify_geometry_proof,verify_clearance,DRIVER_RADIUS
    verify_geometry_proof(context.bundle,context.geometry_proof());verify_clearance(context,truth,context.clearance(truth,DRIVER_RADIUS))
    if not excludes(window,truth,1024):raise ValueError('True source inside observation')
    for point in points:
        for coordinate,origin in zip(point,window.origin):
            index=Q(float(coordinate))*1024-Q(1,2)
            if index.denominator!=1 or not origin<=index<origin+window.size:raise ValueError('Constraint is not an observed voxel centre')
    if not np.array_equal(context.bind(truth).response(truth,points),labels):raise ValueError('Inverse observation labels differ from physical truth')
    retained=report['outer_cells'];rejected=report.get('rejected_cells')
    if rejected is None:raise ValueError('Missing rejected-cell certificate')
    partition([*retained,*rejected]);models={};planes={};exact=0;window_lo=[Q(int(x),1024) for x in window.origin];window_hi=[Q(int(x+window.size),1024) for x in window.origin]
    for cell in rejected:
        lo=tuple(Q(float(v)) for v in cell['lower']);hi=tuple(Q(float(v)) for v in cell['upper']);reason=cell['reason']
        if reason=='inside_window':
            if not all(a>=b and c<d for a,b,c,d in zip(lo,window_lo,hi,window_hi)):raise ValueError('Invalid source-exclusion pruning')
            continue
        if reason=='too_near_truth':
            if separation<=0 or max_distance_sq(lo,hi,truth)>=Q(float(separation))**2:raise ValueError('Invalid separation pruning')
            continue
        i=cell['observation']
        if not isinstance(i,int) or not 0<=i<len(points):raise ValueError('Invalid constraint witness index')
        if i not in models:
            models[i]=context.bind(points[i],independent=True);planes[i]=tuple(shadow_planes(b,points[i]) for b in context.bodies)
        y,m=models[i].native.certificate(cell['lower'],cell['upper'],True)
        good=m==1 if reason=='solid_source_box' else y==0 if reason=='lit_violated' else y==1 if reason=='shadow_violated' else False
        if not good:
            exact+=1;m=exact_union(context.bodies,lo,hi);s=exact_union(planes[i],lo,hi);y=-1 if s<0 else 1-s
            good=m==1 if reason=='solid_source_box' else bool(labels[i]) and y==0 if reason=='lit_violated' else not bool(labels[i]) and y==1 if reason=='shadow_violated' else False
        if not good:raise ValueError('Independent full-FME constraint did not certify excluded cell')
        if reason=='lit_violated' and not labels[i] or reason=='shadow_violated' and labels[i]:raise ValueError('Pruning observation label reversed')
    exact_volume=sum((Q(1,2**(3*c['depth'])) for c in retained),Q(0))
    if exact_volume!=Q(report['outer_cover_volume_exact']):raise ValueError('Incorrect outer volume')
    if not retained or not any(all(Q(float(a))<=Q(float(p))<=Q(float(b)) for a,p,b in zip(c['lower'],truth,c['upper'])) for c in retained):raise ValueError('Outer cover lost true source')
    radius=max(max_distance_sq(c['lower'],c['upper'],truth) for c in retained)
    inputs={'geometry_id':context.geometry_id,'policy':context.policy_hash,'true_source_bytes':truth.tobytes().hex(),'point_bytes':points.tobytes().hex(),
            'label_bytes':labels.astype(np.uint8).tobytes().hex(),'window':list(window.origin)+[window.size],'separation_hex':float(separation).hex(),
            'retained':retained,'rejected':rejected}
    digest=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
    return {'complete':True,'protocol':'independent-full-FME-source-cover-v1','geometry_id':context.geometry_id,'policy':context.policy_hash,
            'partition_complete':True,'retained_cells':len(retained),'independently_rechecked_rejected_cells':len(rejected),'exact_fraction_box_checks':exact,
            'outer_volume_exact':str(exact_volume),'maximum_truth_distance_squared_upper_exact':str(radius),
            'no_source_at_distance_32_over_1024_or_more':radius<Q(1,32)**2,'continuous_uniqueness_proven':False,
            'full_window_inverse_is_subset_of_verified_sparse_outer_cover':True,'certificate_sha256':digest,'wall_seconds':time.perf_counter()-start}
