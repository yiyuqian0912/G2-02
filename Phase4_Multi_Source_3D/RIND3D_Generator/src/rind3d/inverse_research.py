"""Conservative inverse-source constraints. Isolated; never a Core dependency.

The true feasible set is an intersection of unions/complements. Each receiver's
complete convex union is queried independently. UNKNOWN boxes are retained.
Sparse constraints certify only an OUTER cover, never full-window solutions.
"""
from collections import Counter
from fractions import Fraction as Q
import heapq,time
import numpy as np
from .projection_research import ProjectionSource,topology,value,silhouette_planes
from .rendering.polyhedra import rational_rows
from . import _research
from .windows import sobol_points,excludes

class InverseReceiver:
    """Minimal exact reverse projection; no unused rendering/report setup."""
    def __init__(self,context,receiver,topo,solid_rows):
        self.source=np.array(receiver,copy=True);self.planes=tuple(silhouette_planes(b,self.source,t) for b,t in zip(context.bodies,topo))
        self.native=_research.Projection(solid_rows,rational_rows(self.planes))
    def response(self,source,points):return self.native.points(np.ascontiguousarray(points,dtype='<f8'))[0].astype(bool)

class InverseConstraints:
    def __init__(self,context,points,labels):
        start=time.perf_counter();cpu=time.process_time();self.context=context
        self.points=np.ascontiguousarray(points,dtype='<f8').reshape(-1,3);self.labels=np.asarray(labels,dtype=bool)
        if len(self.points)!=len(self.labels) or not len(self.points) or np.any(context.contains(self.points)):raise ValueError('Inverse observations must be FREE and labelled')
        topo=topology(context);solid_rows=rational_rows(context.bodies);self.models=[InverseReceiver(context,x,topo,solid_rows) for x in self.points]
        self.setup_seconds=time.perf_counter()-start;self.setup_cpu_seconds=time.process_time()-cpu
        self.calls=0;self.rejections=Counter()
    def possible(self,lo,hi):
        # Reject a lit constraint only for union-covered shadow; reject a shadow
        # constraint only if every obstacle cone is certified absent throughout.
        self.last_rejection=None
        for i,(model,label) in enumerate(zip(self.models,self.labels)):
            self.calls+=1;y,m=model.native.certificate(list(lo),list(hi))
            if m==1:self.last_rejection=('solid_source_box',i);self.rejections['solid_source_box']+=1;return False
            if y>=0 and bool(y)!=bool(label):self.last_rejection=('lit_violated' if label else 'shadow_violated',i);self.rejections[self.last_rejection[0]]+=1;return False
        return True
    def screen(self,points):
        p=np.asarray(points,dtype='<f8').reshape(-1,3);keep=~self.context.contains(p)
        for model,label in zip(self.models,self.labels):
            ids=np.flatnonzero(keep)
            if not len(ids):break
            keep[ids]&=model.response(model.source,p[ids])==label
        return keep

def max_distance_sq(lo,hi,p):return sum(max(abs(Q(a)-Q(float(x))),abs(Q(b)-Q(float(x))))**2 for a,b,x in zip(lo,hi,p))

def explore(constraints,truth,window,*,seconds=.2,max_regions=512,max_depth=6,seed=0,separation=32/1024,record_rejections=False):
    """Partition-preserving outer cover; timed out pending cells remain counted.

    Floats here are dyadic subdivision endpoints interpreted exactly. Separation
    rejection uses rational squared distance. Source clearance is checked only
    for proposed points, so untested clearance can enlarge (never shrink) cover.
    """
    if not 1<=max_depth<=16 or max_regions<1 or seconds<0:raise ValueError('Bounded nonnegative inverse search budget required')
    start=time.perf_counter();heap=[(-1.,0,(0.,0.,0.),(1.,1.,1.),0)];leaves=[];candidates=[];serial=1;rejected=Counter();evaluated=0;by_depth=Counter();wlo=np.asarray(window.origin)/1024;whi=(np.asarray(window.origin)+window.size)/1024
    initial_calls=constraints.calls;rng=np.random.default_rng(seed);trace=[]
    def record_box(lo,hi,depth,reason,observation=None):
        if record_rejections:trace.append({'lower':lo.tolist(),'upper':hi.tolist(),'depth':depth,'reason':reason,'observation':observation})
    while heap and evaluated<max_regions and time.perf_counter()-start<seconds:
        negvol,_,lo,hi,depth=heapq.heappop(heap);evaluated+=1;by_depth[depth]+=1;lo=np.array(lo);hi=np.array(hi)
        if np.all(lo>=wlo) and np.all(hi<whi):rejected['inside_window']+=1;record_box(lo,hi,depth,'inside_window');continue
        if separation>0 and max_distance_sq(lo,hi,truth)<Q(float(separation))**2:rejected['too_near_truth']+=1;record_box(lo,hi,depth,'too_near_truth');continue
        if not constraints.possible(lo,hi):
            rejected['necessary_observation_violation']+=1;reason,index=constraints.last_rejection;record_box(lo,hi,depth,reason,index);continue
        if depth==max_depth:
            leaves.append((tuple(lo),tuple(hi),depth));candidates.extend([((lo+hi)/2).tolist(),(lo+(hi-lo)*rng.random(3)).tolist()]);continue
        # Visit multiple components: level/volume priority, seeded tie ordering.
        mid=(lo+hi)/2
        for mask in rng.permutation(8):
            a=np.where([(mask&(1<<k))!=0 for k in range(3)],mid,lo);b=np.where([(mask&(1<<k))!=0 for k in range(3)],hi,mid)
            heapq.heappush(heap,(negvol/8,serial,tuple(a),tuple(b),depth+1));serial+=1
    remaining=[(a,b,d) for _,_,a,b,d in heap];cover=leaves+remaining
    if not candidates:
        candidates=[((np.array(a)+b)/2).tolist() for a,b,_ in cover[:256]]
    candidates=np.asarray(candidates,dtype='<f8').reshape(-1,3)
    # Numerical proposals are hypotheses, not certified members of the inverse.
    candidates=candidates[np.array([np.all(p>0) and np.all(p<1) and excludes(window,p,1024) and np.linalg.norm(p-truth)>=separation for p in candidates],dtype=bool)]
    exact_volume=sum((Q(1,2**(3*d)) for _,_,d in cover),Q(0));volume=float(exact_volume)
    report={'declared_seconds':seconds,'declared_max_regions':max_regions,'declared_max_depth':max_depth,'seed':seed,'separation':separation,
            'candidate_domain':'world_outside_window; free source necessary; clearance left as an outer relaxation',
            'evaluated_regions':evaluated,'surviving_final_cells':len(leaves),'pending_unresolved_cells':len(remaining),'outer_cover_volume_upper_bound':volume,'outer_cover_volume_exact':str(exact_volume),
            'volume_semantics':'Certified outer bound for sparse observations AND declared source domain/separation; unknown and unvisited cells included; not an exact feasible volume.',
            'certified_inner_boxes':0,'rejected':dict(rejected),'regions_by_depth':dict(by_depth),'constraint_calls':constraints.calls-initial_calls,
            'termination':'complete_declared_resolution' if not heap else 'region_budget' if evaluated>=max_regions else 'time_budget',
            'complete_full_window_inverse':False,'uniqueness_proven':False,'wall_seconds':time.perf_counter()-start,
            'outer_cells':[{'lower':a,'upper':b,'depth':d} for a,b,d in cover]}
    if record_rejections:report['rejected_cells']=trace
    return candidates,report

def local_proposals(truth,*,count=512,seed=0,minimum=32/1024):
    rng=np.random.default_rng(seed);direction=rng.normal(size=(count,3));direction/=np.linalg.norm(direction,axis=1)[:,None]
    radius=np.exp(rng.uniform(np.log(max(minimum,2**-22)),np.log(.35),count));return np.asarray(truth)+direction*radius[:,None]

def silhouette_proposals(constraints,truth,*,count=512,seed=0):
    # Approximate SVD constructs proposals only; never a source certificate.
    p=tuple(Q(float(x)) for x in truth);rows=[]
    for model in constraints.models:
        for body in model.planes:
            for row in body:
                n=np.array(list(map(float,row[:3])));norm=np.linalg.norm(n)
                if norm:rows.append((abs(float(value(row,p)))/norm,n/norm))
    rows.sort(key=lambda r:r[0]);normals=np.array([n for _,n in rows[:min(12,len(rows))]])
    _,singular,vh=np.linalg.svd(normals,full_matrices=True);directions=local_proposals([0,0,0],count=count,seed=seed)
    weights=1/(1+32*np.pad(singular,(0,max(0,3-len(singular))))[:3]);scaled=(directions@vh.T)*weights;scaled=scaled@vh
    norm=np.linalg.norm(scaled,axis=1);scaled*=np.maximum(1,(32/1024)/np.maximum(norm,1e-300))[:,None]
    return np.asarray(truth)+scaled,{'heuristic_singular_values':singular.tolist(),'rank_is_not_uniqueness':True,'approximate_plane_intersections_are_proposals_only':True}
