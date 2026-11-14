# Legacy four-side diagnostic; archived results/source snapshots are unchanged.
"""Isolated constructive pilot sampler, NOT imported by rind3d production.

All information witnesses are exact sampled voxel centres. Spatial indexing,
midpoint refinement and analytic origin boxes never assign a physical label.
Every new point calls the unchanged contains AND response oracle.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from fractions import Fraction
from itertools import product
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from rind3d.geometry import Budget,BudgetExhausted,Aborted,NumericalUnresolved
from rind3d.legacy.windows import Window,excludes,sobol_points,placement_summary
from rind3d.legacy.window_contract import WINDOW_SIDES,PILOT_SIDE_COUNTS,require_window_side,validate_window_config
from rind3d.pack import atomic_json
from boundary_experiment_support import plain


@dataclass(frozen=True)
class OriginBox:
    """Inclusive integer origin bounds, not bounds on sampled world points."""
    lower:tuple[int,int,int]
    upper:tuple[int,int,int]
    @property
    def count(self): return int(np.prod([max(0,b-a+1) for a,b in zip(self.lower,self.upper)]))
    def contains(self,o): return all(a<=v<=b for v,a,b in zip(o,self.lower,self.upper))


def subtract_box(box,forbidden):
    lo=np.maximum(box.lower,forbidden.lower);hi=np.minimum(box.upper,forbidden.upper)
    if np.any(lo>hi):return (box,)
    lower=list(box.lower);upper=list(box.upper);result=[]
    for axis in range(3):
        if lower[axis]<lo[axis]:
            end=upper.copy();end[axis]=int(lo[axis])-1
            result.append(OriginBox(tuple(lower),tuple(end)));lower[axis]=int(lo[axis])
        if upper[axis]>hi[axis]:
            start=lower.copy();start[axis]=int(hi[axis])+1
            result.append(OriginBox(tuple(start),tuple(upper)));upper[axis]=int(hi[axis])
    return tuple(result)


def legal_origin_region(positive_world,negative_world,source,side,resolution):
    """ALL origins for half-open containment, clipped and source-subtracted.

    Exact rational arithmetic interprets binary64 input coordinates without
    epsilon. Source subtraction matches the existing excludes() multiplication
    (exact for the power-of-two production resolution). Small-grid tests also
    exercise arbitrary points exactly on a lower/upper window boundary.
    """
    side=require_window_side(side)
    if isinstance(resolution,bool) or not isinstance(resolution,(int,np.integer)) or resolution<side:
        raise ValueError('Resolution must fit the exact window side')
    p=np.asarray(positive_world,dtype=np.float64);q=np.asarray(negative_world,dtype=np.float64)
    s=np.asarray(source,dtype=np.float64)
    if any(v.shape!=(3,) or not np.isfinite(v).all() or np.any(v<0) or np.any(v>1) for v in (p,q,s)):
        raise ValueError('Invalid world coordinates')
    scaled=[[Fraction(float(v))*int(resolution) for v in point] for point in (p,q)]
    lower=tuple(max(0,max(a[axis]//1-side+1 for a in scaled)) for axis in range(3))
    upper=tuple(min(resolution-side,min(a[axis]//1 for a in scaled)) for axis in range(3))
    box=OriginBox(lower,upper)
    if not box.count:return ()
    # origin <= s*N < origin+L, exactly the existing half-open exclusion rule.
    source_voxels=s*resolution
    forbidden=OriginBox(tuple(int(np.floor(x-side))+1 for x in source_voxels),
                        tuple(int(np.floor(x)) for x in source_voxels))
    return subtract_box(box,forbidden)


def pair_region(pair,source,side,resolution):
    return legal_origin_region((np.asarray(pair['positive'])+.5)/resolution,
                               (np.asarray(pair['negative'])+.5)/resolution,source,side,resolution)


def origin_options(region):
    """Finite deterministic representatives, already guaranteed legal."""
    options=set()
    for box in region:
        options.add(tuple((a+b)//2 for a,b in zip(box.lower,box.upper)))
        options.update(product(*zip(box.lower,box.upper)))
    return sorted(options)


def pair_id(p,q):
    return hashlib.sha256(np.asarray([p,q],dtype='<i4').tobytes()).hexdigest()[:24]


class ExactPointCache:
    def __init__(self,source,resolution,oracle,budget,limit,sink=None):
        self.source=np.asarray(source,dtype=np.float64);self.resolution=resolution
        self.oracle=oracle;self.budget=budget;self.limit=limit;self.sink=sink
        self.points={};self.cache_hits=0
    def query(self,coordinate,phase):
        raw=np.asarray(coordinate)
        if raw.shape!=(3,) or not np.issubdtype(raw.dtype,np.integer):raise ValueError('Witness must be an integer voxel coordinate')
        key=tuple(map(int,raw))
        if min(key)<0 or max(key)>=self.resolution:raise ValueError('Witness outside world')
        if key in self.points:self.cache_hits+=1;return self.points[key]
        if len(self.points)>=self.limit:raise BudgetExhausted('experimental_point_cap')
        self.budget.check();point=(raw+.5)/self.resolution
        # Even solid points query the unchanged response oracle. No inferred ray
        # labels and no primitive-filter or predicate optimization.
        solid=bool(self.oracle.contains(point[None],budget=self.budget)[0])
        response=bool(self.oracle.response(self.source,point[None],budget=self.budget)[0])
        if solid and response:raise NumericalUnresolved('Oracle says solid receiver is lit')
        row={'coordinate':list(key),'solid':solid,'response':response,
             'label':'solid' if solid else 'free_lit' if response else 'free_shadow','phase':phase,
             'source':self.source.tolist()}
        self.points[key]=row
        if self.sink:self.sink.write(json.dumps(row)+'\n');self.sink.flush()
        return row


def make_pairs(cache,settings):
    positive=sorted(p for p,row in cache.points.items() if not row['solid'] and row['response'])
    negative=sorted(p for p,row in cache.points.items() if not row['solid'] and not row['response'])
    if not positive or not negative:return []
    neg=np.asarray(negative);tree=cKDTree(neg);count=min(settings['pair_neighbors'],len(negative))
    _,indices=tree.query(np.asarray(positive),k=count,p=np.inf)
    indices=np.asarray(indices).reshape(len(positive),count)
    result=[];bucket_side=settings['pair_bucket_side']
    for a,neighbors in zip(positive,indices):
        # Sort ties explicitly; tree search is only indexing, never physics.
        for index in sorted(set(map(int,neighbors)),key=lambda i:(int(np.max(np.abs(np.asarray(a)-neg[i]))),negative[i])):
            b=negative[index];delta=np.asarray(a)-b;mid=(np.asarray(a)+b+1)/2
            result.append({'id':pair_id(a,b),'positive':list(a),'negative':list(b),
                           'distance_voxels':float(np.linalg.norm(delta)),'max_axis_voxels':int(np.max(np.abs(delta))),
                           'distance_world':float(np.linalg.norm(delta))/cache.resolution,
                           'max_axis_world':int(np.max(np.abs(delta)))/cache.resolution,
                           'midpoint_voxels':mid.tolist(),'bucket':(mid//bucket_side).astype(int).tolist(),
                           'source':cache.source.tolist()})
    return sorted(result,key=lambda r:(r['max_axis_voxels'],r['distance_voxels'],r['id']))


def select_diverse(pairs,source,resolution,settings,seed,scene,stream):
    result=[];audit={};regions={}
    for side in WINDOW_SIDES:
        candidates=[];total_origin_count=0
        for pair in pairs:
            if pair['max_axis_voxels']>=side:continue
            region=pair_region(pair,source,side,resolution)
            if not region:continue
            regions[(pair['id'],side)]=region
            total_origin_count+=sum(box.count for box in region)
            for origin in origin_options(region):
                token=json.dumps([int(seed),int(scene),int(stream),side,pair['id'],origin]).encode()
                tie=int.from_bytes(hashlib.sha256(token).digest()[:8],'little')
                candidates.append((origin,pair,tie))
        chosen=[];used_pairs=set();used_buckets=set();seen=set()
        minimum=max(settings['minimum_center_distance'],side*settings['center_distance_side_fraction'])
        while len(chosen)<PILOT_SIDE_COUNTS[side]:
            best=None;best_key=None
            for origin,pair,tie in candidates:
                bucket=tuple(pair['bucket'])
                if pair['id'] in used_pairs or bucket in used_buckets or origin in seen:continue
                distance=min((sum((a-b)**2 for a,b in zip(origin,old[0])) for old in chosen),default=float('inf'))
                if distance<minimum**2:continue
                key=(distance,-pair['max_axis_voxels'],tie)
                if best_key is None or key>best_key:best_key=key;best=(origin,pair,tie)
            if best is None:break
            origin,pair,tie=best;chosen.append(best);used_pairs.add(pair['id']);used_buckets.add(tuple(pair['bucket']));seen.add(origin)
            w=Window(tuple(origin),side,witnesses=(tuple(pair['positive']),tuple(pair['negative'])))
            if not excludes(w,source,resolution):raise AssertionError('Analytic source subtraction disagrees')
            result.append((w,pair))
        audit[str(side)]={'selected':len(chosen),'unique_pairs':len(used_pairs),'unique_buckets':len(used_buckets),
                          'minimum_center_separation_required':minimum,'analytic_origin_options_considered':len(candidates),
                          'sum_pair_feasible_origin_counts_with_overlap':total_origin_count}
    return result,audit


def refine_pair(cache,pair,settings):
    p=tuple(pair['positive']);q=tuple(pair['negative']);queried=[]
    for _ in range(settings['refinement_steps_per_pair']):
        if max(abs(a-b) for a,b in zip(p,q))<=settings['refinement_target_max_axis']:break
        midpoint=tuple((a+b)//2 for a,b in zip(p,q))
        if midpoint in (p,q):break
        row=cache.query(np.asarray(midpoint,dtype=int),'refinement');queried.append(row['coordinate'])
        if row['solid']:break  # No assumption about free connectivity through an obstacle.
        if row['response']:p=midpoint
        else:q=midpoint
    return {'pair_id':pair['id'],'queried':queried,'final_positive':list(p),'final_negative':list(q)}


def guided_sample(source,cfg,scene,oracle,settings,*,stream,output):
    validate_window_config(cfg)
    if settings['accepted_per_bucket_per_side']!=1 or settings['accepted_per_pair_per_side']!=1:
        raise ValueError('This declared experiment requires one pair/bucket per side')
    output=Path(output);budget=Budget(cfg['sampling_seconds'],cfg['sampling_points'])
    proposals=sobol_points(cfg['seed'],scene,stream,settings['max_global_points'])
    coordinates=np.floor(proposals*cfg['resolution']).astype(int)
    global_index=0;refined=set();refinements=[];selection=[];audit={};termination='not_started'
    started=time.perf_counter();started_cpu=time.process_time()
    with (output/'points.jsonl').open('w') as sink:
        cache=ExactPointCache(source,cfg['resolution'],oracle,budget,settings['max_exact_points'],sink)
        try:
            while True:
                target=min(settings['initial_global_points'] if global_index==0 else global_index+settings['global_batch_points'],len(coordinates))
                with oracle.timed('witness_discovery_global'):
                    while global_index<target:
                        cache.query(coordinates[global_index],'global');global_index+=1
                with oracle.timed('window_construction'):
                    pairs=make_pairs(cache,settings)
                    selection,audit=select_diverse(pairs,source,cfg['resolution'],settings,cfg['seed'],scene,stream)
                if len(selection)==32:termination='complete';break
                remaining=settings['refinement_queries_per_round']
                while remaining>0:
                    existing_buckets={tuple(pair['bucket']) for w,pair in selection if w.size==32}
                    if len(existing_buckets)==PILOT_SIDE_COUNTS[32]:break
                    candidates=[pair for pair in pairs if pair['id'] not in refined and
                                tuple(pair['bucket']) not in existing_buckets and
                                pair['max_axis_voxels']>settings['refinement_target_max_axis'] and
                                pair['max_axis_voxels']<=settings['refinement_max_initial_axis']]
                    if not candidates:break
                    # Do not spend refinement queries near an already covered
                    # side-32 bucket. Global coverage is refreshed if needed.
                    pair=min(candidates,key=lambda x:(x['max_axis_voxels'],x['id']))
                    refined.add(pair['id']);before=len(cache.points)
                    with oracle.timed('witness_discovery_refinement'):
                        trace=refine_pair(cache,pair,settings)
                    refinements.append(trace);remaining-=max(1,len(cache.points)-before)
                    with oracle.timed('window_construction'):
                        pairs=make_pairs(cache,settings)
                        selection,audit=select_diverse(pairs,source,cfg['resolution'],settings,cfg['seed'],scene,stream)
                    if len(selection)==32:break
                if len(selection)==32:termination='complete';break
                if global_index>=len(coordinates):termination='declared_global_point_pool_exhausted';break
        except BudgetExhausted as exc:termination='budget:'+exc.kind
        except (Aborted,NumericalUnresolved) as exc:
            # Fatal reference failure is never normal search exhaustion.
            atomic_json(output/'fatal.json',{'type':type(exc).__name__,'reason':str(exc)})
            raise
    with oracle.timed('window_construction'):
        pairs=make_pairs(cache,settings)
        selection,audit=select_diverse(pairs,source,cfg['resolution'],settings,cfg['seed'],scene,stream)
        for pair in pairs:
            pair['feasible_origins']={str(side):sum(box.count for box in pair_region(pair,source,side,cfg['resolution']))
                                      for side in WINDOW_SIDES}
        atomic_json(output/'pairs.json',plain(pairs))
        atomic_json(output/'refinements.json',plain(refinements))
        rows=[{'proposal':i+1,'origin':list(w.origin),'side':w.size,'accepted':True,'outcome':'accepted',
               'pair_id':pair['id'],'bucket':pair['bucket'],'witnesses':plain(w.witnesses),
               'pair_max_axis_voxels':pair['max_axis_voxels'],'source':plain(source),
               'proof':'exact cached opposite free sampled centres'} for i,(w,pair) in enumerate(selection)]
        with (output/'proposals.jsonl').open('w') as sink:
            for row in rows:sink.write(json.dumps(row)+'\n')
    counts=Counter(row['label'] for row in cache.points.values())
    report={'termination':termination,'wall_seconds':time.perf_counter()-started,'process_seconds':time.process_time()-started_cpu,
            'cold_start':True,'exact_points':len(cache.points),'cache_hits':cache.cache_hits,'labels':dict(counts),
            'global_points':global_index,'refinement_points':sum(row['phase']=='refinement' for row in cache.points.values()),
            'opposite_pairs':len(pairs),'refinement_attempts':len(refinements),'by_side':audit,
            'complete_quota':len(selection)==32,'proposed_windows':len(selection),
            'placement':placement_summary([w for w,pair in selection],cfg['resolution'])}
    return [w for w,pair in selection],report
