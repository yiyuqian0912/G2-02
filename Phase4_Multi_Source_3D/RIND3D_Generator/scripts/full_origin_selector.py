# Legacy four-side diagnostic; archived results/source snapshots are unchanged.
"""C: matched-witness sampling over the FULL legal integer origin volume.

B discovery and pair choices remain unchanged. This selector only changes the
origins, making the main B/C feature comparison paired on exact witnesses.
No oracle queries, interpolation, midpoint/corner injection or production hook.
"""
from __future__ import annotations
import hashlib,json,numbers
from itertools import combinations
import numpy as np
from rind3d.legacy.windows import Window,excludes
from rind3d.legacy.window_contract import WINDOW_SIDES,require_window_side
from experimental_boundary_guided import OriginBox,pair_id,pair_region


def canonical_boxes(boxes):
    result=sorted(boxes,key=lambda b:(b.lower,b.upper))
    for b in result:
        if len(b.lower)!=3 or len(b.upper)!=3 or any(isinstance(v,bool) or not isinstance(v,numbers.Integral) for v in (*b.lower,*b.upper)):
            raise ValueError('Integer 3D boxes required')
        if not b.count:raise ValueError('Empty origin box')
    for a,b in combinations(result,2):
        if all(max(x,y)<=min(u,v) for x,y,u,v in zip(a.lower,b.lower,a.upper,b.upper)):
            raise ValueError('Overlapping/duplicate origin boxes would distort the sampling measure')
    return result


def sample_integer_volume(boxes,count,rng):
    """Uniform distinct integer origins, O(number of boxes + requested count).

    A uniform rank in the concatenated integer volume weights each disjoint box
    by its exact integer cardinality. Floyd's algorithm avoids enumeration.
    Canonical ordering makes both weights and seeded output box-order invariant.
    """
    boxes=canonical_boxes(boxes)
    if isinstance(count,bool) or not isinstance(count,numbers.Integral) or count<0:raise ValueError('Invalid sample count')
    counts=[b.count for b in boxes];total=sum(counts);size=min(int(count),total)
    if not size:return []
    if total>np.iinfo(np.int64).max:raise ValueError('Integer-volume RNG range exceeded')
    selected=set();ranks=[]
    for end in range(total-size,total):
        value=int(rng.integers(0,end+1))
        if value in selected:value=end
        selected.add(value);ranks.append(value)
    rng.shuffle(ranks);ends=np.cumsum(counts,dtype=np.int64);result=[]
    for rank in ranks:
        index=int(np.searchsorted(ends,rank,side='right'));b=boxes[index]
        offset=rank-(int(ends[index-1]) if index else 0)
        dx,dy,dz=[v-u+1 for u,v in zip(b.lower,b.upper)]
        result.append((b.lower[0]+offset//(dy*dz),b.lower[1]+(offset//dz)%dy,b.lower[2]+offset%dz))
    if len(result)!=len(set(result)):raise AssertionError('Duplicate sampled origin')
    return result


def seeded_rng(seed,scene,stream,side,pair,source):
    text=json.dumps(['full-origin-v1',int(seed),int(scene),int(stream),side,pair,tuple(map(float,source))]).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(text).digest()[:16],'little'))


def pair_origin_candidates(pair,source,side,resolution,settings,seed,scene,stream):
    side=require_window_side(side);region=pair_region(pair,source,side,resolution)
    rng=seeded_rng(seed,scene,stream,side,pair['id'],source)
    return sample_integer_volume(region,settings['candidate_origins_per_pair'],rng),region


def assign_diverse(domains,pairs,minimum,node_budget):
    """Bounded deterministic MRV assignment; domains stay in seeded random order.

    Only existing pair/bucket limits and centre separation condition the sampled
    origins. There is no objective rewarding a face, corner or central origin.
    """
    assignment={};best={};nodes=0;exhausted=False
    def allowed(origin):
        return all(origin!=old and sum((a-b)**2 for a,b in zip(origin,old))>=minimum**2 for old in assignment.values())
    def visit():
        nonlocal nodes,best,exhausted
        if len(assignment)>len(best):best=dict(assignment)
        if len(assignment)==len(domains):return True
        if nodes>=node_budget:exhausted=True;return False
        remaining=[]
        for key,options in domains.items():
            if key in assignment:continue
            valid=[o for o in options if allowed(o)]
            spread=min((sum((a-b)**2 for a,b in zip(pairs[key]['midpoint_voxels'],pairs[k]['midpoint_voxels'])) for k in assignment),default=0)
            remaining.append((len(valid),-spread,key,valid))
        _,_,key,valid=min(remaining,key=lambda r:r[:3])
        if not valid:return False
        for origin in valid:
            if nodes>=node_budget:exhausted=True;return False
            nodes+=1;assignment[key]=origin
            if visit():return True
            del assignment[key]
        return False
    complete=visit()
    return dict(assignment) if complete else best,{'nodes':nodes,'complete':complete,'node_budget_exhausted':exhausted}


def select_full_volume(b_windows,pairs,source,cfg,b_settings,settings,scene,stream):
    lookup={p['id']:p for p in pairs};chosen=[];rows=[];audit={}
    for side in WINDOW_SIDES:
        group=[w for w in b_windows if w.size==side];keys=[pair_id(*w.witnesses) for w in group]
        if len(keys)!=len(set(keys)):raise ValueError('Same B pair repeated within a side')
        buckets=[tuple(lookup[k]['bucket']) for k in keys]
        if len(buckets)!=len(set(buckets)):raise ValueError('Same B spatial bucket repeated within a side')
        domains={};region_count=0;volume=0
        for key in sorted(keys):
            domains[key],region=pair_origin_candidates(lookup[key],source,side,cfg['resolution'],settings,cfg['seed'],scene,stream)
            region_count+=len(region);volume+=sum(b.count for b in region)
        minimum=max(b_settings['minimum_center_distance'],side*b_settings['center_distance_side_fraction'])
        assignment,state=assign_diverse(domains,lookup,minimum,settings['assignment_node_budget'])
        for key in sorted(assignment):
            pair=lookup[key];origin=assignment[key]
            w=Window(origin,side,witnesses=(tuple(pair['positive']),tuple(pair['negative'])))
            if not excludes(w,source,cfg['resolution']):raise AssertionError('Source exclusion failed')
            chosen.append(w);rows.append({'origin':list(origin),'side':side,'pair_id':key,'bucket':pair['bucket'],
                'witnesses':[pair['positive'],pair['negative']],'source':list(map(float,source)),
                'accepted':True,'proof':'same exact certified pair as B; volume sampled legal origin'})
        audit[str(side)]={'B_pair_count':len(keys),'selected':len(assignment),'unique_pairs':len(assignment),
            'spatial_buckets':len({tuple(lookup[k]['bucket']) for k in assignment}),
            'legal_region_boxes':region_count,'sum_pair_origin_counts_including_overlap':volume,
            'candidate_origins':sum(map(len,domains.values())),'minimum_center_distance':minimum,**state}
    if len({w.record() for w in chosen})!=len(chosen):raise AssertionError('Duplicate C windows')
    return chosen,rows,audit
