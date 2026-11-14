"""Exact stored-bit boundary discovery and full-volume origin fallback.

Leaf adjacency and spatial indexes propose queries; only actual packed bits
label points. There is no geometry interpolation or oracle optimization here.
All reuse limits are global compute budgets, never windows-per-bucket caps.
"""
from collections import Counter,defaultdict
from itertools import product
import hashlib,math,time
import numpy as np
from scipy.spatial import cKDTree
from .origin_domains import voxel_pair_origins,sample_origins
from .windows import Window,excludes


def stratified_cells(mask,limit,rng):
    coordinates=np.argwhere(mask)
    if not len(coordinates):return []
    rng.shuffle(coordinates)
    bucket=coordinates//8;keys=bucket[:,0]*4096+bucket[:,1]*64+bucket[:,2]
    order=np.argsort(keys,kind='stable');coordinates=coordinates[order];keys=keys[order]
    starts=np.r_[0,np.flatnonzero(keys[1:]!=keys[:-1])+1];ends=np.r_[starts[1:],len(keys)]
    groups=list(zip(starts,ends));rng.shuffle(groups);result=[];depth=0
    while len(result)<min(limit,len(coordinates)):
        changed=False
        for begin,end in groups:
            if begin+depth<end:
                result.append(tuple(map(int,coordinates[begin+depth])));changed=True
                if len(result)==limit:return result
        if not changed:break
        depth+=1
    return result


class StoredGuidance:
    def __init__(self,index,source,settings,seed):
        self.index=index;self.source=np.asarray(source);self.settings=settings;self.n=index.shape[0]
        self.rng=np.random.default_rng(seed);self.cache={};self.pairs={};self.point_queries=0
        t=time.perf_counter();c=time.process_time();self._discover()
        counts=Counter(self.cache.values())
        self.report={'discovery_wall_seconds':time.perf_counter()-t,'discovery_process_seconds':time.process_time()-c,
            'exact_stored_point_queries':self.point_queries,'cached_points':len(self.cache),'free_lit_points':counts[1],
            'free_shadow_points':counts[2],'solid_points':counts[0],'pairs':len(self.pairs),
            'regions':len({p['region_bucket'] for p in self.pairs.values()}),'oracle_calls':0,
            'labels':'actual stored response AND solid bits','hard_pair_or_bucket_reuse_cap':None}
    def point(self,p):
        p=tuple(map(int,p))
        if p not in self.cache:self.cache[p]=self.index.point_class(p);self.point_queries+=1
        return self.cache[p]
    def leaf_point(self,zyx,flag):
        z,y,x=zyx;lower=(x*16,y*16,z*16);upper=tuple(min(v+16,self.n) for v in lower)
        _,points=self.index._leaf(lower,upper,flag,True,int(self.rng.integers(0,2**31)))
        if flag not in points:raise AssertionError('Hierarchy class has no actual stored witness')
        p=points[flag]
        if self.point(p)!=flag:raise AssertionError('Stored leaf witness disagreement')
        return p
    def pair(self,p,q):
        if len(self.pairs)>=self.settings['guide_max_pairs']:return
        p=tuple(p);q=tuple(q)
        if self.point(p)!=1 or self.point(q)!=2:raise AssertionError('A solid/non-opposite witness cannot define a pair')
        # Stored, exact midpoint queries; never interpolate a class across space.
        for _ in range(6):
            if max(abs(a-b) for a,b in zip(p,q))<=1:break
            midpoint=tuple((a+b)//2 for a,b in zip(p,q));label=self.point(midpoint)
            if label==0 or midpoint in (p,q):break
            if label==1:p=midpoint
            else:q=midpoint
        key=(p,q)
        if key not in self.pairs:
            delta=np.asarray(p)-q;mid=(np.asarray(p)+q+1)/2
            self.pairs[key]={'id':hashlib.sha256(np.asarray([p,q],dtype='<u2').tobytes()).hexdigest()[:24],
                'positive':p,'negative':q,'distance_voxels':float(np.linalg.norm(delta)),
                'max_axis_voxels':int(np.abs(delta).max()),'region_bucket':tuple(map(int,mid//128))}
    def _discover(self):
        leaf=self.index.levels[0];anchors=leaf==3
        # Between-leaf transitions matter: an exact boundary aligned to 16 need
        # not produce a single mixed leaf. Zero/solid leaves are not shadow.
        for axis in range(3):
            low=[slice(None)]*3;high=low.copy();low[axis]=slice(None,-1);high[axis]=slice(1,None)
            a=leaf[tuple(low)];b=leaf[tuple(high)];boundary=((a==1)&(b==2))|((a==2)&(b==1))
            anchors[tuple(low)]|=boundary;anchors[tuple(high)]|=boundary
        selected=stratified_cells(anchors,self.settings['guide_max_regions'],self.rng)
        positive=[]
        for cell in selected:
            flag=int(leaf[cell])
            if flag==3:p=self.leaf_point(cell,1);q=self.leaf_point(cell,2)
            else:
                neighbors=[]
                for axis in range(3):
                    for d in (-1,1):
                        other=list(cell);other[axis]+=d
                        if all(0<=v<n for v,n in zip(other,leaf.shape)) and int(leaf[tuple(other)])&(2 if flag==1 else 1):neighbors.append(tuple(other))
                if not neighbors:continue
                other=neighbors[int(self.rng.integers(len(neighbors)))]
                p=self.leaf_point(cell if flag==1 else other,1);q=self.leaf_point(other if flag==1 else cell,2)
            positive.append(p);self.pair(p,q)
            if len(self.pairs)>=self.settings['guide_max_pairs']:break
        # Additional local orientations improve origin coverage in tiny regions;
        # they do not consume a per-bucket window quota.
        for p in positive:
            if len(self.pairs)>=self.settings['guide_max_pairs']:break
            for axis in range(3):
                for d in (-1,1):
                    q=list(p);q[axis]+=d
                    if all(0<=v<self.n for v in q) and self.point(q)==2:self.pair(p,q)
        # Disconnected free regions separated by solid need not be adjacent.
        if len(self.pairs)<self.settings['guide_max_pairs']:
            limit=min(1024,self.settings['guide_max_regions'])
            pos=[self.leaf_point(c,1) for c in stratified_cells((leaf&1)!=0,limit,self.rng)]
            neg=[self.leaf_point(c,2) for c in stratified_cells((leaf&2)!=0,limit,self.rng)]
            if pos and neg:
                k=min(4,len(neg));_,neighbors=cKDTree(np.asarray(neg)).query(np.asarray(pos),k=k,p=np.inf)
                for p,ids in zip(pos,np.asarray(neighbors).reshape(len(pos),k)):
                    for i in sorted(set(map(int,ids))):self.pair(p,neg[i])
                    if len(self.pairs)>=self.settings['guide_max_pairs']:break

    def candidates(self,side,needed,seen,seed):
        start=time.perf_counter();cpu=time.process_time();rng=np.random.default_rng(seed);pairs=[];total_volume=0;boxes_count=0
        for pair in self.pairs.values():
            boxes=voxel_pair_origins(pair['positive'],pair['negative'],self.source,side,self.n);volume=sum(b.count for b in boxes)
            if volume:pairs.append((pair,boxes,volume));total_volume+=volume;boxes_count+=len(boxes)
        states=defaultdict(list);per_pair=max(32,math.ceil(4*max(1,needed)/max(1,len(pairs))))
        for pair,boxes,volume in pairs:
            origins=sample_origins(boxes,min(per_pair,self.settings['guide_max_proposals_per_side']),rng)
            states[pair['region_bucket']].append([pair,origins,0])
        keys=sorted(states);rng.shuffle(keys)
        for group in states.values():rng.shuffle(group)
        accepted=[];proposed=0;rejections=Counter();usage=Counter();buckets=Counter();budget=self.settings['guide_max_proposals_per_side']
        while keys and proposed<budget and len(accepted)<needed:
            remaining=[]
            for bucket in keys:
                group=states[bucket]
                while group and group[0][2]>=len(group[0][1]):group.pop(0)
                if not group:continue
                pair,origins,position=group.pop(0);origin=origins[position];group.append([pair,origins,position+1]);remaining.append(bucket);proposed+=1
                if origin in seen:rejections['duplicate_origin']+=1
                else:
                    seen.add(origin);window=Window(origin,side)
                    if not excludes(window,self.source,self.n):raise AssertionError('Analytic source exclusion failed')
                    if not all(all(o<=v<o+side for o,v in zip(origin,p)) for p in (pair['positive'],pair['negative'])):raise AssertionError('Witness containment failed')
                    if not self.index.query(origin,side).informative:raise AssertionError('Constructed stored-pair window failed exact eligibility')
                    accepted.append({'origin':list(origin),'side':side,'provenance':'guided_informative','proposal_id':proposed,
                        'witnesses':[pair['positive'],pair['negative']],'witness_pair_id':pair['id'],'witness_region':list(bucket),
                        'origin_law':'uniform integer volume within this pair legal domain; stratified pair/region schedule'})
                    usage[pair['id']]+=1;buckets[bucket]+=1
                if proposed>=budget or len(accepted)>=needed:break
            keys=remaining
        report={'used':True,'proposed':proposed,'accepted':len(accepted),'rejections':dict(rejections),
            'termination':'desired_pool' if len(accepted)>=needed else 'proposal_budget' if proposed>=budget else 'sampled_pair_domains_exhausted',
            'pairs_with_legal_origins':len(pairs),'legal_origin_boxes':boxes_count,'pair_origin_volume_sum_with_overlap':total_volume,
            'selected_pool_pairs':len(usage),'selected_pool_regions':len(buckets),'maximum_pair_reuse':max(usage.values(),default=0),
            'maximum_region_reuse':max(buckets.values(),default=0),'hard_pair_or_region_reuse_limit':None,
            'wall_seconds':time.perf_counter()-start,'process_seconds':time.process_time()-cpu,'oracle_calls':0}
        return accepted,report
