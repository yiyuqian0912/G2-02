"""Actual broad source discovery, staged screening and two full-window proofs."""
from collections import Counter,OrderedDict
import hashlib,time
import numpy as np
from .geometry import Budget,BudgetExhausted,keyed_rng
from .windows import sobol_points,excludes
from .search import separated,diverse_order
from .lite_acceptance import DRIVER_RADIUS,verify_clearance

class CandidateVerifier:
    def __init__(self,context,resolution=1024,cache_bricks=128):
        self.context=context;self.resolution=resolution;self.limit=cache_bricks;self.cache=OrderedDict();self.stats=Counter()
    def verify(self,source,window,observed,*,independent,budget):
        renderer=self.context.bind(source,independent=independent);covered=0;physical=0
        # Window-relative 64³ tiles avoid rejecting a source for differences
        # outside the observation. Every voxel, including all edges, is checked.
        for z in range(0,window.size,64):
            for y in range(0,window.size,64):
                for x in range(0,window.size,64):
                    budget.check();origin=tuple(np.asarray(window.origin)+[x,y,z]);side=min(64,window.size-max(x,y,z))
                    key=(renderer.source_id,independent,origin,side,self.resolution)
                    if key in self.cache:
                        packed=self.cache.pop(key);self.cache[key]=packed;self.stats['cache_hits']+=1
                        actual=np.unpackbits(np.frombuffer(packed,dtype=np.uint8),bitorder='little').reshape((side,)*3).astype(bool)
                    else:
                        actual,_,_=renderer.render(origin,side,self.resolution,budget=budget);physical+=1
                        self.cache[key]=np.packbits(actual,bitorder='little').tobytes()
                        while len(self.cache)>self.limit:self.cache.popitem(last=False)
                    if not np.array_equal(actual,observed[z:z+side,y:y+side,x:x+side]):return {'status':'mismatch','covered_voxels':covered,'physical_tiles':physical}
                    covered+=side**3
        if covered!=window.size**3:raise AssertionError('Candidate window coverage incomplete')
        return {'status':'grid_exact','covered_voxels':covered,'physical_tiles':physical,'proof':'native-exhaustive-window-tiles'}

def source_pool(context,truth,cfg,slot):
    start=time.perf_counter();positions=sobol_points(cfg['seed'],slot,1701,int(cfg['candidate_pool'])*2)
    valid=~context.contains(positions);points=[];proofs=[];rejections=Counter()
    visited=0
    for p,free in zip(positions,valid):
        visited+=1
        if not free:rejections['solid']+=1;continue
        if np.array_equal(p,truth):continue
        q=context.clearance(p,DRIVER_RADIUS)
        if q['status']!='GT':rejections['clearance_LE']+=1;continue
        verify_clearance(context,p,q);points.append(p);proofs.append(q);context.bind(p)
        if len(points)>=cfg['candidate_pool']:break
    return np.asarray(points,dtype='<f8').reshape(-1,3),proofs,{'seconds':time.perf_counter()-start,'proposed':visited,'sequence_points_available':len(positions),
            'retained':len(points),'rejections':dict(rejections),'prewarmed_source_models':len(points)}

def discover(context,truth,window,observed,mask,pool,cfg,slot,wid,truth_evidence,verifier):
    start=time.perf_counter();cpu=time.process_time();truth_evidence.verify_window(window,context,truth)
    retained=[np.array(truth,copy=True)];proofs=[{'status':'grid_exact','independent_status':'grid_exact','covered_voxels':window.size**3,'proof':'two-physical-brick-replays-v1'}]
    report={'status':'COMPLETE','actual_candidates':1,'all_grid_exact':True,'independent_validation_passed':True,
            'search_termination_reason':'proposal_pool_exhausted','proposed':0,'screen_survivors':0,'full_verifications':0,
            'independent_checks':0,'discarded_unrechecked':0,'screen_seconds':0.,'verification_seconds':0.,'independent_verification_seconds':0.,
            'unique_candidates_evaluated':0,'declared_search_seconds':cfg['window_search_seconds'],'max_candidates':8}
    budget=Budget(cfg['window_search_seconds'],cfg['window_search_points']);independent_budget=Budget(cfg['verification_seconds'],cfg['verification_points'])
    rng=keyed_rng(cfg['seed'],slot,1901+wid*7);requested=int(cfg['screen_probes']);count=requested//2
    raw=rng.integers(0,window.size,(requested*8,3))
    if len(window.witnesses)==2:
        witness=(np.asarray(window.witnesses)-np.asarray(window.origin))[:,::-1];raw=np.r_[witness,raw]
    raw=np.unique(raw,axis=0);v=observed[tuple(raw.T)];free=~mask[tuple(raw.T)]
    positive=raw[v&free];negative=raw[~v&free]
    if not len(positive) or not len(negative):raise ValueError('Exact stored informative witnesses unavailable')
    probes=np.r_[positive[:count],negative[:count]]
    labels=observed[tuple(probes.T)];points=(probes[:,::-1]+np.asarray(window.origin)+.5)/1024
    order=diverse_order(pool,cfg['seed'],slot,wid);used=set()
    try:
        for p in order:
            budget.check();key=p.tobytes()
            if key in used:continue
            used.add(key)
            if cfg['candidate_domain']=='world_outside_window' and not excludes(window,p,1024):continue
            if not all(separated(p,q,32/1024) for q in retained):continue
            report['proposed']+=1;renderer=context.bind(p);t=time.perf_counter();match=True
            for point,label in zip(points,labels):
                if renderer.response(p,point[None],budget=budget)[0]!=label:match=False;break
            report['screen_seconds']+=time.perf_counter()-t
            if not match:continue
            report['screen_survivors']+=1;t=time.perf_counter()
            first=verifier.verify(p,window,observed,independent=False,budget=budget)
            report['verification_seconds']+=time.perf_counter()-t;report['full_verifications']+=1
            if first['status']!='grid_exact':continue
            t=time.perf_counter()
            try:second=verifier.verify(p,window,observed,independent=True,budget=independent_budget)
            except BudgetExhausted:
                report['discarded_unrechecked']+=1;report['search_termination_reason']='independent_verification_budget';break
            report['independent_verification_seconds']+=time.perf_counter()-t;report['independent_checks']+=1
            if second['status']!='grid_exact':raise RuntimeError('Independent candidate physics disagreed')
            retained.append(p.copy());proofs.append(dict(first,independent_status='grid_exact'))
            if len(retained)==8:report['search_termination_reason']='max_candidates';break
    except BudgetExhausted as exc:report['search_termination_reason']='search_'+exc.kind+'_budget'
    report.update(actual_candidates=len(retained),unique_candidates_evaluated=len(used),elapsed_seconds=time.perf_counter()-start,
                  cpu_seconds=time.process_time()-cpu,min_pairwise_distance=min((float(np.linalg.norm(a-b)) for i,a in enumerate(retained) for b in retained[i+1:]),default=None))
    return np.asarray(retained),proofs,report
