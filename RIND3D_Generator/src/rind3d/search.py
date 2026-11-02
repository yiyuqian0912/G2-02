"""Broad proposals, sound probe rejection, and exhaustive grid verification."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import hashlib,time
import numpy as np
from .geometry import Budget,BudgetExhausted,Aborted,NumericalUnresolved,centres,keyed_rng
from .windows import sobol_points,excludes
from .window_contract import require_window_side

def separated(a,b,distance):
    delta=np.asarray(a)-b;v=float(delta@delta);limit=distance*distance
    margin=32*np.finfo(float).eps*max(v,limit,1e-300)
    if v>limit+margin:return True
    if v<limit-margin:return False
    return sum((Fraction(float(x))-Fraction(float(y)))**2 for x,y in zip(a,b))>=Fraction(float(distance))**2

def farthest(points,target,min_distance,*,keep_first=True):
    p=np.asarray(points,dtype=np.float64).reshape(-1,3)
    if not len(p):return []
    selected=[0];remaining=set(range(1,len(p)))
    distances=np.linalg.norm(p-p[0],axis=1)
    while remaining and len(selected)<target:
        i=max(remaining,key=lambda j:(distances[j],-j));remaining.remove(i)
        if not all(separated(p[i],p[j],min_distance) for j in selected):continue
        selected.append(i);distances=np.minimum(distances,np.linalg.norm(p-p[i],axis=1))
    return selected

class CandidateBank:
    def __init__(self):self.coordinates=[];self.by_value={}
    def add(self,point):
        p=np.asarray(point,dtype='<f8')
        if p.shape!=(3,) or not np.isfinite(p).all() or np.any(p<=0) or np.any(p>=1):raise ValueError('Invalid source coordinates')
        key=p.tobytes()
        if key not in self.by_value:self.by_value[key]=len(self.coordinates);self.coordinates.append(p.copy())
        return self.by_value[key]
    @property
    def array(self):return np.array(self.coordinates,dtype='<f8').reshape(-1,3)
    @property
    def index_dtype(self):return np.dtype('<u2' if len(self.coordinates)<=65535 else '<u4')

def free_pool(oracle,cfg,scene_id,budget):
    proposals=sobol_points(cfg['seed'],scene_id,401,int(cfg['candidate_pool']))
    result=[]
    for start in range(0,len(proposals),64):
        budget.check();part=proposals[start:start+64];valid=~oracle.contains(part,budget=budget)
        result.extend(part[valid])
    return np.asarray(result,dtype=np.float64).reshape(-1,3)

def probe_set(oracle,truth,w,cfg,scene_id,window_id,stream,budget,*,count=None):
    requested=int(cfg['screen_probes'] if count is None else count)
    if requested<2:raise ValueError('A balanced screen requires at least two probes')
    rng=keyed_rng(cfg['seed'],scene_id,stream+window_id*11)
    raw=np.unique(rng.integers(0,w.size,(requested*4,3))+w.origin,axis=0)
    if w.witnesses:raw=np.unique(np.vstack([raw,w.witnesses]),axis=0)
    points=(raw+.5)/cfg['resolution'];mask=oracle.contains(points,budget=budget)
    labels=oracle.response(truth,points,budget=budget)
    positive=np.flatnonzero(labels&~mask);negative=np.flatnonzero(~labels&~mask)
    if not len(positive) or not len(negative):raise NumericalUnresolved('Accepted informative window has no recoverable free positive/negative witnesses')
    count=requested//2
    chosen=np.r_[positive[:count],negative[:count]]
    # Opposite-label pairs nearest among these probes supply an explicit
    # boundary-oriented screen; no claim of exhaustive boundary coverage.
    dist=((points[positive][:,None]-points[negative][None])**2).sum(axis=-1)
    a,b=np.unravel_index(np.argmin(dist),dist.shape)
    chosen=np.unique(np.r_[chosen,positive[a],negative[b]])
    return points[chosen],labels[chosen]

def screen(oracle,candidates,points,labels,budget):
    keep=np.arange(len(candidates))
    for point,value in zip(points,labels):
        if not len(keep):break
        predictions=oracle.response(candidates[keep],point[None],budget=budget)
        keep=keep[predictions==value]
    return candidates[keep]

def verify_grid_exact(oracle,truth,candidate,w,resolution,budget,*,observed=None,leaf_side=8):
    """All voxel centres are compared or covered by valid uniform certificates.
    The source identity is an analytic tautology in a feasibility pilot, not a
    probe-only equivalence claim. Stored truth disables this shortcut.
    """
    if observed is None and np.array_equal(truth,candidate):
        return {'status':'grid_exact','covered_voxels':w.size**3,'proof':'identical-source-identity','point_checks':0}
    pending=[(tuple(w.origin),(w.size,)*3)];covered=0;point_checks=0
    while pending:
        budget.check();origin,lengths=pending.pop();lx,ly,lz=lengths
        lo=(np.asarray(origin)+.5)/resolution;hi=(np.asarray(origin)+lengths-.5)/resolution
        candidate_y,_=oracle.certificate(candidate,lo,hi,budget=budget)
        if observed is None:truth_y,_=oracle.certificate(truth,lo,hi,budget=budget)
        else:
            x,y,z=np.asarray(origin)-w.origin
            block=observed[z:z+lz,y:y+ly,x:x+lx]
            truth_y=int(block.flat[0]) if np.all(block==block.flat[0]) else -1
        if candidate_y>=0 and truth_y>=0:
            if candidate_y!=truth_y:return {'status':'mismatch','covered_voxels':covered,'proof':'uniform-certificate-mismatch'}
            covered+=lx*ly*lz;continue
        if max(lengths)<=leaf_side:
            points=centres(origin,lengths,resolution)
            actual=oracle.response(candidate,points,budget=budget)
            expected=oracle.response(truth,points,budget=budget) if observed is None else block.reshape(-1)
            point_checks+=len(points)
            if not np.array_equal(actual,expected):return {'status':'mismatch','covered_voxels':covered,'point_checks':point_checks}
            covered+=len(points)
        else:
            cuts=[[(0,n)] if n<=leaf_side else [(0,n//2),(n//2,n-n//2)] for n in lengths]
            for dz,nz in cuts[2]:
                for dy,ny in cuts[1]:
                    for dx,nx in cuts[0]:pending.append(((origin[0]+dx,origin[1]+dy,origin[2]+dz),(nx,ny,nz)))
    if covered!=w.size**3:raise AssertionError('Incomplete grid coverage')
    return {'status':'grid_exact','covered_voxels':covered,'point_checks':point_checks,'proof':'adaptive-exhaustive-centres-with-uniform-certificates'}

def diverse_order(points,seed,scene_id,window_id):
    """Round-robin 4^3 world cells; no local refinement around accepted sources."""
    points=np.asarray(points,dtype=np.float64).reshape(-1,3)
    unique=[];seen=set()
    for p in points:
        key=p.astype('<f8').tobytes()
        if key in seen:continue
        seen.add(key)
        if np.isfinite(p).all() and np.all(p>0) and np.all(p<1):unique.append(p)
    cells={}
    for p in unique:
        cell=tuple(np.minimum(3,(p*4).astype(int)))
        cells.setdefault(cell,[]).append(p)
    keys=sorted(cells);keyed_rng(seed,scene_id,17001+window_id).shuffle(keys)
    result=[]
    for depth in range(max((len(v) for v in cells.values()),default=0)):
        for key in keys:
            if depth<len(cells[key]):result.append(cells[key][depth])
    return np.asarray(result,dtype=np.float64).reshape(-1,3)


def distant_mask(points,accepted,distance):
    if not len(points):return np.empty(0,dtype=bool)
    if not len(accepted):return np.ones(len(points),dtype=bool)
    p=np.asarray(points);a=np.asarray(accepted);v=((p[:,None,:]-a[None,:,:])**2).sum(axis=2)
    limit=distance**2;margin=32*np.finfo(float).eps*np.maximum(v,limit)
    ok=np.all(v>=limit-margin,axis=1)
    for i,j in zip(*np.where((np.abs(v-limit)<=margin)&ok[:,None])):
        if not separated(p[i],a[j],distance):ok[i]=False
    return ok


def search_candidates(oracle,truth,w,pool,cfg,scene_id,window_id,*,observed=None):
    """Return 1..max_candidates, all twice verified, or an explicit failure.

    Our search exhaustion is a normal stop. A canonical ABORTED/UNRESOLVED
    decision is still fatal. The true source is verified twice before optional
    search, and no unfinished or unrechecked proposal enters the returned set.
    """
    require_window_side(w.size)
    if not cfg.get('include_true_source',True):raise ValueError('Contract v2 requires the true source')
    maximum=int(cfg['max_candidates'])
    if not 1<=maximum<=64:raise ValueError('max_candidates must be 1..64')
    start=time.perf_counter();retained=[];evidence=[];first=[]
    report={'status':'ABORTED','max_candidates':maximum,'proposed':0,'unique_proposals':0,
            'full_verifications':0,'full_verification_attempts':0,'screen_survivors':0,
            'verification_seconds':0.,'independent_verification_seconds':0.,'search_seconds':0.,
            'physical_point_calls':0,'independent_point_calls':0,'skipped_near_accepted':0,
            'skipped_domain':0,'discarded_unrechecked':0,'continuous_certified':0,'sample_only':0,
            'search_termination_reason':'not_started','recheck_termination_reason':'not_started',
            'true_source_included':True,'independent_validation_passed':False}
    oracle.evaluated_sources.clear()
    truth=np.asarray(truth,dtype=np.float64)
    if truth.shape!=(3,) or not np.isfinite(truth).all() or np.any(truth<=0) or np.any(truth>=1):raise ValueError('Invalid true source')
    if not excludes(w,truth,cfg['resolution']):raise ValueError('True source is inside the observation')
    seconds=cfg['verification_seconds'];points_limit=cfg['verification_points']
    initial=Budget(seconds,points_limit);independent=Budget(seconds,points_limit)
    search_budget=None;extra_budget=None
    try:
        t=time.perf_counter()
        try:
            if oracle.contains(truth[None],budget=initial)[0]:raise NumericalUnresolved('True source is not certified free')
            result=verify_grid_exact(oracle,truth,truth,w,cfg['resolution'],initial,observed=observed,leaf_side=cfg['leaf_side'])
            if result['status']!='grid_exact':raise NumericalUnresolved('Stored response disagrees with true source')
        finally:report['verification_seconds']+=time.perf_counter()-t
        independent=Budget(seconds,points_limit);t=time.perf_counter()
        try:
            if oracle.contains(truth[None],budget=independent)[0]:raise NumericalUnresolved('Independent source membership disagreed')
            again=verify_grid_exact(oracle,truth,truth,w,cfg['resolution'],independent,observed=observed,leaf_side=cfg['leaf_side'])
            if again['status']!='grid_exact':raise NumericalUnresolved('Independent true-source comparison disagreed')
        finally:report['independent_verification_seconds']+=time.perf_counter()-t
        retained=[truth.copy()];evidence=[dict(result,independent_status='grid_exact')];first=[truth.copy()]
        oracle.evaluated_sources.add(truth.astype('<f8').tobytes())
        search_budget=Budget(cfg['window_search_seconds'],cfg['window_search_points']);t_search=time.perf_counter()
        try:
            if maximum==1:report['search_termination_reason']='max_candidates'
            else:
                probes,labels=probe_set(oracle,truth,w,cfg,scene_id,window_id,501,search_budget)
                other,other_y=probe_set(oracle,truth,w,cfg,scene_id,window_id,10001,search_budget,count=cfg['independent_probes'])
                n=int(cfg['global_proposals'])
                extra=sobol_points(cfg['seed'],scene_id,6001+window_id,n)
                proposals=diverse_order(np.vstack([np.asarray(pool).reshape(-1,3),extra]),cfg['seed'],scene_id,window_id)
                report['proposed']=len(pool)+n;report['unique_proposals']=len(proposals)
                batch_size=int(cfg['search_batch_size'])
                report['search_termination_reason']='proposal_pool_exhausted'
                for offset in range(0,len(proposals),batch_size):
                    search_budget.check();batch=proposals[offset:offset+batch_size]
                    if cfg['candidate_domain']=='world_outside_window':
                        valid=np.array([excludes(w,p,cfg['resolution']) for p in batch])
                        report['skipped_domain']+=int((~valid).sum());batch=batch[valid]
                    valid=distant_mask(batch,first,cfg['min_source_separation'])
                    report['skipped_near_accepted']+=int((~valid).sum());batch=batch[valid]
                    if not len(batch):continue
                    batch=batch[~oracle.contains(batch,budget=search_budget)]
                    if not len(batch):continue
                    batch=screen(oracle,batch,probes,labels,search_budget)
                    batch=screen(oracle,batch,other,other_y,search_budget)
                    report['screen_survivors']+=len(batch)
                    for p in batch:
                        search_budget.check()
                        if not distant_mask(p[None],first,cfg['min_source_separation'])[0]:
                            report['skipped_near_accepted']+=1;continue
                        t=time.perf_counter();report['full_verification_attempts']+=1
                        try:match=verify_grid_exact(oracle,truth,p,w,cfg['resolution'],search_budget,observed=observed,leaf_side=cfg['leaf_side'])
                        finally:report['verification_seconds']+=time.perf_counter()-t
                        report['full_verifications']+=1
                        if match['status']=='grid_exact':first.append(p.copy())
                        if len(first)==maximum:break
                    if len(first)==maximum:report['search_termination_reason']='max_candidates';break
        except BudgetExhausted as exc:report['search_termination_reason']='search_'+exc.kind+'_budget'
        finally:report['search_seconds']=time.perf_counter()-t_search
        extra_budget=Budget(seconds,points_limit);report['recheck_termination_reason']='complete';t=time.perf_counter()
        try:
            for p in first[1:]:
                extra_budget.check()
                if oracle.contains(p[None],budget=extra_budget)[0]:raise NumericalUnresolved('Independent candidate membership disagreed')
                match=verify_grid_exact(oracle,truth,p,w,cfg['resolution'],extra_budget,observed=observed,leaf_side=cfg['leaf_side'])
                if match['status']!='grid_exact':raise NumericalUnresolved('Independent final verification disagreed')
                retained.append(p.copy());evidence.append(dict(match,independent_status='grid_exact'))
        except BudgetExhausted as exc:report['recheck_termination_reason']='verification_'+exc.kind+'_budget'
        finally:report['independent_verification_seconds']+=time.perf_counter()-t
        report['status']='COMPLETE';report['reason']=''
    except BudgetExhausted as exc:
        report['status']='ABORTED';report['reason']='Mandatory true-source verification '+exc.kind+' budget exhausted'
    except Aborted as exc:report['status']='ABORTED';report['reason']=str(exc)
    except NumericalUnresolved as exc:report['status']='UNRESOLVED';report['reason']=str(exc)
    report.update(actual_candidates=len(retained),first_pass_candidates=len(first),
        discarded_unrechecked=max(0,len(first)-len(retained)),
        independent_validation_passed=bool(retained) and all(e['independent_status']=='grid_exact' for e in evidence),
        all_grid_exact=bool(retained) and all(e['status']=='grid_exact' and e['covered_voxels']==w.size**3 for e in evidence),
        reached_max_candidates=len(retained)==maximum,
        physical_point_calls=initial.points+(search_budget.points if search_budget else 0),
        independent_point_calls=independent.points+(extra_budget.points if extra_budget else 0),
        unique_physical_candidate_sources=len(oracle.evaluated_sources),
        termination_reason=(report['recheck_termination_reason'] if report['recheck_termination_reason'] not in ('complete','not_started') else report['search_termination_reason']),
        elapsed_seconds=time.perf_counter()-start,
        min_pairwise_distance=min((float(np.linalg.norm(p-q)) for i,p in enumerate(retained) for q in retained[i+1:]),default=None))
    return np.asarray(retained,dtype=np.float64).reshape(-1,3),evidence,report
