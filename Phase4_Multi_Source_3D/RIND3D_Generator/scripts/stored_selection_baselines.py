"""Quota-only A/B/C ports for stored-field comparison, not publication code.

A uses the original full-window boolean predicate. B/C preserve their original
point discovery, corner/farthest or matched-domain assignment, but all methods
receive exactly the same stored labels. This deliberately favors B/C compared
with the historical geometry-oracle experiment; it is not a replay of its times.
"""
from contextlib import contextmanager
from collections import Counter
from pathlib import Path
from types import FunctionType
import json,time
import numpy as np
from rind3d.geometry import Budget,BudgetExhausted
from rind3d.information import InformationResult
from rind3d.stored_windows import uniform_candidates,keyed_seed
from rind3d.window_contract import WINDOW_SIDES,balanced_quotas
from rind3d.windows import Window,sobol_points
from rind3d.origin_domains import voxel_pair_origins,sample_origins
from rind3d.pack import atomic_json
from experimental_boundary_guided import ExactPointCache,make_pairs,refine_pair,select_diverse
from full_origin_selector import assign_diverse


def window_row(w,method):
    return {'origin':list(w.origin),'side':w.size,'witnesses':[list(p) for p in w.witnesses],
            'provenance':'uniform_accepted' if method=='A_scan' else 'guided_informative','method':method}


class FullScan:
    """Original production full local bool scan, with proof reuse, no hierarchy."""
    def __init__(self,r,m,progress):self.response=r;self.solid=m;self.shape=r.shape;self.proofs={};self.progress=progress;self.calls=0
    def query(self,origin,size,*,witnesses=False,**_):
        key=(tuple(origin),size)
        if witnesses and key in self.proofs:return self.proofs[key]
        self.calls+=1;y=self.response.read_window(*origin,size);m=self.solid.read_window(*origin,size)
        lit=y&~m;shadow=~y&~m;flags=int(lit.any())|int(shadow.any())<<1;p=q=None
        if flags==3:
            p=tuple(int(v+o) for v,o in zip(np.unravel_index(int(np.argmax(lit)),lit.shape)[::-1],origin))
            q=tuple(int(v+o) for v,o in zip(np.unravel_index(int(np.argmax(shadow)),shadow.shape)[::-1],origin))
            self.proofs[key]=InformationResult(flags,p,q)
            self.progress({'origin':list(origin),'side':size,'witnesses':[p,q],'provenance':'uniform_accepted','method':'A_scan'})
        return InformationResult(flags,p,q)


class StoredPointOracle:
    """Fixed-source receiver-label adapter only; cannot verify other sources."""
    def __init__(self,r,m,source):self.r=r;self.m=m;self.source=np.asarray(source);self.counters=Counter()
    def contains(self,points,budget=None):
        p=np.asarray(points).reshape(-1,3);coordinates=p*self.r.shape[0]-.5
        if not np.array_equal(coordinates,np.rint(coordinates)):raise ValueError('The stored adapter only accepts sampled centers')
        if budget:budget.check(len(p))
        self.counters['stored_membership_queries']+=len(p)
        return np.array([self.m.read_point(*map(int,q)) for q in coordinates],dtype=bool)
    def response(self,source,points,budget=None):
        if not np.array_equal(np.asarray(source),self.source):raise ValueError('This adapter cannot rerender a candidate source')
        p=np.asarray(points).reshape(-1,3);coordinates=p*self.r.shape[0]-.5
        if not np.array_equal(coordinates,np.rint(coordinates)):raise ValueError('Sampled centers required')
        if budget:budget.check(len(p))
        self.counters['stored_response_queries']+=len(p)
        return np.array([self.r.read_point(*map(int,q)) for q in coordinates],dtype=bool)
    @contextmanager
    def timed(self,stage):yield


def scan(r,m,source,count,seed,scene,maximum,output):
    saved=[];path=Path(output)/'partial-selected.jsonl'
    with path.open('w') as f:
        def progress(row):saved.append(row);f.write(json.dumps(row)+'\n');f.flush()
        adapter=FullScan(r,m,progress);rows=[];reports={}
        for side,target in balanced_quotas(count).items():
            pool,meta=uniform_candidates(adapter,source,side,target,maximum,keyed_seed(seed,scene,302,side))
            for row in pool:
                proof=adapter.query(tuple(row['origin']),side,witnesses=True);row['witnesses']=[proof.lit,proof.shadow];row['method']='A_scan'
            rows.extend(pool);reports[str(side)]=meta
    return rows,{'by_side':reports,'full_window_scans':adapter.calls,'oracle_calls':0,'baseline':'original full local bool predicate with common three-side Sobol proposals'}


def guided(r,m,source,count,seed,scene,settings,method,output):
    output=Path(output);quotas=balanced_quotas(count)
    if max(quotas.values())>512:
        return [],{'termination':'structural_128_bucket_capacity','upper_bound_per_side':512,'quotas':quotas,'oracle_calls':0}
    # The old selection kernel and math are unchanged; only declared sides and
    # quota globals are supplied to its copied function object.
    globals_=dict(select_diverse.__globals__,WINDOW_SIDES=WINDOW_SIDES,PILOT_SIDE_COUNTS=quotas)
    choose=FunctionType(select_diverse.__code__,globals_,select_diverse.__name__,select_diverse.__defaults__)
    adapter=StoredPointOracle(r,m,source);cache=ExactPointCache(source,r.shape[0],adapter,Budget(1e6,50_000_000),settings['max_exact_points'])
    coordinates=np.floor(sobol_points(seed,scene,302,settings['max_global_points'])*r.shape[0]).astype(int)
    global_index=0;refined=set();selection=[];pairs=[];termination='global_point_pool';started=time.perf_counter()
    def checkpoint():
        values=[window_row(w,'B_stored') for w,p in selection]
        (output/'partial-selected.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in values))
        atomic_json(output/'progress.json',{'phase':'B_support_discovery','global_points':global_index,'exact_points':len(cache.points),
                    'point_calls':dict(adapter.counters),'accepted_B_support':len(values),'elapsed':time.perf_counter()-started})
    try:
        while global_index<len(coordinates):
            target=min(settings['initial_global_points'] if global_index==0 else global_index+settings['global_batch_points'],len(coordinates))
            while global_index<target:cache.query(coordinates[global_index],'global');global_index+=1
            pairs=make_pairs(cache,settings);selection,audit=choose(pairs,source,r.shape[0],settings,seed,scene,302);checkpoint()
            if len(selection)==count:termination='quota_complete';break
            remaining=settings['refinement_queries_per_round']
            while remaining>0:
                used={tuple(p['bucket']) for w,p in selection if w.size==64}
                if len(used)>=quotas[64]:break
                choices=[p for p in pairs if p['id'] not in refined and tuple(p['bucket']) not in used and
                         settings['refinement_target_max_axis']<p['max_axis_voxels']<=settings['refinement_max_initial_axis']]
                if not choices:break
                pair=min(choices,key=lambda p:(p['max_axis_voxels'],p['id']));refined.add(pair['id']);before=len(cache.points)
                refine_pair(cache,pair,settings);remaining-=max(1,len(cache.points)-before)
                pairs=make_pairs(cache,settings);selection,audit=choose(pairs,source,r.shape[0],settings,seed,scene,302);checkpoint()
                if len(selection)==count:break
            if len(selection)==count:termination='quota_complete';break
    except BudgetExhausted as exc:termination='stored_point_budget:'+exc.kind
    b_time=time.perf_counter()-started
    if method=='B_stored':
        rows=[window_row(w,method) for w,p in selection]
        return rows,{'termination':termination,'stored_point_calls':dict(adapter.counters),'exact_points':len(cache.points),'B_support_wall':b_time,'oracle_calls':0}
    # C pays fresh B discovery/support, then changes only complete-domain origins.
    rows=[];assignments={};rng=np.random.default_rng(keyed_seed(seed,scene,35002,0))
    (output/'partial-selected.jsonl').write_text('')  # B support is not C output.
    for side,target in quotas.items():
        chosen=[p for w,p in selection if w.size==side];lookup={p['id']:p for p in chosen};domains={}
        for p in sorted(chosen,key=lambda p:p['id']):domains[p['id']]=sample_origins(voxel_pair_origins(p['positive'],p['negative'],source,side,r.shape[0]),32,rng)
        assignment,state=assign_diverse(domains,lookup,max(64,side/2),100000);assignments[str(side)]=state
        for key,origin in sorted(assignment.items()):
            p=lookup[key];w=Window(origin,side,witnesses=(tuple(p['positive']),tuple(p['negative'])));rows.append(window_row(w,method))
        (output/'partial-selected.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        atomic_json(output/'progress.json',{'phase':'C_origin_assignment','stored_point_calls':dict(adapter.counters),'assignment':assignments,'B_support_wall':b_time,'C_rows':len(rows)})
    return rows,{'termination':'quota_complete' if len(rows)==count else 'incomplete_B_support_or_C_assignment','B_support_wall':b_time,
                'stored_point_calls':dict(adapter.counters),'exact_points':len(cache.points),'assignment':assignments,'oracle_calls':0}
