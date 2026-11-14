"""D: exact stored-field candidate discovery, separate final selection/control.

Uniform-first discovery and independently seeded controls use exact index
eligibility, not probes. Soft selection is explicitly distribution-changing.
Guidance is loaded only after a measured/budgeted uniform shortfall.
"""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,math,time
import numpy as np
from scipy.stats import qmc
from .window_contract import WINDOW_SIDES,balanced_quotas
from .windows import Window,excludes,placement_summary
from .window_diversity import soft_select
from .pack import atomic_json

DEFAULT_SELECTION={
    'valid_pool_multiplier':4,'uniform_max_proposals_per_side':65536,'uniform_pilot_proposals':4096,
    'guided_fallback':True,'guide_max_regions':2048,'guide_max_pairs':2048,'guide_max_proposals_per_side':131072,
    'coverage_cell_side':128,'overlap_weight':0.05,'swap_attempts_per_selected':2,
    'greedy_evaluations_per_selected':32,
    'selection_mode':'soft','control_per_side':128,'control_max_proposals_per_side':65536,
    'record_proposal_trace':False,
}


def selection_settings(cfg=None):
    settings={**DEFAULT_SELECTION,**(cfg or {})}
    if set(settings)!=set(DEFAULT_SELECTION):raise ValueError('Unknown window-selection setting')
    for name in ('valid_pool_multiplier','uniform_max_proposals_per_side','uniform_pilot_proposals','guide_max_regions','guide_max_pairs',
                 'guide_max_proposals_per_side','coverage_cell_side','control_max_proposals_per_side','greedy_evaluations_per_selected'):
        if isinstance(settings[name],bool) or not isinstance(settings[name],int) or settings[name]<1:raise ValueError('Invalid '+name)
    for name in ('swap_attempts_per_selected','control_per_side'):
        if isinstance(settings[name],bool) or not isinstance(settings[name],int) or settings[name]<0:raise ValueError('Invalid '+name)
    if settings['selection_mode'] not in ('uniform','soft'):raise ValueError('Unknown final selection mode')
    if not isinstance(settings['guided_fallback'],bool) or not isinstance(settings['record_proposal_trace'],bool) or not np.isfinite(settings['overlap_weight']) or settings['overlap_weight']<0:raise ValueError('Invalid soft/guided setting')
    return settings


def keyed_seed(seed,scene,stream,side):
    return int(np.random.SeedSequence([int(seed),int(scene),int(stream),int(side)]).generate_state(1)[0])


@dataclass
class SelectionResult:
    windows:list
    provenance:list
    controls:list
    report:dict


def uniform_candidates(index,source,side,target,maximum,seed,*,pilot=None,minimum_required=None,exclude=(),sink=None):
    if target<=0:return [],{'proposed':0,'accepted':0,'rejections':{},'termination':'zero_target'}
    n=index.shape[0];engine=qmc.Sobol(3,scramble=True,seed=seed);seen=set(exclude);accepted=[];rejections=Counter();proposed=0
    start=time.perf_counter();cpu=time.process_time();termination='proposal_budget';stop=False;stop_at_minimum=False
    while proposed<maximum and len(accepted)<target:
        # A fixed Sobol stream is independent of batch size and stopping time.
        batch=engine.random_base2(10) if proposed==0 else engine.random(min(1024,maximum-proposed))
        for position in batch:
            if proposed>=maximum or len(accepted)>=target:break
            origin=tuple(np.floor(position*(n-side+1)).astype(int).tolist());proposed+=1
            reason=None;flags=None
            if origin in seen:reason='duplicate_origin'
            else:
                seen.add(origin);window=Window(origin,side)
                if not excludes(window,source,n):reason='contains_true_source'
                else:
                    result=index.query(origin,side);flags=result.flags
                    if not result.informative:reason='no_free_voxels' if flags==0 else 'no_lit_free' if not flags&1 else 'no_shadow_free'
            if reason:rejections[reason]+=1
            else:accepted.append({'origin':list(origin),'side':side,'provenance':'uniform_accepted','proposal_id':proposed,'witnesses':None})
            if sink:sink.write(json.dumps({'origin':origin,'side':side,'proposal_id':proposed,'accepted':reason is None,'reason':reason,'class_flags':flags})+'\n')
            if pilot and proposed==min(pilot,maximum):
                rate=len(accepted)/proposed;needed=max(0,(target if minimum_required is None else minimum_required)-len(accepted))
                if rate*(maximum-proposed)<needed:
                    termination='projected_uniform_shortfall';stop=True;break
                if minimum_required is not None and rate*(maximum-proposed)<target-len(accepted):stop_at_minimum=True
            if stop_at_minimum and len(accepted)>=minimum_required:
                termination='hard_quota_met_soft_pool_shortfall';stop=True;break
        if stop:break
    if len(accepted)==target:termination='pool_target'
    return accepted,{'proposed':proposed,'accepted':len(accepted),'acceptance_rate':len(accepted)/proposed if proposed else None,
        'rejections':dict(rejections),'termination':termination,'wall_seconds':time.perf_counter()-start,'process_seconds':time.process_time()-cpu,
        'stream_seed':seed,'uniform_accepted_distribution':'exact informative condition; scrambled Sobol; no diversity selection'}


def select_stored_windows(index,source,count=4096,*,seed=0,scene=0,settings=None,output=None):
    settings=selection_settings(settings);source=np.asarray(source,dtype=np.float64)
    if source.shape!=(3,) or not np.isfinite(source).all() or np.any(source<=0) or np.any(source>=1):raise ValueError('A fixed interior true source is required')
    if len(set(index.shape))!=1 or index.shape[0]<max(WINDOW_SIDES):raise ValueError('Selection needs a cubic stored field fitting side 256')
    n=index.shape[0];quotas=balanced_quotas(count);start=time.perf_counter();cpu=time.process_time();counter_before=Counter(index.counters);pools={};rows=[];controls=[];by_side={};guide=None
    root=Path(output) if output is not None else None
    if root:root.mkdir(parents=True,exist_ok=True)
    if index.global_flags!=3:
        report={'complete':False,'termination':'global_missing_free_response_class','global_flags':index.global_flags,
            'quotas':quotas,'side_counts':dict.fromkeys(WINDOW_SIDES,0),'accepted':0,'duplicate_selected_windows':0,'provenance_counts':{},
            'oracle_calls':0,'wall_seconds':time.perf_counter()-start,'process_seconds':time.process_time()-cpu,'settings':settings,'is_dataset':False,'production_ready':False}
        if root:
            atomic_json(root/'selection.json',report)
            for name in ('selected.jsonl','uniform-control.jsonl'):(root/name).write_text('')
        return SelectionResult([],[],[],report)
    for side,target in quotas.items():
        desired=target*settings['valid_pool_multiplier'] if settings['selection_mode']=='soft' else target
        sink=(root/f'proposals-{side}.jsonl').open('w') if root and settings['record_proposal_trace'] else None
        try:
            pool,uniform=uniform_candidates(index,source,side,desired,settings['uniform_max_proposals_per_side'],keyed_seed(seed,scene,302,side),
                pilot=settings['uniform_pilot_proposals'] if settings['guided_fallback'] else None,minimum_required=target,sink=sink)
        finally:
            if sink:sink.close()
        guided={'used':False};seen={tuple(row['origin']) for row in pool}
        # Oversampling is a preference, never a reason to introduce guidance or
        # reject an already sufficient uniform pool.
        if len(pool)<target and settings['guided_fallback']:
            from .stored_guidance import StoredGuidance
            if guide is None:guide=StoredGuidance(index,source,settings,keyed_seed(seed,scene,7001,0))
            extra,guided=guide.candidates(side,desired-len(pool),seen,keyed_seed(seed,scene,7002,side));pool.extend(extra)
        pools[side]=pool;by_side[str(side)]={'target':target,'desired_pool':desired,'actual_pool':len(pool),'uniform':uniform,'guided':guided}
        if root:atomic_json(root/'progress.json',{'phase':'candidate_discovery','by_side':by_side,'elapsed_seconds':time.perf_counter()-start})
    selection={}
    for side,target in quotas.items():
        pool=pools[side]
        if settings['selection_mode']=='uniform':
            chosen=list(range(min(target,len(pool))));scores={i:None for i in chosen};selection[str(side)]={'mode':'uniform','soft_diversity_applied':False}
        else:
            chosen,scores,selection[str(side)]=soft_select([r['origin'] for r in pool],side,target,settings,keyed_seed(seed,scene,8001,side))
        for i in chosen:
            row=dict(pool[i]);origin=tuple(row['origin'])
            if row['witnesses'] is None:
                proof=index.query(origin,side,witnesses=True,witness_seed=keyed_seed(seed,scene,8002,side))
                if not proof.informative:raise AssertionError('Index eligibility changed')
                row['witnesses']=[proof.lit,proof.shadow];row['witness_policy']='index_hashed_tree_proof'
            else:row['witness_policy']='guided_construction_pair'
            row.update(selection_score=scores[i],pool_index=i,window_id=len(rows),selection_mode=settings['selection_mode'])
            for key in ('witness_pair_id','witness_region','origin_law'):row.setdefault(key,None)
            rows.append(row)
        if root:
            (root/'partial-selected.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            atomic_json(root/'progress.json',{'phase':'final_selection','completed_side':side,'finalized_windows':len(rows),'elapsed_seconds':time.perf_counter()-start})
    windows=[Window(tuple(r['origin']),r['side'],witnesses=tuple(map(tuple,r['witnesses']))) for r in rows]
    if len({w.record() for w in windows})!=len(windows):raise AssertionError('Duplicate selected window')
    # Independent stored-bit witnesses, ordered by brick for bounded reader IO.
    tasks=[(point,flag) for w in windows for point,flag in zip(w.witnesses,(1,2))]
    for point,flag in sorted(tasks,key=lambda p:tuple(x//64 for x in p[0])):
        if index.point_class(point)!=flag:raise AssertionError('Independent stored witness disagreement')
    selected_phase_wall=time.perf_counter()-start;selected_phase_cpu=time.process_time()-cpu
    control_start=time.perf_counter();control_cpu=time.process_time()
    selected_keys={(w.size,w.origin) for w in windows};control_report={}
    for side in WINDOW_SIDES:
        control,meta=uniform_candidates(index,source,side,settings['control_per_side'],settings['control_max_proposals_per_side'],keyed_seed(seed,scene,19001,side))
        for row in control:
            proof=index.query(tuple(row['origin']),side,witnesses=True,witness_seed=1);row['witnesses']=[proof.lit,proof.shadow]
            row['overlaps_selected_identity']=(side,tuple(row['origin'])) in selected_keys;row['control_only']=True;row['control_id']=len(controls)
            controls.append(row)
        control_report[str(side)]={**meta,'target':settings['control_per_side'],'complete':len(control)==settings['control_per_side']}
    control_tasks=[(point,flag) for row in controls for point,flag in zip(row['witnesses'],(1,2))]
    for point,flag in sorted(control_tasks,key=lambda p:tuple(x//64 for x in p[0])):
        if index.point_class(point)!=flag:raise AssertionError('Independent control witness disagreement')
    actual={side:sum(w.size==side for w in windows) for side in WINDOW_SIDES}
    report={'complete':actual==quotas,'termination':'quota_complete' if actual==quotas else 'candidate_discovery_exhausted','quotas':quotas,'side_counts':actual,
        'accepted':len(windows),'provenance_counts':dict(Counter(r['provenance'] for r in rows)),
        'by_side':by_side,'selection':selection,'guide_discovery':guide.report if guide else None,'independent_uniform_control':control_report,
        'independent_selected_witness_checks':len(tasks),'duplicate_selected_windows':0,'oracle_calls':0,'geometry_oracle_used':False,
        'independent_control_witness_checks':len(control_tasks),
        'uniform_control_used_for_selection':False,'settings':settings,'index_ram_bytes':index.nbytes,'index_queries':dict(Counter(index.counters)-counter_before),
        'selector_wall_seconds':selected_phase_wall,'selector_process_seconds':selected_phase_cpu,
        'control_wall_seconds':time.perf_counter()-control_start,'control_process_seconds':time.process_time()-control_cpu,
        'placement':placement_summary(windows,n),'wall_seconds':time.perf_counter()-start,'process_seconds':time.process_time()-cpu,
        'is_dataset':False,'production_ready':False}
    if root:
        atomic_json(root/'selection.json',report)
        for name,values in (('selected.jsonl',rows),('uniform-control.jsonl',controls)):
            (root/name).write_text(''.join(json.dumps(row)+'\n' for row in values))
    return SelectionResult(windows,rows,controls,report)
