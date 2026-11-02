"""Optional, bounded lit-ratio sampling. Natural D is called unchanged.

All bin objectives are soft and dataset-level. Unavailable bins are reported;
no window information, geometry or source acceptance rule is relaxed.
"""
from collections import Counter
from dataclasses import dataclass,field
import time
import numpy as np
from .stored_windows import select_stored_windows,uniform_candidates,keyed_seed,selection_settings,SelectionResult
from .window_contract import balanced_quotas
from .window_diversity import SpatialSelection
from .windows import Window,placement_summary

DEFAULT={'mode':'natural','natural':.3,'uniform':.5,'extreme':.2,'pool_multiplier':8,'max_proposals_per_side':65536,'control_per_side':128}

def settings(cfg=None):
    c={**DEFAULT,**(cfg or {})}
    if set(c)!=set(DEFAULT) or c['mode'] not in ('natural','stratified','mixed'):raise ValueError('Unknown observation sampling option')
    if any(not np.isfinite(c[k]) or c[k]<0 for k in ('natural','uniform','extreme')) or not np.isclose(sum(c[k] for k in ('natural','uniform','extreme')),1):raise ValueError('Mixture weights must sum to one')
    for k in ('pool_multiplier','max_proposals_per_side','control_per_side'):
        if isinstance(c[k],bool) or not isinstance(c[k],int) or c[k]<(0 if k=='control_per_side' else 1):raise ValueError('Invalid '+k)
    return c

@dataclass
class DatasetBalance:
    bins:np.ndarray=field(default_factory=lambda:np.zeros(10,dtype=np.int64))
    components:Counter=field(default_factory=Counter)
    def component(self,c):
        if c['mode']=='stratified':return 'uniform'
        total=sum(self.components.values())+1
        return max(('natural','uniform','extreme'),key=lambda k:(total*c[k]-self.components[k],k))
    def bin(self,component,available):
        if component=='natural':return None
        allowed=[b for b in available if component!='extreme' or b in (0,9)]
        if not allowed:allowed=list(available) # Explicit shortage, bounded fallback.
        n=int(self.bins.sum())+1
        return max(allowed,key=lambda b:((n/10-self.bins[b]) if component=='uniform' else (n*.5-self.bins[b]),-b))
    def record(self,component,bin):self.bins[bin]+=1;self.components[component]+=1
    def json(self):return {'bins':self.bins.tolist(),'components':dict(self.components)}

def annotate(row,c):
    return dict(row,lit_free=c.lit,shadow_free=c.shadow,solid=c.solid,lit_ratio=c.lit_ratio,ratio_bin=c.ratio_bin)

def select(info,count_index,source,*,seed,scene,balance=None,observation=None,natural_settings=None,count=256):
    c=settings(observation);balance=balance if balance is not None else DatasetBalance();start=time.perf_counter();cpu=time.process_time()
    if c['mode']=='natural':
        result=select_stored_windows(info,source,count,seed=seed,scene=scene,settings=natural_settings)
        result.provenance=[dict(annotate(row,count_index.query(w.origin,w.size)),sampling_mode='natural',mixture_component='natural') for row,w in zip(result.provenance,result.windows)]
        for row in result.provenance:balance.record('natural',row['ratio_bin'])
        result.report.update(observation_sampling=c,ratio_bins=dict(Counter(r['ratio_bin'] for r in result.provenance)),dataset_balance=balance.json(),
                             natural_sampler_unchanged=True,wall_with_counts_seconds=time.perf_counter()-start)
        return result
    quotas=balanced_quotas(count);s=selection_settings(natural_settings);rows=[];controls=[];reports={};n=info.shape[0]
    for side,target in quotas.items():
        candidates,meta=uniform_candidates(info,source,side,target*c['pool_multiplier'],c['max_proposals_per_side'],keyed_seed(seed,scene,61001,side))
        pool=[annotate(r,count_index.query(tuple(r['origin']),side)) for r in candidates]
        available=Counter(r['ratio_bin'] for r in pool);before=balance.json();selected=[];scores={};components={};shortages=Counter()
        spatial=SpatialSelection([r['origin'] for r in pool],side,s['coverage_cell_side'],s['overlap_weight'])
        rng=np.random.default_rng(keyed_seed(seed,scene,61002,side));remaining=set(range(len(pool)))
        while len(selected)<target and remaining:
            component=balance.component(c);bins=sorted({pool[i]['ratio_bin'] for i in remaining});chosen_bin=balance.bin(component,bins)
            if component=='extreme' and not any(b in (0,9) for b in bins):shortages['extreme_unavailable_fallback_all_bins']+=1
            eligible=[i for i in sorted(remaining) if chosen_bin is None or pool[i]['ratio_bin']==chosen_bin]
            shortlist=rng.choice(eligible,min(len(eligible),s['greedy_evaluations_per_selected']),replace=False).tolist()
            index=max(shortlist,key=lambda i:(spatial.gain(i),-i));scores[index]=spatial.gain(index);spatial.add(index)
            selected.append(index);components[index]=component;remaining.remove(index);balance.record(component,pool[index]['ratio_bin'])
        for i in selected:
            row=pool[i];proof=info.query(tuple(row['origin']),side,witnesses=True,witness_seed=keyed_seed(seed,scene,61003,side))
            if not proof.informative:raise AssertionError('Stored eligibility/count disagreement')
            row.update(window_id=len(rows),witnesses=[proof.lit,proof.shadow],witness_policy='index_hashed_tree_proof',
                       sampling_mode=c['mode'],mixture_component=components[i],selection_mode='ratio_stratified_soft_coverage',
                       selection_score=scores[i],pool_index=i,witness_pair_id=None,witness_region=None,origin_law=None)
            rows.append(row)
        control,control_meta=uniform_candidates(info,source,side,c['control_per_side'],c['max_proposals_per_side'],keyed_seed(seed,scene,19001,side))
        for row in control:
            proof=info.query(tuple(row['origin']),side,witnesses=True,witness_seed=1);row.update(witnesses=[proof.lit,proof.shadow],control_only=True,control_id=len(controls));controls.append(row)
        reports[str(side)]={'discovery':meta,'available_pool_bins':[available[i] for i in range(10)],'selected_bins':[sum(r['side']==side and r['ratio_bin']==b for r in rows) for b in range(10)],
                            'unavailable_bins':[b for b in range(10) if not available[b]],'fallbacks':dict(shortages),'balance_before':before,'control':control_meta}
    windows=[Window(tuple(r['origin']),r['side'],witnesses=tuple(map(tuple,r['witnesses']))) for r in rows]
    if len({w.record() for w in windows})!=len(windows):raise AssertionError('Duplicate selected windows')
    for w in windows:
        for p,label in zip(w.witnesses,(1,2)):
            if info.point_class(p)!=label:raise AssertionError('Exact free witness mismatch')
    actual={side:sum(w.size==side for w in windows) for side in quotas}
    report={'complete':actual==quotas,'accepted':len(windows),'quotas':quotas,'side_counts':actual,'observation_sampling':c,'by_side':reports,
            'ratio_bins':dict(Counter(r['ratio_bin'] for r in rows)),'dataset_balance':balance.json(),'uniform_target_deficits':[max(0,int(balance.bins.sum()/10)-int(v)) for v in balance.bins],
            'fallback_policy':'Use available bins with greatest cumulative deficit; extremes absent -> all bins; quota shortfall -> explicit INCOMPLETE.',
            'proposal_law':'uniform legal integer origins from scrambled Sobol, informative rejection; ratio/coverage selection is distribution-changing',
            'placement':placement_summary(windows,n),'duplicate_selected_windows':0,'oracle_calls':0,'wall_seconds':time.perf_counter()-start,
            'process_seconds':time.process_time()-cpu,'uniform_control_used_for_selection':False,'provenance_counts':dict(Counter(r['provenance'] for r in rows))}
    return SelectionResult(windows,rows,controls,report)
