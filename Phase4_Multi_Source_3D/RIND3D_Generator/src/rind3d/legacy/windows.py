# LEGACY DIAGNOSTICS ONLY: copied before the three-side contract correction.
"""Stored-response sampling and a separate diagnostic, proof-witness pilot."""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import numpy as np
from scipy.stats import qmc
from ..geometry import Budget,Aborted,centres,keyed_rng,render_region
from .window_contract import WINDOW_SIDES,PILOT_SIDE_COUNTS,PRODUCTION_SIDE_COUNTS,require_window_side,side_counts,validate_window_config

def sobol_points(seed,scene,stream,n,dimension=3):
    seed=int(np.random.SeedSequence([int(seed),int(scene),int(stream)]).generate_state(1)[0])
    return qmc.Sobol(d=dimension,scramble=True,seed=seed,bits=52).random_base2(max(0,int(np.ceil(np.log2(max(1,n))))))[:n]

@dataclass(frozen=True)
class Window:
    origin:tuple[int,int,int]
    size:int
    stratum:int|None=None  # compatibility name: index of one exact side, never a range
    witnesses:tuple=()

    def __post_init__(self):
        side=require_window_side(self.size);index=WINDOW_SIDES.index(side)
        if self.stratum is not None and self.stratum!=index:raise ValueError('Window side/category mismatch')
        object.__setattr__(self,'stratum',index)

    def record(self):return (*self.origin,self.size)

def excludes(window,source,resolution):
    p=np.asarray(source)*resolution;o=np.asarray(window.origin)
    return not bool(np.all((p>=o)&(p<o+window.size)))

def informative(response,solid):
    free=~np.asarray(solid,dtype=bool)
    values=np.asarray(response,dtype=bool)
    n=int(free.sum());lit=int((values&free).sum())
    return n>1 and 0<lit<n,{'free_voxels':n,'lit_free_voxels':lit}

def placement_summary(windows,resolution):
    bins=Counter(tuple(np.minimum(1,((np.asarray(w.origin)+w.size/2)/resolution*2).astype(int))) for w in windows)
    counts=side_counts(w.size for w in windows)
    return {'size_histogram':{str(k):v for k,v in counts.items()},'side_counts':counts,
            'occupied_octants':len(bins),'octants':{','.join(map(str,k)):v for k,v in bins.items()}}

def sample_windows(source,cfg,scene_id,count,*,response=None,solid=None,oracle=None,
                   proposal_stream=302,exclude_records=()):
    """Production checks stored truth; pilot accepts only proven positive/negative
    free witnesses. Inconclusive pilot probes are never called constant truth.
    """
    validate_window_config(cfg)
    resolution=int(cfg['resolution']);rng=keyed_rng(cfg['seed'],scene_id,proposal_stream-1)
    if count not in (32,4096):raise ValueError('Balanced sampling requires 32 pilot or 4096 production windows')
    quota_map=PILOT_SIDE_COUNTS if count==32 else PRODUCTION_SIDE_COUNTS
    quotas=[quota_map[side] for side in WINDOW_SIDES]
    # Only position is sampled continuously. Side comes directly from the enum.
    proposals=sobol_points(cfg['seed'],scene_id,proposal_stream,cfg['max_window_attempts'],3)
    accepted=[[] for _ in WINDOW_SIDES];seen=set(exclude_records);reasons=Counter();attempts=0
    budget=Budget(cfg.get('sampling_seconds',30),cfg.get('sampling_points',50_000_000))
    def summary():
        result=[w for group in accepted for w in group]
        report={'attempts':attempts,'accepted':len(result),'rejections':dict(reasons),
                'interrupted_attempts':attempts-len(result)-sum(reasons.values()),
                'target':count,'expected_side_counts':dict(zip(WINDOW_SIDES,quotas)),'placement_version':cfg['placement_version'],'proposal_stream':proposal_stream,
                'sampling_mode':'stored-full-window' if response is not None else 'pilot-certified-positive-negative-witnesses',
                **placement_summary(result,resolution)}
        return result,report
    def checked(call,*args,**kwargs):
        # Preserve partial rejection/placement work if a certified query aborts.
        try:return call(*args,**kwargs)
        except RuntimeError as exc:
            exc.partial_windows,exc.sampling_report=summary()
            raise
    for v in proposals:
        checked(budget.check);eligible=[i for i in range(len(WINDOW_SIDES)) if len(accepted[i])<quotas[i]]
        if not eligible:break
        s=eligible[attempts%len(eligible)];attempts+=1
        side=WINDOW_SIDES[s]
        origin=tuple(np.floor(v[:3]*(resolution-side+1)).astype(int).tolist());w=Window(origin,side,s)
        if w.record() in seen:reasons['duplicate']+=1;continue
        seen.add(w.record())
        if not excludes(w,source,resolution):reasons['contains_true_source']+=1;continue
        if response is not None:
            y=response.read_window(*origin,side);m=solid.read_window(*origin,side)
            ok,counts=informative(y,m)
            if not ok:reasons['constant_or_no_free']+=1;continue
            pos=np.unravel_index(int(np.argmax(y&~m)),y.shape);neg=np.unravel_index(int(np.argmax(~y&~m)),y.shape)
            witnesses=tuple(tuple(int(a+o) for a,o in zip(point[::-1],origin)) for point in (pos,neg))
            w=Window(origin,side,s,witnesses)
        else:
            lo=(np.asarray(origin)+.5)/resolution;hi=(np.asarray(origin)+side-.5)/resolution
            yv,mv=checked(oracle.certificate,source,lo,hi,budget=budget)
            if yv>=0:reasons['certified_constant_or_solid']+=1;continue
            coordinates=rng.integers(0,side,(48,3))+origin
            p=(coordinates+.5)/resolution
            mask=checked(oracle.contains,p,budget=budget);y=checked(oracle.response,source,p,budget=budget)
            pos=np.flatnonzero(y&~mask);neg=np.flatnonzero(~y&~mask)
            if not len(pos) or not len(neg):
                # Explicit diagnostic rejection, not a claim about the complete
                # window. Production never uses this conditional proposal gate.
                reasons['pilot_probes_inconclusive']+=1;continue
            w=Window(origin,side,s,(tuple(map(int,coordinates[pos[0]])),tuple(map(int,coordinates[neg[0]]))))
        accepted[s].append(w)
    result,report=summary();rng.shuffle(result)
    return result,report
