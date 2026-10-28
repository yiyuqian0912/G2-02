"""Small deterministic experiments; no training/data download. Run from any cwd."""
from pathlib import Path
import sys, os, json, time, itertools
from fractions import Fraction as F
os.environ.setdefault('NUMBA_NUM_THREADS','2')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/rind-adjacency-numba')
os.environ.setdefault('MPLCONFIGDIR','/tmp/rind-adjacency-matplotlib')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent/'Phase2_Multi_Source/vendor/rind-dataset'))
import numpy as np
from rind_dataset.render import render_region
from rind_dataset.geometry import is_driver_valid_mixed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
# Embed an exact integer configuration in a 1024 world. W is a quadtree leaf:
# A lies in its parent [288,320)^2, outside W=[288,304)^2.
SHIFT=256
SOURCES=np.array([[52,61],[4,0],[11,22]])+SHIFT
PARAMS=np.zeros((2,17)); PARAMS[:,:5]=[[24+SHIFT,17+SHIFT,3,4,0],[22+SHIFT,45+SHIFT,4,6,0]]
TYPES=np.zeros(2,np.uint8)
X=Y=32+SHIFT; L=16
A=np.array([.3,.2,.4])

def render(s,backend='numba'):
    return render_region(s,PARAMS,obs_types=TYPES,x=X,y=Y,size=L,backend=backend)

def rational_visible(s,p):
    # Independent exact closed-box slab predicate using rational pixel centers.
    for cx,cy,hw,hh,_ in PARAMS[:,:5].astype(int):
        lo,hi=F(0),F(1)
        for u,v,b0,b1 in zip(s,p,[cx-hw,cy-hh],[cx+hw,cy+hh]):
            u=F(int(u)); d=v-u
            if not d:
                if not b0<=u<=b1:lo,hi=F(1),F(0);break
            else:
                t0,t1=sorted([(F(int(b0))-u)/d,(F(int(b1))-u)/d])
                lo=max(lo,t0);hi=min(hi,t1)
        if lo<=hi:return False
    return True

def integrate(y,edges,tol=1e-10):
    labels=sorted(set(round(abs(float(y[v]-y[u])),10) for u,v in edges if abs(y[v]-y[u])>tol))
    graph=[[] for _ in y]
    for u,v in edges:graph[u].append(v);graph[v].append(u)
    masks=[]
    for a in labels:
        t=np.full(len(y),np.nan);t[0]=0;queue=[0]
        for u in queue:
            for v in graph[u]:
                dy=y[v]-y[u]; delta=int(np.sign(dy)) if abs(abs(dy)-a)<tol else 0
                z=t[u]+delta
                if np.isnan(t[v]):t[v]=z;queue.append(v)
                elif t[v]!=z:raise ValueError('nonzero cycle integral')
        if np.isnan(t).any() or np.ptp(t)!=1:raise ValueError('not connected/binary')
        masks.append(t-t.min())
    masks=np.array(masks);res=y-np.array(labels)@masks
    if np.ptp(res)>tol or res.min()<-tol:raise ValueError('invalid constant residual')
    if res.mean()>tol:labels.append(float(res.mean()));masks=np.vstack([masks,np.ones(len(y))])
    return np.array(labels),masks.astype(bool)

def main():
    started=time.perf_counter();timing={}
    states=np.array([render(s) for s in SOURCES]); V=states==1
    timing['initial_render_including_jit_seconds']=time.perf_counter()-started
    assert not (states==2).any()
    t=time.perf_counter()
    refs=np.array([render(s,'reference')==1 for s in SOURCES])
    exact=np.array([[rational_visible(s,(F(2*(X+x)+1,2),F(2*(Y+y)+1,2))) for y in range(L) for x in range(L)] for s in SOURCES]).reshape(V.shape)
    assert np.array_equal(V,refs) and np.array_equal(V,exact)
    timing['reference_and_rational_verification_seconds']=time.perf_counter()-t
    edges=np.array([(y*L+x,y*L+x+1) for y in range(L) for x in range(L-1)]+[(y*L+x,(y+1)*L+x) for y in range(L-1) for x in range(L)])
    flat=V.reshape(3,-1); d=flat[:,edges[:,1]].astype(int)-flat[:,edges[:,0]].astype(int)
    mixed=(d>0).any(0)&(d<0).any(0)
    codes=np.array([1,2,4])@flat; y=A@flat
    assert set(np.unique(codes))=={3,5,7} and not mixed.any()
    t=time.perf_counter(); aa,bb=integrate(y,edges)
    assert all(any(abs(a-c)<1e-9 and np.array_equal(b,m) for c,m in zip(aa,bb)) for a,b in zip(A,flat))
    error=float(np.max(abs(aa@bb-y)))
    timing['label_recovery_seconds']=time.perf_counter()-t
    # Finite fixed source domain near the embedded geometry, not whole 1024 domain.
    pool=np.array([(x+SHIFT,z+SHIFT) for z in range(0,64,2) for x in range(0,64,2)
                   if not(32<=x<48 and 32<=z<48)])
    pool=np.unique(np.vstack([pool,SOURCES]),axis=0)
    pool=np.array([s for s in pool if is_driver_valid_mixed(s,TYPES,PARAMS,2,1024)])
    t=time.perf_counter();D=np.array([render(s)==1 for s in pool]).reshape(len(pool),-1)
    timing['pool_render_seconds']=time.perf_counter()-t
    dp=D[:,edges[:,1]].astype(int)-D[:,edges[:,0]].astype(int)
    plus=(dp>0).mean(0);minus=(dp<0).mean(0)
    # Select pair using geometry/domain, BEFORE random source trials; it is then fixed.
    ix=int(np.argmax(plus*minus));pp=float(plus[ix]);pm=float(minus[ix]);p0=1-pp-pm
    rng=np.random.default_rng(20261010); stats=[]
    enter=int(np.flatnonzero(dp[:,ix]>0)[0]); leave=int(np.flatnonzero(dp[:,ix]<0)[0])
    mixed_witness=dict(enter_source=pool[enter].tolist(),leave_source=pool[leave].tolist(),
        pixel_centers=[[X+int(p)%L+.5,Y+int(p)//L+.5] for p in edges[ix]],
        visibility=D[[enter,leave]][:,edges[ix]].astype(int).tolist())
    for k in (1,2,3,4):
        draws=rng.integers(len(pool),size=(100000,k)); switches=dp[draws,ix]
        mm=(switches>0).any(1)&(switches<0).any(1);changed=(switches!=0).any(1)
        theory=1-(1-pp)**k-(1-pm)**k+p0**k
        stats.append(dict(k=k,trials=len(draws),theory=theory,observed=float(mm.mean()),bound=k*(k-1)*pp*pm,
                          changed_trials=int(changed.sum()),conditional_theory=theory/(1-p0**k) if p0<1 else None,
                          conditional_observed=float(mm.sum()/changed.sum()) if changed.any() else None,
                          monte_carlo_standard_error=float(np.sqrt(theory*(1-theory)/len(draws)))))
    # Exhaustive labelled triples on a small declared pool, known original strengths.
    ids=np.unique(np.r_[np.linspace(0,len(pool)-1,13,dtype=int),*[np.flatnonzero(np.all(pool==s,axis=1))[0] for s in SOURCES]])
    small=D[ids]; t=time.perf_counter()
    factors=[np.flatnonzero(np.all(small==b,axis=1)).tolist() for b in flat]
    timing['factor_matching_seconds']=time.perf_counter()-t
    predicted=set(itertools.product(*factors)); feasible=set();t=time.perf_counter()
    for z in itertools.product(range(len(ids)),repeat=3):
        if np.max(abs(A@small[list(z)]-y))<1e-10:feasible.add(z)
    assert predicted==feasible and feasible
    timing['finite_joint_exhaustion_seconds']=time.perf_counter()-t
    sample_error=max(float(np.max(abs(A@small[list(z)]-y))) for z in predicted)
    result=dict(scope='constructed geometry; fixed finite candidate domain, not generator statistics',world_size=1024,
        window=[X,Y,L],sources=SOURCES.tolist(),strengths=A.tolist(),rectangle_params=PARAMS[:,:5].tolist(),
        patterns={str(c):dict(pixels=int((codes==c).sum()),response=float(y[codes==c][0])) for c in np.unique(codes)},
        legal_neighbor_edges=len(edges),mixed_edges=int(mixed.sum()),reference_numba_exact_rational_agree=True,
        recovered_strengths=aa.tolist(),reconstruction_max_error=error,
        greedy_minimum=float(y.min()),greedy_minimum_is_true_effective_strength=bool(np.isclose(A,y.min()).any()),
        source_pool_size=len(pool),fixed_pair_pixel_indices=edges[ix].tolist(),mixed_witness=mixed_witness,p_plus=pp,p_minus=pm,monte_carlo=stats,
        finite_pool_check=dict(n=len(ids),joint_configs=len(ids)**3,factor_candidate_comparisons=3*len(ids),
             feasible_joint_configs=len(feasible),retained_joint_configs=len(predicted),all_feasible_retained=predicted==feasible,
             max_sample_reconstruction_error=sample_error,known_strengths=True,scope='only declared finite pool and fixed labelled strengths'),
        timing=timing)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,4,figsize=(12,3),layout='constrained')
    im=axes[0].imshow(y.reshape(L,L),vmin=0,vmax=1,cmap='viridis');axes[0].set_title('Observed: 0.5 / 0.7 / 0.9');fig.colorbar(im,ax=axes[0])
    for ax,b,a in zip(axes[1:],V,A):ax.imshow(b,vmin=0,vmax=1,cmap='gray');ax.set_title(f'True visibility, a={a}')
    for ax in axes:ax.set_xticks([]);ax.set_yticks([])
    fig.savefig(OUT/'counterexample.png',dpi=170);plt.close(fig)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
