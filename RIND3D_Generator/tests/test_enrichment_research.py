import numpy as np
from tests.test_lite_production import box
from rind3d.rendering.context import GeometryContext
from rind3d.enrichment_research import search,SharedCandidateVerifier
from rind3d.geometry import Budget
from rind3d.windows import Window

def test_core_zero_budget_does_not_prepare_or_read_observations():
    class Forbidden:
        def __getattr__(self,name):raise AssertionError('Core optional discovery performed '+name)
    p=np.array([.1,.2,.3]);sources,proofs,r=search(Forbidden(),p,Window((700,700,700),64),Forbidden(),Forbidden(),seconds=0)
    assert np.array_equal(sources,p[None]) and not proofs and not r['preparation_performed'] and not r['uniqueness_proven']

def test_shared_candidate_bricks_overlap_and_mask_comparison():
    ctx=GeometryContext(box());p=np.array([.1,.2,.3]);model=ctx.bind(p);v=SharedCandidateVerifier(ctx)
    for independent in (False,True):
        first=Window((511,385,383),64);second=Window((512,385,383),64)
        for w in (first,second):
            y,m,_=model.render(w.origin,w.size,1024)
            proof=v.verify(p,w,y,m,independent=independent,budget=Budget(20))
            assert proof['status']=='grid_exact' and proof['covered_voxels']==64**3
        assert v.stats['brick_cache_hits']>0
        y,m,_=model.render(second.origin,second.size,1024);m=m.copy();m[0,0,0]=~m[0,0,0]
        assert v.verify(p,second,y,m,independent=independent,budget=Budget(20))['status']=='mismatch'
