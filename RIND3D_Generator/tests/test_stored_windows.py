"""Current three-side contract, exact fallback, and soft-only diversity."""
from collections import Counter
from itertools import product
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import chisquare
from rind3d.information import InformationIndex
from rind3d.pack import RINDPack
from rind3d.stored_windows import select_stored_windows,selection_settings,uniform_candidates
from rind3d.window_contract import balanced_quotas,PRODUCTION_SIDE_COUNTS,WINDOW_SIDES
from rind3d.windows import Window,excludes
from rind3d.origin_domains import OriginBox,voxel_pair_origins,sample_origins
from rind3d.window_diversity import soft_select
from test_information_index import write_fields,brute


def test_three_sides_and_balanced_counts():
    assert WINDOW_SIDES==(64,128,256) and PRODUCTION_SIDE_COUNTS=={64:1365,128:1365,256:1366}
    assert balanced_quotas(32)=={64:10,128:11,256:11}
    assert balanced_quotas(128)=={64:42,128:43,256:43}
    assert balanced_quotas(512)=={64:170,128:171,256:171}
    with pytest.raises(ValueError):Window((0,0,0),32)


def test_integer_volume_sampler_matches_brute_force_with_source_exclusion():
    n=9;side=4;p=(4,4,4);q=(5,5,5);source=np.array([3,4,6])/n
    boxes=voxel_pair_origins(p,q,source,side,n)
    exact={o for o in product(range(n-side+1),repeat=3) if all(o[i]<=p[i]<o[i]+side and o[i]<=q[i]<o[i]+side for i in range(3))
           and not all(o[i]<=source[i]*n<o[i]+side for i in range(3))}
    assert set(sample_origins(boxes,len(exact),np.random.default_rng(1)))==exact
    rng=np.random.default_rng(618);counts=Counter(sample_origins(boxes,1,rng)[0] for _ in range(12000))
    assert chisquare([counts[p] for p in sorted(exact)]).pvalue>.001
    assert sample_origins(boxes,8,np.random.default_rng(8))==sample_origins(boxes[::-1],8,np.random.default_rng(8))


def test_unit_volume_weight_and_large_domains_without_enumeration():
    boxes=[OriginBox((0,0,0),(0,0,0)),OriginBox((2,0,0),(3,1,1))];rng=np.random.default_rng(1618)
    counts=Counter(sample_origins(boxes,1,rng)[0] for _ in range(9000))
    assert abs(counts[(0,0,0)]/9000-1/9)<.02
    values=sample_origins([OriginBox((0,0,0),(1023,1023,1023))],4096,np.random.default_rng(1))
    assert len(values)==len(set(values))==4096


@pytest.fixture(scope='module')
def fields(tmp_path_factory):
    root=tmp_path_factory.mktemp('stored-windows')
    paths=write_fields(root/'plane',(512,)*3,lambda x,y,z:(x>=257,np.array(False)))
    rare=write_fields(root/'rare',(512,)*3,lambda x,y,z:((x==256)&(y==256)&(z==256),np.array(False)))
    corner=write_fields(root/'corner',(512,)*3,lambda x,y,z:((x==0)&(y==0)&(z==0),np.array(False)))
    return paths,rare,corner


def small_settings(**changes):
    return selection_settings({'uniform_max_proposals_per_side':4096,'uniform_pilot_proposals':64,
        'guide_max_regions':32,'guide_max_pairs':64,'guide_max_proposals_per_side':4096,
        'control_per_side':2,'control_max_proposals_per_side':128,**changes})


def check_result(result,index,source):
    assert len({w.record() for w in result.windows})==len(result.windows)
    for w,row in zip(result.windows,result.provenance):
        assert excludes(w,source,index.shape[0]) and brute(index.response,index.solid,w.origin,w.size)==3
        assert row['provenance'] in ('uniform_accepted','guided_informative') and row['selection_mode'] in ('uniform','soft')
        assert row['window_id']>=0 and row['witnesses'] is not None
        for p,label in zip(w.witnesses,(1,2)):assert index.point_class(p)==label


def test_uniform_exact_acceptance_determinism_control_independence_and_unaligned_origins(fields):
    paths=fields[0];source=[.03,.04,.05]
    with RINDPack(paths[0]) as r,RINDPack(paths[1]) as m:
        index=InformationIndex.build(r,m);settings=small_settings(selection_mode='uniform',guided_fallback=False)
        a=select_stored_windows(index,source,32,seed=19,settings=settings)
        b=select_stored_windows(index,source,32,seed=19,settings={**settings,'control_per_side':0})
        assert a.windows==b.windows and a.report['complete'] and a.report['side_counts']==balanced_quotas(32)
        assert a.report['guide_discovery'] is None and a.report['uniform_control_used_for_selection'] is False
        assert a.report['independent_control_witness_checks']==2*len(a.controls)
        assert set(row['provenance'] for row in a.provenance)=={'uniform_accepted'}
        assert any(any(v%16 for v in w.origin) for w in a.windows)
        check_result(a,index,source)


def test_rare_scene_triggers_stored_witness_fallback_and_allows_region_reuse(fields):
    paths=fields[1];source=[.95]*3
    with RINDPack(paths[0]) as r,RINDPack(paths[1]) as m:
        index=InformationIndex.build(r,m);settings=small_settings()
        result=select_stored_windows(index,source,32,seed=15,settings=settings)
        assert result.report['complete'] and result.report['oracle_calls']==0
        assert result.report['provenance_counts']['guided_informative']>0
        assert result.report['guide_discovery']['labels']=='actual stored response AND solid bits'
        assert any(x['guided'].get('maximum_pair_reuse',0)>1 for x in result.report['by_side'].values())
        assert result.windows==select_stored_windows(index,source,32,seed=15,settings=settings).windows
        check_result(result,index,source)


def test_insufficient_distinct_legal_origins_is_explicit_not_padded(fields):
    paths=fields[2]
    with RINDPack(paths[0]) as r,RINDPack(paths[1]) as m:
        index=InformationIndex.build(r,m);result=select_stored_windows(index,[.95]*3,32,seed=15,settings=small_settings())
        assert not result.report['complete'] and len(result.windows)<=3
        assert all(w.origin==(0,0,0) for w in result.windows) and result.report['duplicate_selected_windows']==0


def test_soft_selection_has_no_bucket_or_center_spacing_gate_and_swaps_improve():
    origins=[(i,j,10) for i in range(12) for j in range(12)]
    settings=small_settings();selected,scores,report=soft_select(origins,64,128,settings,17)
    assert len(selected)==128 and len(set(selected))==128
    assert report['hard_bucket_cap'] is None and report['hard_center_spacing'] is None
    assert report['occupied_center_cells']==1  # legal even though every center occupies one cell
    assert all(s['objective_gain']>0 for s in report['swaps'])
    assert report['score_evaluations']<=128*settings['greedy_evaluations_per_selected']+report['swap_attempts']
    assert selected==soft_select(origins,64,128,settings,17)[0]


def test_oversampling_shortfall_does_not_force_guidance_when_hard_quota_is_satisfied(fields):
    paths=fields[0]
    with RINDPack(paths[0]) as r,RINDPack(paths[1]) as m:
        index=InformationIndex.build(r,m)
        result=select_stored_windows(index,[.03]*3,32,seed=4,settings=small_settings(valid_pool_multiplier=100))
        assert result.report['complete'] and result.report['guide_discovery'] is None
