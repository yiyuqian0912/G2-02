"""Whole-region sufficient proofs, not sampled topology or complete inverse sets."""
from fractions import Fraction as Q
from itertools import product
import json
from pathlib import Path

import gmpy2 as mp
import numpy as np
import pytest

from experiments.gaussian_geometry.b2_native.api import NativeOracle
from experiments.gaussian_geometry.b2_native.controls import bridge
from experiments.gaussian_geometry.b2_regions.api import RegionOracle,extension
from experiments.gaussian_geometry.models import Kernel,KernelScene,Threshold
from experiments.gaussian_geometry.queries import FieldOracle


@pytest.fixture(scope='module',autouse=True)
def built():
    try:extension()
    except ImportError as e:pytest.skip(str(e))


def scene():
    return KernelScene('gaussian_sum',[Kernel([.4,.5,.5],np.eye(3)*.08),Kernel([.6,.5,.5],np.eye(3)*.08)],Threshold('binary64',.7))


def exact_box_hit(source,receiver,lo,hi):
    left,right=Q(0),Q(1)
    for a,b,l,h in zip(source,receiver,lo,hi):
        a,b,l,h=map(lambda x:Q(float(x)),(a,b,l,h));d=b-a
        if not d:
            if not l<=a<=h:return False
            continue
        u,v=sorted(((l-a)/d,(h-a)/d));left=max(left,u);right=min(right,v)
        if left>right:return False
    return True


def test_independent_geometric_screens_against_exact_rational_clipping():
    rng=np.random.default_rng(4138);module=extension()
    for _ in range(160):
        s=rng.uniform(.01,.99,3);lo=rng.uniform(.05,.7,3);hi=lo+rng.uniform(.005,.2,3)
        a=rng.uniform(.05,.8,3);b=a+rng.uniform(.001,.16,3)
        r=module.geometry_screen(s,lo,hi,a,b)
        receivers=np.r_[np.array(list(product(*zip(lo,hi)))),rng.uniform(lo,hi,(12,3))]
        truth=[exact_box_hit(s,x,a,b) for x in receivers]
        for name,value in r.items():
            if value:assert all(truth) if name.endswith('all_shadow') else not any(truth)
    # Closed tangent/end contact is never certified as a volume exclusion.
    for source,receiver in [([0,.4,.4],[1,.4,.4]),([0,.5,.5],[.4,.5,.5]),([.4,.5,.5],[.8,.5,.5]),([.4,.5,.5],[0,.5,.5])]:
        r=module.geometry_screen(source,receiver,receiver,[.4]*3,[.6]*3)
        assert not r['primary_volume_excluded'] and not r['reference_volume_excluded']


@pytest.mark.parametrize('independent',[False,True])
def test_parameter_partition_proves_diagonal_empty_ray_volume(independent):
    s=KernelScene('gaussian_sum',[Kernel([.15,.4,.5],np.eye(3)*.05)],Threshold('exp',-.5))
    source=[.1]*3;lo=[.89]*3;hi=[.91]*3
    assert NativeOracle(s,independent=independent).certificate(source,lo,hi)[0]<0
    e=RegionOracle(s,independent=independent,mode=2,parameter_nodes=63)
    (y,m),work=e.region(source,lo,hi)
    assert (y,m)==(1,0) and work['parameter_whole_volume_FREE']>0
    assert e.region(source,lo,hi)[0]==(y,m)


@pytest.mark.parametrize('independent',[False,True])
def test_fusion_shadow_uses_full_sum_without_any_individual_inner_hit(independent):
    s=bridge();source=[.575,.5,.05];lo=[.574,.499,.94];hi=[.576,.501,.96]
    e=RegionOracle(s,independent=independent,mode=3)
    (y,m),work=e.region(source,lo,hi)
    assert y==0 and m==0 and work['fused_cross_section_SHADOW']>0
    if not independent:assert NativeOracle(s).certificate(source,lo,hi)[0]<0
    old=FieldOracle(s)
    for p in product(*zip(lo,hi)):assert old.segment(source,p).status=='HIT'


@pytest.mark.parametrize('independent',[False,True])
def test_multiple_corner_blockers_do_not_prove_whole_region_shadow(independent):
    s=KernelScene('gaussian_sum',[Kernel([.45,y,z],np.eye(3)*.045) for y,z in product((.34,.66),repeat=2)],Threshold('exp',-.5))
    source=[.05,.5,.5];lo=[.75,.2,.2];hi=[.85,.8,.8]
    assert NativeOracle(s).segment(source,[.8,.5,.5]).status=='MISS'
    for mode in (2,3,4):assert RegionOracle(s,independent=independent,mode=mode).region(source,lo,hi)[0][0]!=0


@pytest.mark.parametrize('independent',[False,True])
def test_all_new_modes_complete_small_fields_match_frozen_scalar(independent):
    source=[.03125]*3;s=scene();a=NativeOracle(s,independent=independent).field(source,8,30,scalar=True);assert a['complete']
    for mode in (2,3,4,5):
        v=RegionOracle(s,independent=independent,mode=mode,cover_depth=4,cover_nodes=5000).field(source,8,30)
        assert v['complete'];np.testing.assert_array_equal(v['response'],a['response']);np.testing.assert_array_equal(v['solid'],a['solid'])
        assert not RegionOracle.ready_capable


@pytest.mark.parametrize('independent',[False,True])
def test_pending_unknown_cover_is_not_discarded(independent):
    s=scene();source=[.03125]*3
    v=RegionOracle(s,independent=independent,mode=5,cover_nodes=1,cover_depth=10).field(source,8,30)
    old=NativeOracle(s,independent=independent).field(source,8,30)
    assert v['complete'] and v['work']['cover_UNKNOWN_leaves']==1
    np.testing.assert_array_equal(v['response'],old['response']);np.testing.assert_array_equal(v['solid'],old['solid'])


def test_fast_exp_enclosures_contain_independent_256_bit_results():
    module=extension();rng=np.random.default_rng(7518)
    values=np.r_[0.,-np.nextafter(0.,1.),-np.finfo(float).tiny,-.03125,-1.,-32.,-700.,-744.,-745.,-746.,-1074.,-1e308,-np.inf,np.inf,709.,710.,1000.,rng.uniform(-800,1,256)]
    for x in values:
        lo,hi,_=module.exponent(float(x),True)
        with mp.context(precision=256,round=mp.RoundDown):low=mp.exp(mp.mpfr(float(x)))
        with mp.context(precision=256,round=mp.RoundUp):high=mp.exp(mp.mpfr(float(x)))
        assert mp.mpfr(lo)<=low<=high<=mp.mpfr(hi),(x,lo,hi,low,high)
    with pytest.raises(ValueError):module.exponent(float('nan'),True)


@pytest.mark.parametrize('independent',[False,True])
def test_budget_invalid_and_exact_contact_remain_fail_closed(independent):
    e=RegionOracle(scene(),independent=independent,mode=5,cover_seconds=.001)
    v=e.field([.03125]*3,8,1e-9);assert not v['complete'] and not v['valid'].any()
    assert e.region([0,0,0],[1,0,0],[0,1,1])[0]==(-2,-2)
    with pytest.raises(ValueError):e.field([0,0,0],1024,1)
    with pytest.raises(TypeError):e.field([Q(1,3)]*3,8,1)
    with pytest.raises(ValueError):RegionOracle(scene(),mode=6).field([.01]*3,8,1)
    s=KernelScene('gaussian_sum',[Kernel([x,.5,.5],np.eye(3)*.125,.5) for x in (.4375,.5625)],Threshold('exp',-.25))
    # General transcendental contact remains unresolved in original fallbacks;
    # no new region is allowed to certify the containing volume all-visible.
    assert RegionOracle(s,independent=independent,mode=4).region([.5,.5625,.03125],[.5,.5625,.96875],[.5,.5625,.96875])[0][0]!=1


def test_old_native_sources_keep_measured_identity():
    from experiments.gaussian_geometry.b2_native.build import sources
    root=Path(__file__).parents[1]
    old=json.loads((root/'results/b2-native-measured-v1.json').read_text())
    assert sources()==old['native_build']['source_sha256']


@pytest.mark.parametrize('independent',[False,True])
def test_unaligned_tiles_match_complete_field_without_world_sized_allocation(independent):
    s=scene();source=[.03125]*3;full=NativeOracle(s,independent=independent).field(source,16,30)
    origin=[3,1,7];e=RegionOracle(s,independent=independent,cover_depth=3)
    tile=e.tile(source,16,origin,8,30);assert tile['complete'] and tile['response'].shape==(8,)*3
    for k in ('response','solid'):np.testing.assert_array_equal(tile[k],full[k][7:15,1:9,3:11])
    assert tile['origin_xyz']==origin and tile['world_resolution']==16
    with pytest.raises(ValueError):e.tile(source,16,[9,0,0],8,30)
    with pytest.raises(TypeError):e.tile(source,16,[1.,0.,0.],8,30)


def test_fast_filter_region_decisions_and_fallback_remain_certified():
    s=KernelScene('gaussian_sum',[Kernel([.15,.4,.5],np.eye(3)*.05)],Threshold('exp',-.5))
    source=[.1]*3;lo=[.89]*3;hi=[.91]*3
    a=RegionOracle(s,mode=2,fast_exp=False).region(source,lo,hi)
    b=RegionOracle(s,mode=2,fast_exp=True).region(source,lo,hi)
    assert a[0]==b[0]==(1,0) and b[1]['fast_exp_polynomials']>0
    module=extension();x=np.array([-1.,-10.,-700.,0.,-np.nextafter(0.,1.)]);tau=np.array([float(mp.exp(mp.mpfr(v))) for v in x]);tau=np.maximum(tau,np.nextafter(0.,1.))
    plain,_=module.exp_classes(x,tau,False,True);fast,stats=module.exp_classes(x,tau,True,True)
    assert plain==fast and stats['high_precision_fallbacks']>0 and fast[-1]==-1


@pytest.mark.parametrize('independent',[False,True])
def test_source_endpoint_contact_is_shadow_even_directed_away(independent):
    # This source is intentionally invalid for dataset generation. The physical
    # closed-segment diagnostic still cannot erase contact at t=0.
    s=KernelScene('gaussian_sum',[Kernel([.5]*3,np.eye(3)*.125)],Threshold('exp',-.5))
    v=RegionOracle(s,independent=independent,mode=5,cover_depth=3).field([.625,.5,.5],8,30)
    assert v['complete'] and not v['response'].any()


def test_optional_leaf_trace_matches_actual_ray_accounting():
    s=scene();source=[.03125]*3
    for independent in (False,True):
        a=RegionOracle(s,independent=independent,cover_depth=3).field(source,8,30)
        b=RegionOracle(s,independent=independent,cover_depth=3,trace=True).field(source,8,30)
        for k in ('response','solid','valid'):np.testing.assert_array_equal(a[k],b[k])
        count=np.count_nonzero(np.isin(b['evaluation_path'],[6,7,8]))
        assert count==b['work'].get('reference_ray_queries' if independent else 'ray_queries',0)
        assert b['ray_nanoseconds'].dtype==np.uint64


def test_all_fast_diagnostics_reject_wrong_rounding_environment():
    import ctypes
    libc=ctypes.CDLL(None);saved=libc.fegetround();m=extension()
    try:
        assert libc.fesetround(0x400)==0
        with pytest.raises(RuntimeError):m.geometry_screen([0,0,0],[1,1,1],[1,1,1],[.4]*3,[.6]*3)
        with pytest.raises(RuntimeError):m.exponent(-1.,True)
        with pytest.raises(RuntimeError):RegionOracle(scene(),independent=True)
    finally:libc.fesetround(saved)
    assert RegionOracle(scene()).region([2,0,0],[.4]*3,[.6]*3)[0]==(-2,-2)
