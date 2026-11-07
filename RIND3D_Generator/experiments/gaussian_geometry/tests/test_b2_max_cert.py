"""Maximum-proof arithmetic, complete covers and fail-closed adversaries."""
import copy
import json
import os
from fractions import Fraction
from pathlib import Path

from flint import arb,ctx
import numpy as np
import pytest

from experiments.gaussian_geometry.b2_max_cert.api import MaximumOracle,CertificateChecker,dumps,extension
from experiments.gaussian_geometry.b2_native.api import NativeOracle
from experiments.gaussian_geometry.models import Kernel,KernelScene,Threshold,rational
from experiments.gaussian_geometry.queries import FieldOracle,Limits


@pytest.fixture(scope='module',autouse=True)
def built():
    try:extension()
    except ImportError as e:pytest.skip(str(e))


def mixture():
    return KernelScene('gaussian_sum',[Kernel([x,.5,.5],np.eye(3)*.1,.6) for x in (.42,.58)],Threshold('binary64',.9))


SOURCE=[.05,.5,.5]
RECEIVER=[.95,.5,.5]


@pytest.mark.parametrize('method',[1,2,3])
def test_continuous_mixture_miss_checked_complete_cover(method):
    scene=mixture();engine=MaximumOracle(scene,method=method);result=engine.segment(SOURCE,RECEIVER,proof=True)
    assert result['status']=='MISS' and result['certificate']
    check=CertificateChecker(scene).check(dumps(result['certificate']),SOURCE,RECEIVER)
    assert check['accepted']
    assert NativeOracle(scene).segment(SOURCE,RECEIVER).status=='MISS'
    assert NativeOracle(scene,independent=True).segment(SOURCE,RECEIVER).status=='MISS'


@pytest.mark.parametrize('tamper',['geometry','source','receiver','query','threshold','version','upper','gap','left_endpoint','right_endpoint','op','remove','nan','extra'])
def test_checker_rejects_tampering(tamper):
    scene=mixture();proof=MaximumOracle(scene).segment(SOURCE,RECEIVER,proof=True)['certificate'];bad=copy.deepcopy(proof)
    if tamper=='geometry':bad['geometry_id']='0'*64
    elif tamper=='source':bad['source_hex'][0]=float(.1).hex()
    elif tamper=='receiver':bad['receiver_hex'][0]=float(.9).hex()
    elif tamper=='query':bad['query_id']='0'*64
    elif tamper=='threshold':bad['threshold']['value_hex']=float(.99).hex()
    elif tamper=='version':bad['schema']='unsupported'
    elif tamper=='upper':bad['partition'][0]['upper']=float(0).hex()
    elif tamper=='gap':
        p=bad['partition'][0];p['lo']=np.nextafter(float.fromhex(p['lo']),np.inf).hex()
    elif tamper=='left_endpoint':bad['partition']=[p for p in bad['partition'] if float.fromhex(p['lo'])!=0]
    elif tamper=='right_endpoint':bad['partition']=[p for p in bad['partition'] if float.fromhex(p['hi'])!=1]
    elif tamper=='op':bad['partition'][0]['op']='sampled_fit'
    elif tamper=='remove':bad['partition']=[]
    elif tamper=='nan':bad['partition'][0]['upper']='nan'
    elif tamper=='extra':bad['partition'][0]['trusted_primary_label']='MISS'
    assert not CertificateChecker(scene).check(bad,SOURCE,RECEIVER)['accepted']


def test_overlap_allowed_but_gap_under_overlap_rejected():
    scene=mixture();p=MaximumOracle(scene).segment(SOURCE,RECEIVER,proof=True)['certificate'];checker=CertificateChecker(scene)
    p['partition'].extend(copy.deepcopy(p['partition']));p['partition'].reverse();assert checker.check(p,SOURCE,RECEIVER)['accepted']
    p['partition']=[r for r in p['partition'] if float.fromhex(r['lo'])!=0]
    assert not checker.check(p,SOURCE,RECEIVER)['accepted']


def test_duplicate_json_fields_and_unsupported_precision_rejected():
    scene=mixture();p=MaximumOracle(scene).segment(SOURCE,RECEIVER,proof=True)['certificate'];c=CertificateChecker(scene)
    encoded=dumps(p);assert not c.check(b'{"schema":"ignored",'+encoded[1:],SOURCE,RECEIVER)['accepted']
    assert not c.check(encoded,SOURCE,RECEIVER,precision=40)['accepted']


def test_dyadic_endpoints_are_exact_and_deterministic():
    scene=mixture();e=MaximumOracle(scene);a=e.segment(SOURCE,RECEIVER,proof=True);b=e.segment(SOURCE,RECEIVER,proof=True)
    assert dumps(a['certificate'])==dumps(b['certificate'])
    for row in a['certificate']['partition']:
        for key in ('lo','hi','upper'):
            f=float.fromhex(row[key]);q=Fraction(f);assert q.denominator&(q.denominator-1)==0


def test_unsupported_rational_endpoint_and_invalid_inputs():
    e=MaximumOracle(mixture())
    with pytest.raises(TypeError):e.segment([Fraction(1,3)]*3,RECEIVER)
    assert e.segment([np.nan,0,0],RECEIVER)['status']=='INVALID'
    assert e.segment(SOURCE,[np.inf,0,0])['status']=='INVALID'
    with pytest.raises(ValueError):MaximumOracle(mixture(),method=0).segment(SOURCE,RECEIVER)
    assert MaximumOracle(mixture(),max_nodes=0).segment(SOURCE,RECEIVER,proof=True)['certificate'] is None


@pytest.mark.parametrize('method',[1,2,3])
def test_nontrivial_transcendental_contact_remains_unknown(method):
    scene=KernelScene('gaussian_sum',[Kernel([x,.5,.5],np.eye(3)*.125,.5) for x in (.4375,.5625)],Threshold('exp',-.25))
    result=MaximumOracle(scene,method=method,max_nodes=256,max_depth=32).segment([.5,.5625,.03125],[.5,.5625,.96875],proof=True)
    assert result['status']=='UNKNOWN' and result['certificate'] is None


@pytest.mark.parametrize('method',[1,2,3])
def test_flat_quartic_maximum_degenerate_stationary_contact(method):
    scene=KernelScene('gaussian_sum',[Kernel([x,.5,.5],np.eye(3)*.125,.5) for x in (.375,.625)],Threshold('exp',-.5))
    r=MaximumOracle(scene,method=method,max_nodes=256,max_depth=32).segment([.0625,.5,.5],[.9375,.5,.5],proof=True)
    # Symmetric mixture: f' and f'' vanish at t=.5; exact max=exp(-.5).
    assert r['status'] in ('UNKNOWN','HIT') and r['certificate'] is None


@pytest.mark.parametrize('offset,expected',[(2**-20,'MISS'),(-2**-20,'HIT')])
def test_near_tangency_strict_sides(offset,expected):
    scene=KernelScene('gaussian_sum',[Kernel([x,.5,.5],np.eye(3)*.125,.5) for x in (.4375,.5625)],Threshold('exp',-.25))
    s=[.5,.5625+offset,.03125];x=[.5,.5625+offset,.96875]
    for method in (1,2,3):
        r=MaximumOracle(scene,method=method,max_nodes=8192).segment(s,x,proof=True);assert r['status']==expected
        if expected=='MISS':assert CertificateChecker(scene).check(r['certificate'],s,x)['accepted']


def test_closed_endpoints_and_analytic_fallback_not_false_miss():
    scene=KernelScene('gaussian_sum',[Kernel([.5]*3,np.eye(3),.5) for _ in range(2)],Threshold('exp',-.5))
    # Source on the exact closed boundary, directed away; old analytic reduction.
    s=[1.5,.5,.5];x=[2.,.5,.5]
    r=MaximumOracle(scene,max_nodes=32,max_depth=16,fallback=True).segment(s,x,proof=True)
    assert r['status']=='HIT' and r['fallback_used'] and r['certificate'] is None
    assert MaximumOracle(mixture()).segment([.5,.5,.5],[.9,.5,.5])['status']=='MISS'  # sum remains below .9


def test_many_local_maxima_and_thin_rotated_features():
    kernels=[Kernel([.15+.1*i,.5,.5],np.diag([.015,.04,.04]),.8) for i in range(8)]
    scene=KernelScene('gaussian_sum',kernels,Threshold('binary64',.85))
    s=[.01,.5,.5];x=[.99,.5,.5]
    for method in (1,2,3):
        r=MaximumOracle(scene,method=method).segment(s,x,proof=True);assert r['status']=='MISS'
        assert CertificateChecker(scene).check(r['certificate'],s,x)['accepted']
    m=np.array([[2**-18,2**-20,0],[0,.03,.002],[0,0,.04]])
    scene=KernelScene('gaussian_sum',[Kernel([.5]*3,m,.6),Kernel([.5+2**-20,.5,.5],m,.6)],Threshold('binary64',.9))
    for x in ([.9,.5,.5],[.9,.65,.5]):
        expected=NativeOracle(scene).segment(s,x).status;r=MaximumOracle(scene).segment(s,x)
        assert r['status']==expected


def test_positive_underflow_remainder_and_zero_length_segment():
    scene=KernelScene('gaussian_sum',[Kernel([.5]*3,np.eye(3))],Threshold('binary64',1e-300))
    s=[1000]*3;x=[1001]*3;r=MaximumOracle(scene).segment(s,x,proof=True)
    assert r['status']=='MISS';assert CertificateChecker(scene).check(r['certificate'],s,x)['accepted']
    assert any(float.fromhex(q['upper'])>0 for q in r['certificate']['partition'])
    r=MaximumOracle(mixture()).segment(SOURCE,SOURCE,proof=True);assert r['status']=='MISS'
    assert CertificateChecker(mixture()).check(r['certificate'],SOURCE,SOURCE)['accepted']


def test_primary_derivative_and_taylor_enclosures_against_arb_samples():
    scene=mixture();engine=MaximumOracle(scene);s=np.array(SOURCE);x=np.array(RECEIVER)
    coef=[k.coefficients(s,x) for k in scene.kernels]
    with ctx.workprec(256):
        for lo,hi in [(0,1),(.3,.4),(.45,.55),(.4999,.5001),(.0625,.53125)]:
            bound=engine.native.bounds(s,x,lo,hi)
            for t in np.linspace(lo,hi,17):
                t=rational(t);f=arb(0);d=arb(0);h=arb(0)
                for k,(a,b,c) in zip(scene.kernels,coef):
                    q=(a*t+b)*t+c;g=arb(rational(k.weight))*(-arb(q)/2).exp();dot=a*t+b/2
                    f+=g;d-=arb(dot)*g;h+=arb(dot*dot-a)*g
                for key,value in [('f',f),('first',d),('second',h)]:
                    assert arb(bound[key][0])<=value and value<=arb(bound[key][1]),(key,t,bound,value)
                assert f<=arb(bound['local_upper'][0])


def test_known_hard_full_sum_miss_and_arb_third_path():
    from experiments.gaussian_geometry.generators import generate
    scene=generate('gaussian_sum',20261105,7,.01);s=np.random.default_rng(20261112).random(3);x=(np.array([23,38,45])+.5)/64
    r=MaximumOracle(scene).segment(s,x,proof=True);assert r['status']=='MISS'
    assert CertificateChecker(scene).check(r['certificate'],s,x)['accepted']
    assert FieldOracle(scene,Limits(max_nodes=8192,seconds=10)).segment(s,x).status=='MISS'


def test_no_ready_authority_or_change_to_original_baseline():
    assert not MaximumOracle.ready_capable and not CertificateChecker.ready_capable and not extension().ready_capable
    scene=mixture();before=NativeOracle(scene,independent=True).segment(SOURCE,RECEIVER)
    MaximumOracle(scene).segment(SOURCE,RECEIVER,proof=True)
    after=NativeOracle(scene,independent=True).segment(SOURCE,RECEIVER)
    assert before==after


def test_stationary_evidence_is_recomputed_not_trusted():
    scene=mixture();proof=MaximumOracle(scene,method=2).segment(SOURCE,RECEIVER,proof=True)['certificate']
    stationary=[i for i,p in enumerate(proof['partition']) if p['op']=='stationary'];assert stationary
    i=stationary[0];checker=CertificateChecker(scene)
    for what in ('empty','narrow','operation'):
        p=copy.deepcopy(proof);r=p['partition'][i]
        if what=='empty':r['empty']=not r['empty']
        elif what=='narrow':r['jlo']=r['jhi']=r['lo']
        else:r['inner_op']='increasing'
        assert not checker.check(p,SOURCE,RECEIVER)['accepted']


@pytest.mark.parametrize('seed',[1201,1202,1203,1204])
def test_small_random_continuous_problems_against_unchanged_renderer(seed):
    rng=np.random.default_rng(seed);kernels=[]
    for _ in range(5):
        m=np.diag(rng.uniform(.04,.18,3));m[0,1]=rng.uniform(-.01,.01)
        kernels.append(Kernel(rng.uniform(.25,.75,3),m,rng.uniform(.3,.8)))
    scene=KernelScene('gaussian_sum',kernels,Threshold('binary64',.8));reference=NativeOracle(scene,independent=True)
    engines=[MaximumOracle(scene,method=m) for m in (1,2,3)];checker=CertificateChecker(scene)
    for _ in range(12):
        s,x=rng.uniform(0,1,(2,3));expected=reference.segment(s,x).status;assert expected in ('MISS','HIT')
        for e in engines:
            r=e.segment(s,x,proof=True);assert r['status']==expected
            if expected=='MISS':assert checker.check(r['certificate'],s,x)['accepted']


def bridge():
    from experiments.gaussian_geometry.b2_max_cert.field_bridge import extension
    directory=Path(os.environ.get('RIND_B2_MAX_FIELD_BUILD','outputs/gaussian-b2-max-cert-v1/field-hook-v3'))
    if not (directory/'identity.json').exists():pytest.skip('Build the isolated field hook')
    return extension(directory)


@pytest.mark.parametrize('checked',[False,True])
def test_scalar_only_field_hook_matches_original_both_fields(checked):
    from experiments.gaussian_geometry.b2_max_cert.api import parameters
    from experiments.gaussian_geometry.b2_regions.api import RegionOracle
    scene=KernelScene('gaussian_sum',[Kernel([x,.5,.5],np.eye(3)*.1,.6) for x in (.42,.58)],Threshold('binary64',.7));s=[.05,.2,.5]
    m=bridge();engine=m.Engine(*parameters(scene),False);cfg=m.Settings();cfg.cover_depth=3;cfg.cover_nodes=4096;cfg.check_miss_proof=checked;cfg.mode=5
    options=m.Options();new=engine.field(s,8,20,options,cfg)
    old=RegionOracle(scene,independent=True,cover_depth=3,cover_nodes=4096,mode=5).field(s,8,20)
    assert new['complete'] and old['complete'];assert new['work'].get('maximum_calls',0)>0
    for key in ('response','solid','valid'):assert np.array_equal(new[key],old[key])
    assert new['work'].get('MISS_proof_rejections',0)==0
    if checked:assert new['work'].get('MISS_proof_checks',0)>0


def test_field_hook_disabled_and_exact_unknown_fail_closed():
    from experiments.gaussian_geometry.b2_max_cert.api import parameters
    from experiments.gaussian_geometry.b2_regions.api import RegionOracle
    m=bridge();scene=mixture();cfg=m.Settings();cfg.maximum_method=0;cfg.cover_depth=3;options=m.Options()
    r=m.Engine(*parameters(scene),False).field(SOURCE,8,20,options,cfg)
    old=RegionOracle(scene,cover_depth=3).field(SOURCE,8,20)
    assert r['complete'] and all(np.array_equal(r[k],old[k]) for k in ('response','solid','valid'))
    assert r['work'].get('maximum_calls',0)==0
    scene=KernelScene('gaussian_sum',[Kernel([x,.5,.5],np.eye(3)*.125,.5) for x in (.5,.625)],Threshold('exp',-.25))
    cfg.maximum_method=3;cfg.check_miss_proof=True;options.max_nodes=128;options.query_seconds=.1
    r=m.Engine(*parameters(scene),False).field([.5625,.5625,.0625],8,20,options,cfg)
    assert not r['complete'] and not r['valid'][7,4,4]


def test_unaligned_tile_reuses_original_geometry_not_sampling_as_truth():
    from experiments.gaussian_geometry.b2_max_cert.api import parameters
    from experiments.gaussian_geometry.b2_regions.api import RegionOracle
    scene=mixture();m=bridge();cfg=m.Settings();cfg.cover_depth=3;cfg.check_miss_proof=True;o=m.Options()
    new=m.Engine(*parameters(scene),False).tile(SOURCE,32,[1,3,7],8,20,o,cfg)
    old=RegionOracle(scene,independent=True,cover_depth=3).tile(SOURCE,32,[1,3,7],8,20)
    assert new['complete'] and all(np.array_equal(new[k],old[k]) for k in ('response','solid','valid'))


def test_failed_or_corrupt_proof_checkpoint_not_silently_resumed(tmp_path):
    import hashlib
    from experiments.gaussian_geometry.b2_max_cert.benchmark import validate_checkpoint
    row={'ray_id':'synthetic-test','case':'fixture','expected':'MISS'};p=tmp_path/'proof.json';p.write_bytes(b'test fixture')
    saved={**row,'method':'D_proof','repetition':0,'status':'MISS','checker':{'accepted':True},'proof_SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}
    validate_checkpoint(saved,row,'D_proof',0,p)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError):validate_checkpoint(saved,row,'D_proof',0,p)
    saved['checker']['accepted']=False
    with pytest.raises(ValueError):validate_checkpoint(saved,row,'D_proof',0,p)


def test_optional_lean_proposals_do_not_remove_closed_endpoint_coverage():
    scene=mixture();r=MaximumOracle(scene,endpoint_probes=False).segment(SOURCE,RECEIVER,proof=True)
    assert r['status']=='MISS';p=r['certificate'];assert CertificateChecker(scene).check(p,SOURCE,RECEIVER)['accepted']
    assert min(float.fromhex(q['lo']) for q in p['partition'])==0
    assert max(float.fromhex(q['hi']) for q in p['partition'])==1
    contact=KernelScene('gaussian_sum',[Kernel([.5]*3,np.eye(3),.5) for _ in range(2)],Threshold('exp',-.5))
    r=MaximumOracle(contact,endpoint_probes=False,max_nodes=32,max_depth=16,fallback=True).segment([1.5,.5,.5],[2.,.5,.5],proof=True)
    assert r['status']=='HIT' and r['certificate'] is None
