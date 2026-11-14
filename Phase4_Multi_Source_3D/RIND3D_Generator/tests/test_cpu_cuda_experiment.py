"""Scientific regression tests for the isolated optional native/GPU prototype."""
from fractions import Fraction as F
import math
import os
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
native=pytest.importorskip('experiments.cpu_cuda._efficient_native')
from experiments.cpu_cuda.cpu import GeometryGroup
from experiments.cpu_cuda.gpu import enclose,encode
from experiments.cpu_cuda.benchmark import frozen,Control
from experiments.shadow_volumes.cases import make_case,SOURCES
from rind3d.geometry import centres


@pytest.fixture(scope='module')
def group(): return GeometryGroup(make_case('box')[0])


def encloses(lo,q,hi):
    return (lo==-math.inf or F(lo)<=q) and (hi==math.inf or q<=F(hi))


def test_nextafter_bits_match_all_ieee_cases():
    values=np.r_[[-math.inf,math.inf,-0.,0.,np.nan,np.finfo(float).tiny,-np.finfo(float).tiny,
                  np.nextafter(0.,1.),-np.nextafter(0.,1.),np.finfo(float).max],
                  np.random.default_rng(4).integers(0,2**64,4096,dtype=np.uint64).view(np.float64)]
    for x in values:
        for sign in (False,True):
            y=native.step(float(x),sign); z=math.nextafter(float(x),math.inf if sign else -math.inf)
            assert (math.isnan(y) and math.isnan(z)) or np.float64(y).tobytes()==np.float64(z).tobytes()


def test_cpu_outward_arithmetic_subnormal_overflow():
    rng=np.random.default_rng(11)
    values=[0.,-0.,1.,-1.,np.nextafter(0.,1.),-np.nextafter(0.,1.),2.**-1022,2.**1023]
    pairs=[(a,b) for a in values for b in values]
    pairs += [(math.ldexp(float(rng.uniform(-1,1)),int(rng.integers(-1070,1023))),
               math.ldexp(float(rng.uniform(-1,1)),int(rng.integers(-1070,1023)))) for _ in range(1000)]
    for a,b in pairs:
        for op,q in [(0,F(a)+F(b)),(1,F(a)*F(b))]+([] if b==0 else [(2,F(a)/F(b))]):
            lo,hi=native.interval(a,b,op); assert encloses(lo,q,hi),(a,b,op,lo,hi)


def test_rational_coefficients_enclosed_without_precision_loss():
    for q in [F(0),F(1,3),F(-2,7),F(1,2**1100),F(-1,2**1100),F(2**1023)]:
        assert encloses(*[enclose(q)[0],q,enclose(q)[1]])


@pytest.mark.parametrize('name',['box','rotated_box','thin_slab','prism','wedge','poly16','poly32','multiscale12'])
def test_cached_convex_certificates_vs_unchanged_reference(name):
    g=GeometryGroup(make_case(name)[0]); a=g.bind(SOURCES[0]); ref=Control(g,SOURCES[0],'C_shadow')
    for origin,side,n in [((0,0,0),32,32),((7,11,3),33,64)]:
        y,m,_=a.render(SOURCES[0],origin,side,n)
        yy,mm,_=ref.native.points(centres(origin,side,n),True)
        assert np.array_equal(y.ravel(),yy) and np.array_equal(m.ravel(),mm)
    assert g._oracle is None  # no eager CGAL init in the complete scoped path


def test_closed_contacts_and_union_gap(group):
    source=np.array([.125,.5,.5]); a=group.bind(source)
    p=np.array([[.375,.5,.5],[.625,.75,.5],[.625,.75,.75],[.625,np.nextafter(.75,1.),.5],[.625,np.nextafter(.75,0.),.5]])
    y,m,_=a.native.points(p)
    assert y.tolist()==[0,0,0,1,0] and m[0]==1
    with pytest.raises(ValueError):group.bind([.375,.5,.5])
    base=group.canonical
    r=base.reference.segment_intersect(np.repeat(source[None],len(p),axis=0),p)
    assert r.valid.all() and np.array_equal(y,r.value==0)


def test_exact_boundedness_gate_and_publication():
    with pytest.raises(ValueError,match='[Uu]nbounded'):native.Geometry([[['1','0','0','-1']]])
    g=GeometryGroup(make_case('poly8')[0]); assert not g.capability['ready_capable']


def test_immutable_source_and_evidence_keys(group):
    s=SOURCES[0].copy();a=group.bind(s);s[:]=SOURCES[1]
    assert np.array_equal(a.source,SOURCES[0]) and not a.source.flags.writeable
    assert a.source_id!=group.bind(SOURCES[1]).source_id
    with pytest.raises(ValueError):a.render(s,(0,0,0),8,32)


def test_evidence_recomputes_physics_rejects_mismatch_and_incomplete(group,tmp_path):
    from rind3d.pack import RINDPackWriter
    from rind3d.stream import scene_identity
    from experiments.cpu_cuda.evidence import build_evidence,BrickEvidence
    from rind3d.windows import Window
    a=group.bind(SOURCES[0]);other=Control(group,SOURCES[0],'B_native')
    n=128; origin=(0,0,0);y,m,_=a.render(SOURCES[0],origin,64,n)
    for name,v in [('response',y),('solid',m)]:
        w=RINDPackWriter(tmp_path/(name+'.rindb'),(n,)*3,identity=scene_identity(a,a.source),policy=a.policy_hash)
        raw=np.packbits(v,bitorder='little').tobytes()
        for i in range(8):w.write(i,raw if i==0 else bytes(32768))
        w.finish()
    windows=[Window((0,0,0),64)]
    proof,r=build_evidence(tmp_path,windows,a,other,primary_backend='new',independent_backend='old')
    assert all(stage['freshly_rendered_bricks']==1 for stage in r['stages'])
    assert proof.verify_window(windows[0],a,a.source)['covered_voxels']==64**3
    with pytest.raises(ValueError,match='coverage'):proof.verify_window(Window((1,0,0),64),a,a.source)
    with pytest.raises(ValueError,match='identity'):proof.verify_window(windows[0],a,SOURCES[1])
    with pytest.raises(ValueError,match='Separate'):build_evidence(tmp_path,windows,a,other,primary_backend='same',independent_backend='same')
    with pytest.raises(AssertionError,match='mismatch'):
        build_evidence(tmp_path,[Window((64,64,64),64)],a,other,primary_backend='new',independent_backend='old')
    proof.validate_files(tmp_path)
    with (tmp_path/'response.rindb').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='content changed'):proof.validate_files(tmp_path)


def test_experiment_cache_never_reuses_another_source_or_corrupt_file(group,tmp_path):
    from experiments.cpu_cuda.benchmark import field
    a=group.bind(SOURCES[0]);b=group.bind(SOURCES[1]);path=tmp_path/'cached'
    first=field(a,path,32,'CPU_clip')
    assert first['complete'] and field(a,path,32,'CPU_clip')['source_id']==a.source_id
    for active,n,method in [(b,32,'CPU_clip'),(a,64,'CPU_clip'),(a,32,'CPU_projected')]:
        with pytest.raises(ValueError,match='identity mismatch'):field(active,path,n,method)
    with (path/'response.rindb').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError):field(a,path,32,'CPU_clip')


def test_gpu_budget_exhaustion_after_blocking_batch_never_seals_field(group,tmp_path):
    from types import SimpleNamespace
    import time
    from experiments.cpu_cuda.benchmark import gpu_stream
    cpu=group.bind(SOURCES[0])
    class SlowBatch:
        source=cpu.source
        runtime=SimpleNamespace(timings={'kernel_seconds':0.},memory=lambda:{'free_bytes':1_000_000,'total_bytes':1_000_000})
        group=SimpleNamespace(batch=32)
        def __getattr__(self,name):return getattr(cpu,name)
        def packed_batch(self,*args):
            time.sleep(.02)
            return [(bytes(32768),bytes(32768))]
    result=gpu_stream(SlowBatch(),tmp_path,32,.01)
    assert not result['complete'] and result['error_type']=='BudgetExhausted'
    assert result['bricks']==0 and not (tmp_path/'response.rindb').exists()


def gpu_required():
    if os.environ.get('RIND_TEST_CUDA')!='1':pytest.skip('Explicit real-GPU opt-in required: RIND_TEST_CUDA=1 outside sandbox')


def test_real_cuda_directed_arithmetic():
    gpu_required()
    from experiments.cpu_cuda.cuda_runtime import Runtime
    rng=np.random.default_rng(882)
    a=np.r_[0.,-0.,np.nextafter(0.,1.),-np.nextafter(0.,1.),2.**-1022,2.**1023,
            np.ldexp(rng.uniform(-1,1,4096),rng.integers(-1070,1023,4096))]
    b=np.r_[0.,-1.,.5,.5,2.**-100,2.,np.ldexp(rng.uniform(-1,1,4096),rng.integers(-1070,1023,4096))]
    with Runtime() as r:
        pa=r.upload(a);pb=r.upload(b);out=r.alloc(len(a)*4*8)
        r.launch('arithmetic',((len(a)+255)//256,1,1),(256,1,1),[pa,pb,len(a),out])
        values=r.download(out,(len(a),4),np.float64)
        for i,(x,y) in enumerate(zip(a,b)):
            assert encloses(values[i,0],F(x)+F(y),values[i,1])
            assert encloses(values[i,2],F(x)*F(y),values[i,3])
        assert r.timings['kernel_seconds']>0


def test_real_cuda_unknown_uses_canonical_and_unaligned_regions(group):
    gpu_required()
    from experiments.cpu_cuda.cuda_runtime import Runtime
    from experiments.cpu_cuda.gpu import GPUGroup
    with Runtime() as r:
        g=GPUGroup(group,r);a=g.bind([.0625,.5,.5])
        try:
            # Slope 2/5 has a non-dyadic coefficient enclosure. The exact
            # silhouette contacts cannot all be decided by binary64 intervals.
            p=np.array([[.375,.5,.5],[.6875,.75,.5],[.6875,.75,.75],[.6875,np.nextafter(.75,1.),.5],[.6875,np.nextafter(.75,0.),.5]])
            yy,mm=a.point_labels(p,resolve=False)
            assert np.any(yy<0)  # exact contacts MUST not be epsilon-labeled
            y,m=a.point_labels(p);ry,rm,_=a.cpu.native.points(p)
            assert np.array_equal(y,ry) and np.array_equal(m,rm)
            for origin,side,n in [((7,11,3),32,64),((0,0,0),64,64)]:
                pair=a.packed_batch([origin],side,n)[0]
                cy,cm,_=a.cpu.render(a.source,origin,side,n)
                for raw,reference in zip(pair,(cy,cm)):
                    v=np.unpackbits(np.frombuffer(raw,dtype=np.uint8),bitorder='little').reshape((64,)*3)
                    assert np.array_equal(v[:side,:side,:side],reference)
            assert a.work['canonical_fallback_rays']>0
        finally:a.close();g.close()


def test_real_cuda_certified_native_fallback_is_explicit_and_independently_checked():
    gpu_required()
    from experiments.cpu_cuda.cuda_runtime import Runtime
    from experiments.cpu_cuda.gpu import GPUGroup
    g=GeometryGroup(make_case('box')[0]);source=np.array([.0625,.5,.5])
    ref=Control(g,source,'C_shadow');wrong=Control(g,SOURCES[1],'C_shadow')
    with Runtime() as r:
        group=GPUGroup(g,r);a=group.bind(source,fallback_backend='certified_native')
        try:
            with pytest.raises(ValueError,match='identity'):a.attach_independent_reference(wrong)
            a.attach_independent_reference(ref)
            p=np.array([[.6875,.75,.5],[.6875,.75,.75],[.6875,np.nextafter(.75,1.),.5]])
            y,m=a.point_labels(p)
            assert y.tolist()==[0,0,1] and not m.any()
            assert a.work['native_fallback_rays']>0
            assert a.work['independent_fallback_points_checked']==a.work['native_fallback_rays']
            assert g._oracle is None and not a.report()['production_ready']
            # Original default remains the original canonical fallback.
            b=group.bind(source)
            try:assert b.fallback_backend=='canonical_cgal'
            finally:b.close()
        finally:a.close();group.close()
