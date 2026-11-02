"""Production paths, independent physical replay and publication fail-closed."""
import copy,json,os
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
import pytest
from rind.contracts import make_leaf,Geometry,RootKind
from rind.storage import CanonicalBundle
from rind3d import _convex
from rind3d.cli import load_config
from rind3d.geometry import Unsupported,Aborted,BudgetExhausted
from rind3d.profiles import FULL,LITE,profile_for,validate_manifest_profile
from rind3d.lite_pipeline import manifest,generate_lite
from rind3d.lite_generators import FAMILIES,generate
from rind3d.lite_acceptance import geometry_acceptance,sample_source,verify_geometry_proof,verify_clearance,DRIVER_RADIUS
from rind3d.lite_certification import small_field,cuda_field
from rind3d.rendering.context import GeometryContext
from rind3d.rendering.backends import BackendFactory
from rind3d.rendering.cuda_runtime import CUDAUnavailable
from rind3d.stream import stream_response
from rind3d.pack import RINDPack,atomic_json
from rind3d.windows import Window
from rind3d.physical_evidence import verify_physical_bricks,load_evidence

def box(center=(.5,.5,.5),half=(.125,.125,.125),matrix=None):
    b=CanonicalBundle(3);leaf=make_leaf('affine_box_v1',3,center=center,matrix=np.diag(half) if matrix is None else matrix)
    b.add_leaf(leaf);b.add_geometry(Geometry(3,RootKind.REF,leaf.content_hash));return b

def config(n=128):
    c=load_config('configs/lite-v0.1.yaml');c.update(resolution=n,build_information_index=False);return c

def test_profiles_preserve_full_and_versioned_lite():
    full=load_config('configs/pilot-extended.yaml');lite=load_config('configs/lite-v0.1.yaml')
    assert profile_for(full).name==FULL and profile_for(full).quotas=={64:1365,128:1365,256:1366}
    assert profile_for(lite).name==LITE and profile_for(lite).quotas=={64:85,128:85,256:86}
    m=manifest(lite);assert validate_manifest_profile(m).ready_state=='READY_LITE'
    for k,v in [('production_windows',4096),('max_candidates',64),('split_group','scene_id'),('window_sides',[32,64,128,256])]:
        bad=copy.deepcopy(m);bad[k]=v
        with pytest.raises(ValueError):validate_manifest_profile(bad)

def test_config_json_roundtrip_preserves_existing_hashes():
    import hashlib
    from rind3d.pilot import config_hash
    from rind3d.cli import validate_config
    for path in ('configs/lite-v0.1.yaml','configs/pilot-extended.yaml'):
        cfg=load_config(path);old_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        restored=json.loads(json.dumps(cfg))
        assert config_hash(cfg)==config_hash(restored)==old_hash
        assert validate_config(restored)==cfg

@pytest.mark.parametrize('family',FAMILIES)
def test_random_families_exact_acceptance_and_independent_complete_fields(family):
    i=FAMILIES.index(family);b,r=generate(1209,i,family);other,_=generate(1209,i,family)
    assert b.geometries[0].content_hash==other.geometries[0].content_hash and not r['proposal_failures']
    ctx=GeometryContext(b);report=geometry_acceptance(ctx,family,1209,i);p,q=sample_source(ctx,1209,i)
    assert report['passed'] and verify_geometry_proof(b,report['geometry_proof']) and verify_clearance(ctx,p,q['clearance'])
    a,m=small_field(ctx.bind(p),32);z,n=small_field(ctx.bind(p,independent=True),32)
    assert np.array_equal(a,z) and np.array_equal(m,n) and not np.any(a&m)

def test_interior_world_duals_and_clearance_cannot_be_forged():
    ctx=GeometryContext(box());proof=ctx.geometry_proof();assert verify_geometry_proof(ctx.bundle,proof)
    for mutate in ('interior','bound','gap'):
        bad=copy.deepcopy(proof)
        if mutate=='interior':bad['components'][0]['interior_radius']='1000'
        if mutate=='bound':bad['components'][0]['bound_duals'][0]['weights'][0]='0'
        if mutate=='gap':bad['components'][0]['strictly_inside_world']=False
        with pytest.raises(ValueError):verify_geometry_proof(ctx.bundle,bad)
    for b in (box((.98,.5,.5)),box((.875,.5,.5))):
        c=GeometryContext(b)
        with pytest.raises(ValueError):verify_geometry_proof(b,c.geometry_proof())
    p=np.array([.1,.1,.1]);q=ctx.clearance(p,DRIVER_RADIUS);assert verify_clearance(ctx,p,q)
    with pytest.raises(ValueError):verify_clearance(ctx,[.5]*3,q)
    with pytest.raises(ValueError,match='Unbounded'):_convex.Geometry([[['1','0','0','-1']]])

def test_production_independent_closed_tangent_endpoint_thin():
    ctx=GeometryContext(box());p=np.array([.0625,.5,.5]);a=ctx.bind(p);b=ctx.bind(p,independent=True)
    points=np.array([[.375,.5,.5],[.6875,.75,.5],[.6875,.75,.75],[.6875,np.nextafter(.75,1),.5]])
    assert a.response(p,points).tolist()==[False,False,False,True]
    assert np.array_equal(a.response(p,points),b.response(p,points))
    original=ctx.canonical;v=original._values(original.reference.segment_intersect(np.repeat(p[None],len(points),axis=0),points))
    assert np.array_equal(v==0,a.response(p,points))
    thin=GeometryContext(box(half=(.2,.2,2**-40)));verify_geometry_proof(thin.bundle,thin.geometry_proof())
    with pytest.raises(ValueError):thin.bind([.5,.5,.5])

def test_missing_cuda_auto_cpu_and_explicit_failure(monkeypatch):
    from rind3d.rendering import cuda_runtime
    def unavailable():raise CUDAUnavailable('injected missing hardware')
    monkeypatch.setattr(cuda_runtime,'Runtime',unavailable)
    with BackendFactory() as f:
        a=f.source(box(),[.1]*3,'auto');assert a.backend=='cpu_native_certified'
        assert 'unavailable' in a.selection_reason
        with pytest.raises(CUDAUnavailable):f.source(box(),[.1]*3,'cuda_certified')
        b=f.source(box(),[.1]*3,'cpu_reference');assert b.backend=='cpu_reference'
        assert np.array_equal(small_field(a,32)[0],small_field(b,32)[0])

def test_unsupported_function_uses_original_restriction():
    b=CanonicalBundle(3);leaf=make_leaf('cosine_field_v1',3,amplitude=[1.],frequency=[[3.,0.,0.]],phase=[0.],threshold=.5)
    b.add_leaf(leaf);b.add_geometry(Geometry(3,RootKind.REF,leaf.content_hash))
    with BackendFactory() as f:
        item=f.source(b,[.9,.1,.1],'auto');assert item.backend=='cpu_reference' and not item.capability['ready_capable']
    with pytest.raises(Unsupported):GeometryContext(b)

@pytest.mark.parametrize('backend',['cpu_native_certified','cuda_certified'])
def test_interrupted_stream_resume_and_corruption(backend,tmp_path):
    if backend.startswith('cuda') and os.getenv('RIND_TEST_CUDA')!='1':pytest.skip('Real GPU opt-in')
    cfg=config();ctx=GeometryContext(box());source=np.array([.1,.3,.6])
    with BackendFactory() as f:
        a=f.source(ctx.bundle,source,backend);cfg['stop_after_bricks']=2
        with pytest.raises(Aborted):stream_response(ctx,source,tmp_path,cfg,renderer=a)
        assert not (tmp_path/'response.rindb').exists()
        cfg.pop('stop_after_bricks');result=stream_response(ctx,source,tmp_path,cfg,resume=True,renderer=a)
        assert result['complete'] and result['bricks_computed_this_run']==6
        a.close()
        with RINDPack(tmp_path/'response.rindb',verify_file=True) as r:
            expected=small_field(ctx.bind(source,independent=True),128)[0]
            assert np.array_equal(r.read_window(0,0,0,128),expected)
        a=f.source(ctx.bundle,source,backend)
        with pytest.raises(ValueError,match='source mismatch'):stream_response(ctx,[.2,.3,.6],tmp_path,cfg,resume=True,renderer=a)
        a.close()
        wrong=f.source(ctx.bundle,[.2,.3,.6],backend)
        with pytest.raises(ValueError,match='identity'):stream_response(ctx,wrong.source,tmp_path,cfg,resume=True,renderer=wrong)
        wrong.close()
    with (tmp_path/'response.rindb').open('ab') as file:file.write(b'corruption')
    with pytest.raises(ValueError):RINDPack(tmp_path/'response.rindb',verify_file=True)

def test_shared_physical_evidence_exact_coverage_and_staleness(tmp_path):
    ctx=GeometryContext(box());source=np.array([.1,.3,.6]);a=ctx.bind(source)
    stream_response(ctx,source,tmp_path,config(),renderer=a)
    windows=[Window((1,1,1),64),Window((2,3,4),64)]
    proof,report=verify_physical_bricks(tmp_path,windows,ctx,source)
    assert len(proof.covered)==8 and all(s['rendered_bricks']==8 for s in report['stages'])
    assert proof.verify_window(windows[0],ctx,source)['covered_voxels']==64**3
    with pytest.raises(ValueError):proof.verify_window(windows[0],ctx,[.2,.3,.6])
    bad=copy.deepcopy(report);bad['bricks'].pop();atomic_json(tmp_path/'bad.json',bad)
    with pytest.raises(ValueError,match='coverage'):load_evidence(tmp_path/'bad.json',report['key'],windows)
    bad=copy.deepcopy(report);bad['key']['response_sha256']='0'*64;atomic_json(tmp_path/'bad.json',bad)
    with pytest.raises(ValueError,match='Stale'):load_evidence(tmp_path/'bad.json',report['key'],windows)

def test_budget_after_gpu_batch_cannot_commit(tmp_path):
    import time
    from types import SimpleNamespace
    ctx=GeometryContext(box());a=ctx.bind([.1]*3)
    class Slow:
        backend='cuda_certified';device=SimpleNamespace(batch=32)
        def __getattr__(self,name):return getattr(a,name)
        def packed_batch(self,*args,**kw):time.sleep(.02);return [(bytes(32768),bytes(32768))]
    cfg=config(32);cfg['generation_seconds']=.005
    with pytest.raises(BudgetExhausted):stream_response(ctx,a.source,tmp_path,cfg,renderer=Slow())
    assert not (tmp_path/'response.rindb').exists()

def test_real_nontrivial_equivalent_source_is_fully_independently_verified():
    from rind3d.lite_search import discover,CandidateVerifier
    ctx=GeometryContext(box());truth=np.array([.0625,.5,.5]);origin=(768,768,480)
    observed,mask,_=ctx.bind(truth).render(origin,64,1024)
    other,solid,_=ctx.bind(truth,independent=True).render(origin,64,1024)
    assert np.array_equal(observed,other) and np.array_equal(mask,solid)
    lit=np.argwhere(observed&~mask)[0][::-1]+origin;shadow=np.argwhere(~observed&~mask)[0][::-1]+origin
    window=Window(origin,64,witnesses=(tuple(lit),tuple(shadow)))
    class IndependentlyRenderedTruth:
        def verify_window(self,w,c,p):
            assert w==window and c is ctx and np.array_equal(p,truth)
            return {'covered_voxels':64**3,'status':'grid_exact','independent_status':'grid_exact'}
    pool=np.array([[.21875,.5625,.5],[.125,.1,.5]])
    cfg=load_config('configs/lite-v0.1.yaml');cfg.update(window_search_seconds=2)
    points,proofs,report=discover(ctx,truth,window,observed,mask,pool,cfg,0,0,IndependentlyRenderedTruth(),CandidateVerifier(ctx))
    assert len(points)==2 and np.array_equal(points[1],pool[0]) and report['independent_checks']==1
    assert all(p['covered_voxels']==64**3 and p['independent_status']=='grid_exact' for p in proofs)

@pytest.mark.parametrize('name',['box','thin','near_coplanar'])
def test_cuda_production_complete_32_64_128(name):
    if os.getenv('RIND_TEST_CUDA')!='1':pytest.skip('Real GPU opt-in')
    b=box() if name=='box' else box(half=(.2,.2,2**-40)) if name=='thin' else box(matrix=np.array([[.125,2**-35,0],[0,.125,0],[0,0,.125]]))
    ctx=GeometryContext(b);source=np.array([.1,.3,.6])
    with BackendFactory() as f:
        a=f.source(b,source,'cuda_certified')
        try:
            for n in (32,64,128):
                y,m=cuda_field(a,n);z,k=small_field(ctx.bind(source,independent=True),n)
                assert np.array_equal(y,z) and np.array_equal(m,k)
        finally:a.close()

def test_no_publication_without_geometry_fixture_gate(tmp_path):
    c=load_config('configs/lite-v0.1.yaml')
    with pytest.raises(FileNotFoundError):generate_lite(c,tmp_path/'closed',1,certificate_path=tmp_path/'absent')
    with pytest.raises(ValueError,match='pilot'):generate_lite(c,tmp_path/'hundred',100)
    with pytest.raises(ValueError,match='mass'):generate_lite(c,tmp_path/'mass',5000)
    assert not (tmp_path/'mass').exists()

def test_certification_cli_from_clean_output_directory(tmp_path,monkeypatch):
    from rind3d import lite_certification
    from rind3d.cli import main
    target=tmp_path/'new-checkout'/'outputs'/'scientific-certification.json'
    def run(path,*,cuda):
        assert Path(path).parent.is_dir() and not cuda
        atomic_json(path,{'complete':True,'seconds':0.})
        return {'complete':True,'seconds':0.}
    monkeypatch.setattr(lite_certification,'certify',run)
    assert main(['certify-lite','--cpu-only','--output',str(target)])==0
    assert json.loads(target.read_text())['complete']
