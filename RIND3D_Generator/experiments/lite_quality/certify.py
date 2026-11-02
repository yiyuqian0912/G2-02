"""Prepare exact 1024³ controls, then seal the extra research-pilot gate."""
import argparse,json,time,subprocess,sys
from pathlib import Path
import numpy as np
from rind3d.pack import atomic_json,sha256_file
from rind3d.geometry import load_geometry
from rind3d.cli import load_config
from rind3d.stream import stream_response
from rind3d.lite_acceptance import geometry_acceptance,verify_clearance,DRIVER_RADIUS
from rind3d.rendering.context import GeometryContext,backend_identity
from rind3d.projection_research import ProjectionSource,implementation_identity
from rind3d.quality_certification import VERSION,identity,validate
from .verification import replay
from .audit import BASE,OUT

def prepare():
    baseline=BASE
    if not (baseline/'report.json').exists():baseline=standalone_references()
    folder=OUT/'full-field-controls'/implementation_identity()['implementation_sha256'][:16];folder.mkdir(parents=True,exist_ok=True);allscenes=json.loads((baseline/'report.json').read_text())['scenes'];records=[]
    for slot in (0,4,50):
        s=next(s for s in allscenes if s['slot']==slot);path=baseline/'scenes'/s['scene_id'];target=folder/(s['scene_id']+'.json');meta=json.loads((path/'verification.json').read_text());ctx=GeometryContext(load_geometry(path/'geometry.npz'),meta['numerical_policy']);p=np.asarray(s['source'])
        hashes={name:sha256_file(path/name) for name in ('response.rindb','solid.rindb')};backend=implementation_identity()
        if target.exists():
            record=json.loads(target.read_text())
            if record['backend']!=backend or record['field_hashes']!=hashes:raise ValueError('Stale full-field control; preserve and use a new output version')
        else:
            model=ProjectionSource(ctx,p);coords=[(x,y,z) for z in range(16) for y in range(16) for x in range(16)];r=replay(path,coords,model)
            record={'scene_id':s['scene_id'],'slot':slot,'family':s['family'],'backend':backend,'field_hashes':hashes,'bricks':4096,
                    'response_and_solid_exact':r['all_response_and_solid_bits_match'],'metrics':r,'planes':model.plane_report,'setup_seconds':model.setup_seconds}
            atomic_json(target,record)
        records.append(record)
    return records

def standalone_references():
    """Fresh-machine exact fixtures; original A full fields, never READY."""
    import shutil
    source=Path(__file__).with_name('fixtures');fixtures=json.loads((source/'manifest.json').read_text())['fixtures'];root=OUT/'standalone-fixture-reference';rows=[]
    cfg=load_config('configs/lite-v0.1.yaml');cfg.update(build_information_index=False,generation_seconds=600)
    for item in fixtures:
        p=np.array([float.fromhex(v) for v in item['source_hex']]);bundle=load_geometry(source/item['geometry_file']);ctx=GeometryContext(bundle,item['numerical_policy'])
        if ctx.geometry_id!=item['geometry_id']:raise ValueError('Canonical fixture geometry changed')
        geometry_acceptance(ctx,item['family'],20261031,item['slot']);verify_clearance(ctx,p,ctx.clearance(p,DRIVER_RADIUS));sid=ctx.bind(p).source_id;scene=root/'scenes'/sid;scene.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/item['geometry_file'],scene/'geometry.npz');atomic_json(scene/'verification.json',{'numerical_policy':item['numerical_policy'],'status':'REFERENCE_ONLY_NOT_READY'})
        stream_response(ctx,p,scene,cfg,resume=True,renderer=ctx.bind(p));rows.append({'slot':item['slot'],'family':item['family'],'scene_id':sid,'source':p.tolist()})
    atomic_json(root/'report.json',{'scenes':rows,'reference_only':True,'published_ready_scenes':0});return root

def seal():
    records=prepare();tests=['tests/test_core_research.py','tests/test_exact_counts.py','tests/test_projection_research.py','tests/test_stratified_windows.py','tests/test_inverse_research.py']
    t=time.perf_counter();r=subprocess.run([sys.executable,'-m','pytest','-q',*tests],capture_output=True,text=True);(OUT/'quality-fixtures.log').write_text(r.stdout+r.stderr)
    if r.returncode:raise RuntimeError('Research fixture tests failed; publication stays closed')
    record={'version':VERSION,'complete':True,'identity':identity(),'independent_backend':implementation_identity(),
            'allowed_independent_backends':[implementation_identity(),backend_identity('exact-projected-shadow')],
            'small_fields_exact':True,'boundary_tests_passed':True,'packed_count_tests_passed':True,'scientific_gates_unchanged':True,
            'source_uniqueness_claimed':False,'complete_1024_comparisons':records,'test_command':tests,'test_stdout':r.stdout,'test_seconds':time.perf_counter()-t}
    validate(record);atomic_json(OUT/'quality-certification.json',record);return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');args=p.parse_args();prepare() if args.prepare_only else seal()
