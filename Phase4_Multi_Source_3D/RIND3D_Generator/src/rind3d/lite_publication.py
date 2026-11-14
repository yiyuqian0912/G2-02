"""Fail-closed Lite publication checks, also used by the ordinary loader.

Persisted evidence is an audit record of physical work, not a substitute for
that work. ``physical=True`` repeats both implementations and all alternatives.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from .pack import RINDPack, sha256_file
from .profiles import LITE, profile_for
from .pipeline import EVIDENCE, validate_records
from .geometry import load_geometry, Budget, POLICY
from .windows import Window, excludes, placement_summary
from .window_contract import WINDOW_PROTOCOL
from .rendering.context import GeometryContext
from .lite_acceptance import PROTOCOL, verify_geometry_proof, verify_clearance
from .lite_certification import validate_certificate
from .physical_evidence import identity, load_evidence, verify_physical_bricks

REQUIRED = {'geometry.npz','driver.f64','response.rindb','solid.rindb',
            'window_records.bin','candidate_bank.bin','candidate_ids.bin',
            'candidate_offsets.bin','candidate_counts.bin','candidate_evidence.bin',
            'window_witnesses.bin','search_records.parquet','information.rindi',
            'window_selection.parquet','uniform_control.parquet',
            'geometry_acceptance.private.json','sources.private.json',
            'physical_evidence.private.json','scientific_certification.json',
            'recipe.private.json','candidate_search.private.json'}

def verify_lite_scene(path, *, full_index=False, physical=False):
    path=Path(path);meta=json.loads((path/'verification.json').read_text())
    if profile_for(meta).name!=LITE or meta.get('status')!='READY_LITE' or meta.get('profile_version')!='0.1':
        raise ValueError('Scene is not READY_LITE v0.1')
    if meta.get('contract_version')!=2 or meta.get('window_protocol')!=WINDOW_PROTOCOL or meta.get('equivalence_status')!='grid_exact' or meta.get('continuous_certified'):
        raise ValueError('Unsupported Lite scientific contract')
    if not REQUIRED<=meta.get('hashes',{}).keys():raise ValueError('Missing Lite scientific files')
    for name,digest in meta['hashes'].items():
        p=path/name
        if p.parent!=path or sha256_file(p)!=digest:raise ValueError('Lite file hash mismatch: '+name)
    certification=validate_certificate(json.loads((path/'scientific_certification.json').read_text()),current=False)
    if meta.get('scientific_identity')!=certification['scientific_identity']['sha256']:raise ValueError('Scene/fixture scientific identity differs')
    records,counts,bank,ids=validate_records(path,meta)
    geometry=load_geometry(path/'geometry.npz');context=GeometryContext(geometry,meta['numerical_policy'])
    if geometry.scenes or context.geometry_id!=meta.get('geometry_id') or context.geometry_id!=meta.get('geometry_hash'):raise ValueError('Geometry/driver privacy or identity violation')
    acceptance=json.loads((path/'geometry_acceptance.private.json').read_text())
    if acceptance.get('protocol')!=PROTOCOL or not acceptance.get('passed') or acceptance.get('family') not in certification['families'] or meta['families']!=[acceptance['family']]:raise ValueError('Uncertified family/geometry acceptance')
    verify_geometry_proof(geometry,acceptance['geometry_proof'])
    sources=json.loads((path/'sources.private.json').read_text())
    if len(sources.get('bank',[]))!=len(bank):raise ValueError('Source clearance proof table differs from pool')
    truth=np.fromfile(path/'driver.f64',dtype='<f8')
    if truth.shape!=(3,) or not np.isfinite(truth).all():raise ValueError('Invalid Lite true source')
    if not np.array_equal(truth,np.asarray(sources['truth']['position'])):raise ValueError('True source proof mismatch')
    verify_clearance(context,truth,sources['truth']['clearance'])
    from rind.contracts import make_scene,make_physics
    canonical_scene=make_scene(context.geometry.content_hash,3,make_physics(0),truth[None])
    if canonical_scene.content_hash.hex()!=meta['scene_id'] or canonical_scene.content_hash.hex()!=sources['truth']['canonical_scene_id'] or hashlib.sha256(context.geometry.content_hash+truth.astype('<f8').tobytes()).hexdigest()!=meta['source_id']:raise ValueError('Canonical single-source scene identity mismatch')
    for p,proof in zip(bank,sources['bank']):verify_clearance(context,p,proof)
    if meta.get('split_group')!=context.geometry_id:raise ValueError('Lite split must group by geometry_id')
    from .rendering.polyhedra import geometry_split
    if meta['split']!=geometry_split(context.geometry_id,meta.get('split_seed',0)):raise ValueError('Geometry split identity mismatch')
    witnesses=np.fromfile(path/'window_witnesses.bin',dtype='<u2')
    if witnesses.size!=len(records)*6:raise ValueError('Window witness shape mismatch')
    witnesses=witnesses.reshape(-1,2,3)
    windows=[Window(tuple(map(int,r[:3])),int(r[3]),witnesses=tuple(tuple(map(int,p)) for p in w)) for r,w in zip(records,witnesses)]
    if placement_summary(windows,1024)['occupied_octants']<2:raise ValueError('Insufficient Lite window spatial coverage')
    raw=(path/'candidate_evidence.bin').read_bytes()
    if len(raw)!=len(windows)*EVIDENCE.size:raise ValueError('Candidate evidence truncated')
    for i,(flags,count,covered,digest) in enumerate(EVIDENCE.iter_unpack(raw)):
        if flags!=3 or count!=counts[i] or covered!=windows[i].size**3 or digest!=hashlib.sha256(bank[ids[i]].astype('<f8').tobytes()).digest():raise ValueError('Incomplete or mismatched candidate physical evidence')
    search=pq.read_table(path/'search_records.parquet').to_pylist()
    if len(search)!=len(windows) or any(r['window_id']!=i or r.get('status')!='COMPLETE' or r.get('actual_candidates')!=int(counts[i]) or not r.get('all_grid_exact') or not r.get('independent_validation_passed') or not r.get('search_termination_reason') for i,r in enumerate(search)):raise ValueError('Lite candidate search record mismatch')
    selection=pq.read_table(path/'window_selection.parquet').to_pylist()
    if len(selection)!=len(windows) or any(row['window_id']!=i or row['origin']!=list(w.origin) or row['side']!=w.size or row['provenance'] not in ('uniform_accepted','guided_informative') for i,(w,row) in enumerate(zip(windows,selection))):raise ValueError('Window selection provenance mismatch')
    from .information import InformationIndex
    from .stream import scene_identity
    with RINDPack(path/'response.rindb',verify_file=True) as y,RINDPack(path/'solid.rindb',verify_file=True) as m:
        recorded_policy=meta['numerical_policy_sha256']
        recorded_identity=hashlib.sha256(context.geometry.content_hash+truth.astype('<f8').tobytes()+bytes.fromhex(recorded_policy)).hexdigest()
        if y.shape!=(1024,)*3 or m.shape!=y.shape or y.policy!=recorded_policy or m.policy!=y.policy or y.identity!=recorded_identity or m.identity!=y.identity:raise ValueError('Complete Lite field identity/policy invalid')
        InformationIndex.load(path/'information.rindi',y,m)
        provenance=meta['generation']['numerical_policy']
        if provenance['physics']!=POLICY or provenance['oracle_limits']!=meta['numerical_policy'] or hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()!=y.policy:raise ValueError('Incomplete physical policy provenance')
        if full_index:y.verify_all();m.verify_all()
        tasks=[]
        for i,w in enumerate(windows):
            if not excludes(w,truth,1024) or not any(np.array_equal(p,truth) for p in bank[ids[i]]):raise ValueError('Source exclusion/truth inclusion violated')
            if meta['candidate_domain'] not in ('world_free','world_outside_window'):raise ValueError('Invalid candidate domain')
            if meta['candidate_domain']=='world_outside_window' and any(not excludes(w,p,1024) for p in bank[ids[i]]):raise ValueError('Candidate inside forbidden window')
            for p,expected in zip(w.witnesses,(True,False)):
                if np.any(np.asarray(p)<w.origin) or np.any(np.asarray(p)>=np.asarray(w.origin)+w.size):raise ValueError('Witness outside half-open window')
                tasks.append((p,expected))
        for p,expected in sorted(tasks,key=lambda t:tuple(v//64 for v in t[0])):
            if m.read_point(*p) or bool(y.read_point(*p))!=expected:raise ValueError('Window lacks two certified FREE response witnesses')
        solid_fraction=sum(r[7] for r in m.index)/1024**3
    evidence_record=json.loads((path/'physical_evidence.private.json').read_text())
    stored_key=evidence_record['key']
    # Reading archived data does not require the producer's compiler/binary on
    # this machine. Validate the recorded numerical identity, never relabel a
    # current physical replay as that historical implementation.
    from types import SimpleNamespace
    recorded=SimpleNamespace(geometry=context.geometry,geometry_id=context.geometry_id,policy_hash=meta['numerical_policy_sha256'])
    expected=identity(recorded,truth,1024,stored_key['primary_backend'],stored_key['independent_backend'],meta['hashes']['response.rindb'],meta['hashes']['solid.rindb'])
    # Bind both recorded algorithms to the fixture-tested implementation build.
    if stored_key['primary_backend']['kind']!='cached-parametric-clipping' or stored_key['independent_backend']['kind']!='exact-projected-shadow':raise ValueError('Unsupported independent physical pair')
    if stored_key['primary_backend']!=meta['physical_backends'][0] or stored_key['independent_backend']!=meta['physical_backends'][1]:raise ValueError('Physical evidence backend identity differs')
    load_evidence(path/'physical_evidence.private.json',expected,windows)
    if physical:
        # Repeat rather than taking the cache hit. Preserve the historical evidence.
        import tempfile,shutil
        with tempfile.TemporaryDirectory(prefix='rind3d-recheck-') as tmp:
            temp=Path(tmp)
            for name in ('response.rindb','solid.rindb','response.rindb.sha256','solid.rindb.sha256'):shutil.copy2(path/name,temp/name)
            verify_physical_bricks(temp,windows,context,truth)
        from .lite_search import CandidateVerifier
        verifier=CandidateVerifier(context)
        with RINDPack(path/'response.rindb') as response:
            for wid,w in enumerate(windows):
                observed=None
                for p in bank[ids[wid]]:
                    if np.array_equal(p,truth):continue
                    if observed is None:observed=response.read_window(*w.origin,w.size)
                    for independent in (False,True):
                        if verifier.verify(p,w,observed,independent=independent,budget=Budget(300,1_000_000_000))['status']!='grid_exact':raise ValueError('Candidate physical rerender mismatch')
    row={'scene_id':meta['scene_id'],'geometry_id':context.geometry_id,'source_id':meta['source_id'],'split':meta['split'],
         'families':meta['families'],'status':'READY_LITE','window_count':256,'candidate_count_min':int(counts.min()),
         'candidate_count_max':int(counts.max()),'solid_fraction':solid_fraction,
         'bytes':sum(p.stat().st_size for p in path.rglob('*') if p.is_file()),'verification_sha256':sha256_file(path/'verification.json')}
    return {'scene_row':row,'window_rows':[{'scene_id':meta['scene_id'],'window_id':i,'size':w.size,'equivalent_count':int(counts[i])} for i,w in enumerate(windows)]}
