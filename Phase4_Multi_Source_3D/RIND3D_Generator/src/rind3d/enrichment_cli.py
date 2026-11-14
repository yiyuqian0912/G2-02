"""Explicit offline enrichment; never invoked by Core publication."""
import json,time
from pathlib import Path
import numpy as np
import yaml
from .geometry import load_geometry
from .rendering.context import GeometryContext
from .pack import RINDPack
from .pipeline import verify_scene
from .windows import Window
from .lite_certification import validate_certificate
from .enrichment_research import search,save_artifact,SharedCandidateVerifier

def run(scene,window_ids,config,output):
    start=time.perf_counter();cfg=yaml.safe_load(Path(config).read_text());scene=Path(scene)
    expected={'format':'rind3d-ambiguity-enrichment-research-v1','max_candidates':8,'minimum_source_separation':32/1024,'candidate_domain':'world_outside_window','artifact_policy':'immutable_external_private'}
    if any(cfg.get(k)!=v for k,v in expected.items()) or cfg.get('method') not in ('random','local','silhouette','inverse','hybrid') or cfg.get('probe_strategy') not in ('spatial','transition'):raise ValueError('Unsupported enrichment contract/settings')
    if cfg['window_budget_seconds']<0 or cfg['screen_probes'] not in (16,32,64) or not 1<=cfg['maximum_windows_per_invocation']<=64:raise ValueError('Explicit bounded enrichment budgets required')
    if not window_ids or len(set(window_ids))!=len(window_ids) or len(window_ids)>cfg['maximum_windows_per_invocation']:raise ValueError('Duplicate/empty/excessive enriched windows')
    if Path(output).exists():raise FileExistsError('Choose a new immutable artifact version')
    if scene.resolve() in Path(output).resolve().parents:raise ValueError('Enrichment output must be outside the READY scene')
    checked=verify_scene(scene);meta=json.loads((scene/'verification.json').read_text())
    if meta.get('status')!='READY_LITE':raise ValueError('This research entry accepts certified READY_LITE scenes only')
    validate_certificate(json.loads(Path(cfg['scientific_certificate']).read_text()))
    context=GeometryContext(load_geometry(scene/'geometry.npz'),meta['numerical_policy'],source_cache_size=cfg['source_model_cache']);truth=np.fromfile(scene/'driver.f64',dtype='<f8')
    raw=np.fromfile(scene/'window_records.bin',dtype='<u2').reshape(-1,4);v=SharedCandidateVerifier(context,cache_bricks=cfg['physical_brick_cache'],cache_sources=cfg['source_model_cache']);records=[]
    with RINDPack(scene/'response.rindb',verify_file=True) as y,RINDPack(scene/'solid.rindb',verify_file=True) as m:
        for wid in window_ids:
            if not 0<=wid<len(raw):raise IndexError(wid)
            row=raw[wid];w=Window(tuple(map(int,row[:3])),int(row[3]));positions,proofs,report=search(context,truth,w,y,m,seconds=cfg['window_budget_seconds'],method=cfg['method'],seed=cfg['seed']+wid,
                    verifier=v,probe_count=cfg['screen_probes'],probe_strategy=cfg['probe_strategy'])
            records.append({'window_id':int(wid),'positions':positions.tolist(),'proofs':proofs,'report':report,'configuration':cfg})
    save_artifact(output,scene,records)
    return {'artifact':str(output),'private':True,'windows':len(records),'additional_references':sum(len(r['positions'])-1 for r in records),'wall_seconds_including_validation':time.perf_counter()-start,
            'core_scene_modified':False,'candidate_count_is_not_uniqueness':True}
