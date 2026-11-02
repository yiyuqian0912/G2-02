"""Mandatory, bounded feasibility experiment; never publishes dataset scenes."""
from __future__ import annotations
from collections import Counter
import hashlib,json,platform,time
from pathlib import Path
import resource
import numpy as np
from .geometry import Oracle,generate_geometry,save_geometry,Budget,Aborted,NumericalUnresolved,POLICY,keyed_rng
from .windows import sample_windows
from .search import free_pool,search_candidates
from .pack import atomic_json
from .window_contract import WINDOW_SIDES,WINDOW_PROTOCOL,PILOT_SIDE_COUNTS,require_side_counts,side_counts,validate_window_config

def config_hash(cfg):
    # JSON object keys become strings on disk. Restore the two known integer
    # quota maps before sorting, preserving every existing YAML config hash.
    value=dict(cfg)
    for key in ('pilot_side_counts','production_side_counts'):
        counts=value.get(key)
        if isinstance(counts,dict) and set(counts)==set(map(str,WINDOW_SIDES)):
            value[key]={int(k):v for k,v in counts.items()}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def source_fingerprint():
    """Bind a passing pilot to the implementation that was actually exercised."""
    from . import _native
    from rind.geometry import backend_build_id
    from rind import contracts
    root=Path(__file__).resolve().parents[2]
    files=list((root/'src/rind3d').rglob('*.py'))+list((root/'native').glob('*.cpp'))+list((root/'native').glob('*.hpp'))+[root/'CMakeLists.txt']
    result={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    dependency=Path(contracts.__file__).parent
    result.update({'canonical/'+p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dependency.glob('*.py'))})
    result['compiled_rind3d']=hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest()
    result['canonical_backend']=backend_build_id(3)
    result['rindray_backend']=_native.rindray_build_id()
    return result

def summarize_windows(values):
    valid=[w for w in values if w['status']=='COMPLETE']
    counts=[w['actual_candidates'] for w in valid];times=[w['search_seconds'] for w in values]
    return {'windows_measured':len(values),'verified_windows':len(valid),
            'candidate_count_distribution':dict(sorted(Counter(str(v) for v in counts).items(),key=lambda x:int(x[0]))),
            'min_candidates':min(counts,default=0),'median_candidates':float(np.median(counts)) if counts else None,
            'max_candidates':max(counts,default=0),'with_64':sum(v==64 for v in counts),
            'median_search_seconds':float(np.median(times)) if times else None,'max_search_seconds':max(times,default=0),
            'search_seconds':sum(times),'total_window_seconds':sum(w['elapsed_seconds'] for w in values),
            'verification_seconds':sum(w['verification_seconds']+w['independent_verification_seconds'] for w in values),
            'physical_point_calls':sum(w['physical_point_calls']+w['independent_point_calls'] for w in values),
            'status_counts':dict(Counter(w['status'] for w in values)),
            'termination_counts':dict(Counter(w['termination_reason'] for w in values))}


def run_pilot(cfg,output,scenes=8,windows_per_scene=32):
    from .pipeline import write_candidate_files,validate_records
    from .search import CandidateBank,farthest
    from .candidates import STORAGE
    from .pack import sha256_file
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use a new pilot output directory')
    output.mkdir(parents=True,exist_ok=True)
    validate_window_config(cfg)
    if scenes<1 or windows_per_scene!=32:raise ValueError('This pilot requires exactly 32 windows: 10/11/11 across 64/128/256')
    result={'format':'rind3d-pilot-v2','config':cfg,'config_sha256':config_hash(cfg),'physics_policy':POLICY,
            'scene_count':scenes,'windows_per_scene':windows_per_scene,'is_dataset':False,
            'window_protocol':WINDOW_PROTOCOL,'window_sides':list(WINDOW_SIDES),'expected_side_counts_per_scene':PILOT_SIDE_COUNTS,
            'host':platform.platform(),'scenes':[],'started_unix':time.time(),'source_sha256':source_fingerprint()}
    for scene_id in range(scenes):
        t=time.perf_counter();families=cfg['families'][scene_id%len(cfg['families'])]
        directory=output/f'scene-{scene_id:06d}';directory.mkdir()
        row={'scene_id':scene_id,'families':families,'status':'INCOMPLETE','windows':[],'source_attempts':[],
             'candidate_files':directory.name,'candidate_storage':STORAGE,'max_candidates':64,'resolution':cfg['resolution'],
             'min_source_separation':cfg['min_source_separation'],'true_source_included':True,'window_protocol':WINDOW_PROTOCOL}
        try:
            bundle=generate_geometry(cfg['seed'],scene_id,families);save_geometry(directory/'geometry.npz',bundle)
            oracle=Oracle(bundle,cfg.get('numerical_policy'));row['capability']=oracle.capability
            row['geometry_build_seconds']=time.perf_counter()-t
            setup=Budget(cfg['scene_setup_seconds'],cfg['window_search_points']);pool=free_pool(oracle,cfg,scene_id,setup)
            if not len(pool):raise Aborted('No certified free source in finite proposal pool')
            row['free_candidate_pool']=len(pool);row['source_pool_seconds']=time.perf_counter()-t-row['geometry_build_seconds']
            sample_start=time.perf_counter();best=None
            for source_id in farthest(pool,min(cfg['pilot_source_attempts'],len(pool)),0):
                truth=pool[source_id].copy();oracle.validate_source(truth,Budget(cfg['verification_seconds'],cfg['verification_points']))
                attempt={'pool_index':source_id,'true_source':truth.tolist()};ts=time.perf_counter()
                try:
                    windows,sampling=sample_windows(truth,cfg,scene_id,windows_per_scene,oracle=oracle)
                    attempt['status']='COMPLETE' if len(windows)==windows_per_scene else 'INCOMPLETE'
                except Aborted as exc:
                    windows=getattr(exc,'partial_windows',[]);sampling=getattr(exc,'sampling_report',{})
                    attempt.update(status='ABORTED',reason=str(exc))
                attempt.update(sampling=sampling,elapsed_seconds=time.perf_counter()-ts);row['source_attempts'].append(attempt)
                if best is None or len(windows)>len(best[1]):best=(truth,windows,sampling)
                if len(windows)==windows_per_scene:break
            truth,windows,sampling=best;row['true_source']=truth.tolist();row['numerical_policy']=oracle.provenance
            truth.astype('<f8').tofile(directory/'driver.f64')
            row['sampling']=sampling;row['window_sampling_seconds']=time.perf_counter()-sample_start
            bank=CandidateBank();ids=[];reports=[]
            for window_id,window in enumerate(windows):
                positions,evidence,report=search_candidates(oracle,truth,window,pool,cfg,scene_id,window_id)
                ids.append([bank.add(p) for p in positions]);reports.append(report)
                row['windows'].append(dict(report,window_id=window_id,origin=list(window.origin),size=window.size,
                                           evidence=evidence))
                atomic_json(output/f'scene-{scene_id:06d}.json',row)
            if windows:
                dtype,_=write_candidate_files(directory,windows,bank,ids,reports);row['candidate_id_dtype']=dtype
                validate_records(directory,row,strict=False)
                row['candidate_bytes']={p.name:p.stat().st_size for p in directory.iterdir() if p.name.startswith(('candidate_','window_','search_records'))}
                row['file_hashes']={p.name:sha256_file(p) for p in directory.iterdir() if p.is_file()}
            row['side_counts']=side_counts(w.size for w in windows)
            row['search_feasible']=row['side_counts']==PILOT_SIDE_COUNTS and sampling['occupied_octants']>=2 and all(w['status']=='COMPLETE' for w in reports)
            row['publication_feasible']=row['search_feasible'] and oracle.capability.get('ready_capable',True)
            row['status']='COMPLETE' if row['publication_feasible'] else 'CERTIFICATION_BLOCKED' if row['search_feasible'] else 'INCOMPLETE'
            if not oracle.capability.get('ready_capable',True):row['publication_blocker']='Inherited functional oracle declares ready_capable=False'
            if any(w['status']=='UNRESOLVED' for w in reports):row['status']='UNRESOLVED'
            row['oracle_work']=dict(oracle.metrics)
        except (Aborted,NumericalUnresolved) as exc:
            row.update(status='ABORTED' if isinstance(exc,Aborted) else 'UNRESOLVED',reason=str(exc))
        row['elapsed_seconds']=time.perf_counter()-t;row['summary']=summarize_windows(row['windows'])
        row['diagnostic_bytes']=sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
        result['scenes'].append(row);atomic_json(output/f'scene-{scene_id:06d}.json',row);atomic_json(output/'report.json',result)
        print(f"pilot scene {scene_id}: {row['status']}; {len(row['windows'])}/{windows_per_scene} windows; {row['elapsed_seconds']:.2f}s",flush=True)
    windows=[w for scene in result['scenes'] for w in scene['windows']]
    covered={tuple(s['families']) for s in result['scenes']}=={tuple(s) for s in cfg['families']}
    measured=scenes>=8 and windows_per_scene==32 and covered
    result.update(completed=True,summary=summarize_windows(windows),
        by_window_side={str(side):dict(summarize_windows([w for w in windows if w['size']==side]),window_side=side,expected_windows=scenes*PILOT_SIDE_COUNTS[side]) for side in WINDOW_SIDES},
        actual_side_counts=side_counts(w['size'] for w in windows),
        search_feasible=measured and all(s.get('search_feasible',False) for s in result['scenes']),
        publication_feasible=measured and all(s.get('publication_feasible',False) for s in result['scenes']),
        elapsed_seconds=time.time()-result['started_unix'],max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    result['candidate_storage_bytes']=sum(sum(s.get('candidate_bytes',{}).values()) for s in result['scenes'])
    atomic_json(output/'report.json',result)
    return result
