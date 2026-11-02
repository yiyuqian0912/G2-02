"""Fail-closed publication gates. A pilot failure is not permission to scale."""
from __future__ import annotations
import hashlib,json,os,time,struct
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .pack import RINDPack,sha256_file,atomic_json
from .geometry import Oracle,generate_geometry,save_geometry,load_geometry,filter_primitives,Unsupported,Budget,Aborted,Incomplete,NumericalUnresolved,POLICY,POLICY_HASH
from .stream import stream_response
from .windows import sample_windows,excludes,informative,Window,placement_summary
from .search import CandidateBank,free_pool,search_candidates,separated,verify_grid_exact
from .pilot import config_hash,source_fingerprint
from .candidates import STORAGE,storage_version,read_references
from .window_contract import WINDOW_SIDES,WINDOW_PROTOCOL,PILOT_SIDE_COUNTS,PRODUCTION_SIDE_COUNTS,require_window_side,require_side_counts

EVIDENCE=struct.Struct('<IIQ32s')  # flags(grid_exact=1, independent=2), count, voxels, source-set SHA256

def parquet_write(path,rows,schema=None):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    pq.write_table(pa.Table.from_pylist(rows,schema=schema),tmp,compression='zstd');os.replace(tmp,path)

def split_for(seed,scene_id):
    value=int.from_bytes(hashlib.sha256(f'{seed}/{scene_id}/split-v1'.encode()).digest()[:8],'little')%100
    return 'train' if value<80 else 'val' if value<90 else 'test'

def search_metrics(reports,resampling):
    attempts=reports+[r['discarded_search'] for r in resampling]
    total=sum(r['elapsed_seconds'] for r in attempts)
    verification=sum(r['verification_seconds'] for r in attempts)
    independent=sum(r['independent_verification_seconds'] for r in attempts)
    return {'windows':len(reports),'search_attempts':len(attempts),'resampled_windows':len(resampling),
            'statuses':{k:sum(r['status']==k for r in reports) for k in ('COMPLETE','INCOMPLETE','ABORTED','UNRESOLVED')},
            'search_total_seconds':total,'search_overhead_seconds':max(0.,total-verification-independent),
            'full_verification_seconds':verification,'independent_verification_seconds':independent,
            'full_verifications':sum(r['full_verifications'] for r in attempts),
            'physical_point_calls':sum(r['physical_point_calls']+r['independent_point_calls'] for r in attempts)}

def gate_pilot(cfg,path):
    report=json.loads(Path(path).read_text())
    if report.get('format')!='rind3d-pilot-v2' or report.get('config_sha256')!=config_hash(cfg):raise ValueError('A matching v2 feasibility pilot is required')
    if not report.get('publication_feasible'):raise ValueError('Feasibility gate CLOSED: sampling, certification or retained-source verification remains incomplete. See '+str(path))
    scenes=report.get('scenes',[]);number=report.get('windows_per_scene',0)
    if not report.get('completed') or report.get('scene_count')!=len(scenes) or len(scenes)<8 or number!=32:raise ValueError('Passing pilot lacks the mandatory measured sample')
    if {tuple(s['families']) for s in scenes}!={tuple(s) for s in cfg['families']}:raise ValueError('Pilot does not cover every configured family mixture')
    if report.get('window_protocol')!=WINDOW_PROTOCOL or report.get('window_sides')!=list(WINDOW_SIDES):raise ValueError('Pilot uses the obsolete window-size protocol')
    if report.get('source_sha256')!=source_fingerprint():raise ValueError('Pilot implementation changed; rerun feasibility')
    for scene in scenes:
        if not scene['capability'].get('ready_capable',True) or len(scene['windows'])!=number:raise ValueError('Pilot lacks complete certified-scene evidence')
        require_side_counts((w['size'] for w in scene['windows']),PILOT_SIDE_COUNTS)
        truth=np.asarray(scene['true_source'],dtype=np.float64)
        directory=Path(path).parent/scene['candidate_files']
        if directory.parent!=Path(path).parent:raise ValueError('Invalid pilot scene directory')
        for name,digest in scene['file_hashes'].items():
            if (directory/name).parent!=directory or sha256_file(directory/name)!=digest:raise ValueError('Pilot candidate file hash mismatch')
        records,counts,bank,ids=validate_records(directory,scene,strict=False)
        if len(records)!=number:raise ValueError('Pilot canonical window count mismatch')
        require_side_counts(records[:,3],PILOT_SIDE_COUNTS)
        for index,w in enumerate(scene['windows']):
            points=bank[ids[index]]
            if tuple(records[index])!=(*w['origin'],w['size']):raise ValueError('Pilot window metadata mismatch')
            if w['status']!='COMPLETE' or not w['independent_validation_passed'] or not w['all_grid_exact']:raise ValueError('Pilot contains unverified/interrupted candidates')
            if not 1<=len(points)<=64 or len(points)!=w['actual_candidates'] or len(w['evidence'])!=len(points):raise ValueError('Pilot candidate count mismatch')
            if not np.isfinite(points).all() or np.any(points<=0) or np.any(points>=1) or not any(np.array_equal(p,truth) for p in points):raise ValueError('Pilot true source missing or candidate invalid')
            for e in w['evidence']:
                if e['status']!='grid_exact' or e.get('independent_status')!='grid_exact' or e['covered_voxels']!=w['size']**3:raise ValueError('Pilot contains probe-only evidence')
            for i,p in enumerate(points):
                if any(not separated(p,q,cfg['min_source_separation']) for q in points[i+1:]):raise ValueError('Pilot source separation violated')
    return report


def write_candidate_files(root,windows,bank,ids,reports,*,legacy_diagnostic=False):
    """v2 stores only actual references; zero counts are staging-only."""
    root=Path(root);dtype=bank.index_dtype;coordinates=bank.array
    if legacy_diagnostic:
        from .legacy.window_contract import require_window_side as validate_side
    else:validate_side=require_window_side
    for window in windows:validate_side(window.size)
    rows=list(ids)+[[] for _ in range(len(windows)-len(ids))]
    if len(rows)!=len(windows) or any(len(r)>64 for r in rows):raise ValueError('Invalid variable candidate counts')
    for row in rows:
        if any(int(i)!=i or not 0<=i<len(coordinates) for i in row):raise ValueError('Invalid candidate reference')
    coordinates.tofile(root/'candidate_bank.bin')
    np.asarray([w.record() for w in windows],dtype='<u2').reshape(-1,4).tofile(root/'window_records.bin')
    counts=np.asarray([len(v) for v in rows],dtype='<u2');counts.tofile(root/'candidate_counts.bin')
    offsets=np.r_[np.uint64(0),np.cumsum(counts,dtype=np.uint64)].astype('<u8')
    offsets.tofile(root/'candidate_offsets.bin')
    np.asarray([i for row in rows for i in row],dtype=dtype).tofile(root/'candidate_ids.bin')
    witness=np.zeros((len(windows),2,3),dtype='<u2');evidence=[];search_rows=[]
    for i,w in enumerate(windows):
        if len(w.witnesses)==2:witness[i]=w.witnesses
        report=reports[i] if i<len(reports) else {}
        flags=int(bool(report.get('all_grid_exact')))+2*int(bool(report.get('independent_validation_passed')))
        digest=hashlib.sha256(coordinates[rows[i]].astype('<f8').tobytes()).digest()
        evidence.append(EVIDENCE.pack(flags,len(rows[i]),w.size**3 if flags&1 else 0,digest))
        row={k:v for k,v in report.items() if isinstance(v,(str,int,float,bool)) or v is None}
        row.update(window_id=i,actual_candidates=len(rows[i]));search_rows.append(row)
    witness.tofile(root/'window_witnesses.bin');(root/'candidate_evidence.bin').write_bytes(b''.join(evidence))
    parquet_write(root/'search_records.parquet',search_rows)
    return dtype.str,counts

def validate_records(root,meta,*,strict=True,legacy_diagnostic=False):
    if strict and legacy_diagnostic:raise ValueError('Legacy diagnostics cannot authorize publication')
    root=Path(root);records=np.fromfile(root/'window_records.bin',dtype='<u2')
    if records.size%4 or (root/'window_records.bin').stat().st_size!=records.nbytes:raise ValueError('Truncated window table')
    windows=records.reshape(-1,4);counts=np.fromfile(root/'candidate_counts.bin',dtype='<u2')
    bank=np.fromfile(root/'candidate_bank.bin',dtype='<f8')
    if bank.size%3 or (root/'candidate_bank.bin').stat().st_size!=bank.nbytes or not np.isfinite(bank).all():raise ValueError('Invalid candidate bank')
    bank=bank.reshape(-1,3);dtype=np.dtype(meta['candidate_id_dtype'])
    if dtype.str not in ('<u2','<u4'):raise ValueError('Unsupported candidate index endian/dtype')
    expected_dtype='<u2' if len(bank)<=65535 else '<u4'
    if dtype.str!=expected_dtype:raise ValueError('Candidate index width inconsistent with bank')
    if len(counts)!=len(windows) or (root/'candidate_counts.bin').stat().st_size!=counts.nbytes:raise ValueError('Candidate/window table shape mismatch')
    version=storage_version(root,meta)
    if meta.get('contract_version',1)==2 and version!=2:raise ValueError('Contract v2 requires offsets storage')
    ids=read_references(root,meta,len(windows),counts)
    if len(np.unique(windows,axis=0))!=len(windows):raise ValueError('Duplicate windows')
    if legacy_diagnostic:
        from .legacy.window_contract import require_window_side as validate_side
    else:validate_side=require_window_side
    for side in windows[:,3]:validate_side(side)
    from .profiles import profile_for
    profile=profile_for(meta)
    if strict and (meta['resolution']!=1024 or len(windows)!=profile.windows):raise ValueError('Production READY counts or resolution not met')
    if strict:require_side_counts(windows[:,3],profile.quotas)
    if version==2:
        maximum=profile.max_candidates if strict else 64
        if np.any(counts>maximum) or (strict and np.any(counts<1)):raise ValueError('READY candidate counts violate profile')
        if strict and (meta.get('max_candidates')!=profile.max_candidates or not meta.get('true_source_included')):raise ValueError('Profile candidate cap or true source mismatch')
        if meta['min_source_separation']!=32/1024:raise ValueError('Contract v2 source separation must remain 32/1024')
    elif np.any(counts>128) or (strict and meta.get('status')=='READY' and np.any(counts!=128)):raise ValueError('Legacy READY counts invalid')
    if not np.isfinite(meta['min_source_separation']) or meta['min_source_separation']<=0:raise ValueError('Invalid minimum source separation')
    if len({p.tobytes() for p in bank})!=len(bank):raise ValueError('Duplicate bank coordinates')
    for record,count,row in zip(windows,counts,ids):
        x,y,z,size=map(int,record)
        if min(x,y,z)<0 or max(x,y,z)+size>meta['resolution']:raise ValueError('Window bounds/size invalid')
        valid=row
        if len(np.unique(valid))!=count or np.any(valid>=len(bank)):raise ValueError('Invalid/duplicate candidate references')
        points=bank[valid]
        if len(points)>1:
            i,j=np.triu_indices(len(points),1);distance=((points[i]-points[j])**2).sum(axis=1)
            limit=meta['min_source_separation']**2;margin=32*np.finfo(float).eps*np.maximum(distance,limit)
            if np.any(distance<limit-margin):raise ValueError('Source separation violated')
            for k in np.flatnonzero(np.abs(distance-limit)<=margin):
                if not separated(points[i[k]],points[j[k]],meta['min_source_separation']):raise ValueError('Source separation violated')
    return windows,counts,bank,ids

def generate_dataset(cfg,output,scene_count,pilot_path,*,resume=False):
    from .profiles import profile_for,LITE
    if profile_for(cfg).name==LITE:
        from .lite_pipeline import generate_lite
        return generate_lite(cfg,output,scene_count,pilot_report=pilot_path,resume=resume)
    gate_pilot(cfg,pilot_path)
    if scene_count<1:raise ValueError('At least one production scene is required')
    if cfg['resolution']!=1024:raise ValueError('Production resolution must be exactly 1024')
    root=Path(output);root.mkdir(parents=True,exist_ok=True);(root/'scenes').mkdir(exist_ok=True);(root/'.staging').mkdir(exist_ok=True)
    manifest={'format':'RIND3D','version':2,'config':cfg,'config_sha256':config_hash(cfg),'policy':POLICY,'policy_sha256':POLICY_HASH,
              'implementation':source_fingerprint(),'window_protocol':WINDOW_PROTOCOL,'window_sides':list(WINDOW_SIDES),
              'response_storage':'RINDPack-v1','response_order':'ZYX/LSB-first-X','source_label_access':'private-oracle-only',
              'split_policy':'sha256(seed/scene_id/split-v1) modulo100: train<80 val<90 else test',
              'pilot_sha256':sha256_file(pilot_path),'expected_scenes':scene_count,'max_candidates':64,'candidate_storage':STORAGE}
    p=root/'manifest.json'
    if p.exists():
        if not resume or json.loads(p.read_text())!=manifest:raise ValueError('Immutable manifest mismatch or resume not requested')
    else:atomic_json(p,manifest)
    scene_rows=[];window_rows=[]
    for sid in range(scene_count):
        name=f'scene_{sid:06d}';final=root/'scenes'/name;staging=root/'.staging'/name;families=cfg['families'][sid%len(cfg['families'])]
        if final.exists():
            checked=verify_scene(final,full_index=True);scene_rows.append(checked['scene_row']);window_rows.extend(checked['window_rows']);continue
        staging.mkdir(exist_ok=True);start=time.perf_counter()
        meta={'scene_id':name,'resolution':1024,'status':'INCOMPLETE','families':families,'split':split_for(cfg['seed'],sid),
              'min_source_separation':cfg['min_source_separation'],'max_candidates':64,'contract_version':2,'candidate_storage':STORAGE,
              'candidate_domain':cfg['candidate_domain'],'numerical_policy':cfg['numerical_policy'],
              'config_sha256':config_hash(cfg),'window_protocol':WINDOW_PROTOCOL,
              'true_source_included':cfg['include_true_source'],'equivalence_status':'grid_exact','continuous_certified':False}
        reports=[];resampling=[]
        try:
            bundle=generate_geometry(cfg['seed'],sid,families);save_geometry(staging/'geometry.npz',bundle);oracle=Oracle(bundle,cfg['numerical_policy'])
            if not oracle.capability.get('ready_capable',True):raise NumericalUnresolved('Inherited oracle does not declare this geometry scope READY capable')
            meta['geometry_build_seconds']=time.perf_counter()-start;pool_start=time.perf_counter()
            setup=Budget(cfg['scene_setup_seconds'],cfg['window_search_points']);pool=free_pool(oracle,cfg,sid,setup)
            if not len(pool):raise Aborted('No free source found')
            truth=pool[0];truth.astype('<f8').tofile(staging/'driver.f64')
            meta['source_pool_seconds']=time.perf_counter()-pool_start;meta['geometry_hash']=bundle.geometries[0].content_hash.hex()
            meta['generation']=stream_response(oracle,truth,staging,cfg,resume=resume)
            meta['numerical_policy_sha256']=oracle.policy_hash
            with RINDPack(staging/'response.rindb') as response,RINDPack(staging/'solid.rindb') as solid:
                from .information import InformationIndex
                from .stored_windows import select_stored_windows
                t=time.perf_counter();information=InformationIndex.load(staging/'information.rindi',response,solid)
                selection=select_stored_windows(information,truth,4096,seed=cfg['seed'],scene=sid,settings=cfg.get('window_selection'))
                windows=selection.windows;sampling=selection.report;meta['sampling']=sampling;meta['window_sampling_seconds']=time.perf_counter()-t
                if selection.provenance:parquet_write(staging/'window_selection.parquet',selection.provenance)
                if selection.controls:parquet_write(staging/'uniform_control.parquet',selection.controls)
                if not sampling['complete']:raise Aborted('Exact stored window quotas incomplete: '+str(sampling.get('side_counts')))
                require_side_counts((w.size for w in windows),PRODUCTION_SIDE_COUNTS)
                if len(windows)!=4096 or sampling['placement']['occupied_octants']<2:raise Aborted('Window count or existing scene spatial-coverage requirement not met')
                bank=CandidateBank();ids=[]
                for wid,w in enumerate(windows):
                    observed=response.read_window(*w.origin,w.size)
                    points,evidence,report=search_candidates(oracle,truth,w,pool,cfg,sid,wid,observed=observed)
                    ids.append([bank.add(p) for p in points]);reports.append(report)
                    meta['candidate_window_resampling']=resampling
                    if report['status']!='COMPLETE':
                        dtype,_=write_candidate_files(staging,windows,bank,ids,reports)
                        meta.update(candidate_id_dtype=dtype,window_reports=reports)
                    if report['status'] in ('ABORTED','UNRESOLVED'):raise (Aborted if report['status']=='ABORTED' else NumericalUnresolved)(report['reason'])
                    if report['status']!='COMPLETE' or not 1<=len(points)<=64:raise Incomplete(f'Window {wid}: no fully verified true source')
                dtype,counts=write_candidate_files(staging,windows,bank,ids,reports);meta.update(candidate_id_dtype=dtype,
                    search_summary=search_metrics(reports,resampling))
                meta.pop('window_reports',None)
                validate_records(staging,meta,strict=True)
            meta['hashes']={x.name:sha256_file(x) for x in staging.iterdir() if x.is_file() and x.name!='verification.json'
                            and not x.name.endswith(('.tmp','.partial','.compact','.journal.json'))}
            meta['status']='READY';meta['elapsed_seconds']=time.perf_counter()-start
            atomic_json(staging/'verification.json',meta);verify_scene(staging,full_index=True)
            os.replace(staging,final)
            checked=verify_scene(final);scene_rows.append(checked['scene_row']);window_rows.extend(checked['window_rows'])
            parquet_write(root/'scenes.parquet',scene_rows);parquet_write(root/'windows.parquet',window_rows)
            parquet_write(root/'splits.parquet',[{'scene_id':r['scene_id'],'split':r['split']} for r in scene_rows])
        except (Aborted,Incomplete,NumericalUnresolved) as e:
            meta.update(status='UNRESOLVED' if isinstance(e,NumericalUnresolved) else 'ABORTED' if isinstance(e,Aborted) else 'INCOMPLETE',reason=str(e),elapsed_seconds=time.perf_counter()-start)
            meta['search_summary']=search_metrics(reports,resampling)
            if hasattr(e,'sampling_report'):meta['sampling_interrupted']=e.sampling_report
            atomic_json(staging/'verification.json',meta);raise
    parquet_write(root/'scenes.parquet',scene_rows);parquet_write(root/'windows.parquet',window_rows)
    parquet_write(root/'splits.parquet',[{'scene_id':r['scene_id'],'split':r['split']} for r in scene_rows])
    return scene_rows

def verify_scene(path,*,full_index=False,physical=False):
    path=Path(path);meta=json.loads((path/'verification.json').read_text())
    if meta.get('dataset_profile')=='rind3d-lite-v0.1':
        if 'campaign_version' in meta:
            from .campaign_contract import check_scene
            check_scene(path,meta)
        if 'production_revision' in meta:
            from .core_checks import check
            check(path,meta)
        from .lite_publication import verify_lite_scene
        return verify_lite_scene(path,full_index=full_index,physical=physical)
    if meta['status'] not in ('READY','READY_SHORTFALL'):raise ValueError('Scene is not READY: '+meta['status'])
    if meta.get('contract_version',1)==2 and meta['status']!='READY':raise ValueError('Contract v2 uses READY without shortfall mode')
    if meta.get('equivalence_status')!='grid_exact' or meta.get('continuous_certified'):raise ValueError('Unsupported verification status')
    if meta['status']=='READY_SHORTFALL' and not meta.get('allow_shortfall'):raise ValueError('Shortfall was not explicitly enabled')
    required={'geometry.npz','driver.f64','response.rindb','solid.rindb','window_records.bin',
              'candidate_bank.bin','candidate_ids.bin','candidate_counts.bin','candidate_evidence.bin','window_witnesses.bin'}
    if meta.get('contract_version',1)==2:required|={'candidate_offsets.bin','search_records.parquet'}
    if meta.get('window_protocol')==WINDOW_PROTOCOL:required|={'information.rindi','window_selection.parquet'}
    if not required<=meta['hashes'].keys():raise ValueError('Missing canonical file hashes')
    for name,digest in meta['hashes'].items():
        p=path/name
        if p.parent!=path or sha256_file(p)!=digest:raise ValueError('File hash mismatch: '+name)
    windows,counts,bank,ids=validate_records(path,meta,strict=meta['status']=='READY')
    if meta['resolution']!=1024 or len(windows)!=4096:raise ValueError('Every published scene requires 1024 cubed and 4096 windows, including shortfall mode')
    require_side_counts(windows[:,3],PRODUCTION_SIDE_COUNTS)
    restored=[Window(tuple(map(int,r[:3])),int(r[3])) for r in windows]
    if placement_summary(restored,1024)['occupied_octants']<2:raise ValueError('Insufficient window spatial coverage')
    geometry=load_geometry(path/'geometry.npz')
    if geometry.geometries[0].content_hash.hex()!=meta['geometry_hash']:raise ValueError('Canonical geometry hash mismatch')
    truth=np.fromfile(path/'driver.f64',dtype='<f8')
    if truth.shape!=(3,) or not np.isfinite(truth).all() or np.any(truth<=0) or np.any(truth>=1):raise ValueError('Invalid true source')
    if np.any(bank<=0) or np.any(bank>=1):raise ValueError('Candidate source outside world')
    from ._native import IntervalFilter
    try:
        certificate=IntervalFilter(filter_primitives(geometry));inside=certificate.contains(np.ascontiguousarray(np.vstack([truth,bank]),dtype=np.float64))
        if np.any(inside==1):raise ValueError('Source inside geometry')
        if np.any(inside<0):
            oracle=Oracle(geometry)
            if oracle.contains(np.vstack([truth,bank])[inside<0]).any():raise ValueError('Source not certified free')
    except Unsupported:
        oracle=Oracle(geometry,meta['numerical_policy'])
        if not oracle.capability.get('ready_capable',True):raise ValueError('Canonical oracle does not support READY publication for this geometry')
        if oracle.contains(np.vstack([truth,bank])).any():raise ValueError('Source not certified free')
    raw_evidence=(path/'candidate_evidence.bin').read_bytes()
    if len(raw_evidence)!=len(windows)*EVIDENCE.size:raise ValueError('Candidate evidence length mismatch')
    witnesses=np.fromfile(path/'window_witnesses.bin',dtype='<u2')
    if witnesses.size!=len(windows)*6:raise ValueError('Window witness length mismatch')
    witnesses=witnesses.reshape(-1,2,3)
    for i,(flags,count,covered,digest) in enumerate(EVIDENCE.iter_unpack(raw_evidence)):
        if flags!=3 or count!=counts[i] or covered!=int(windows[i,3])**3:raise ValueError('Incomplete/sample-only candidate evidence')
        if digest!=hashlib.sha256(bank[ids[i]].astype('<f8').tobytes()).digest():raise ValueError('Candidate evidence coordinate hash mismatch')
    if meta.get('contract_version',1)==2:
        search_rows=pq.read_table(path/'search_records.parquet').to_pylist()
        if len(search_rows)!=len(windows) or any(r.get('window_id')!=i or r.get('actual_candidates')!=int(counts[i]) or r.get('status')!='COMPLETE' or not r.get('search_termination_reason') for i,r in enumerate(search_rows)):raise ValueError('Search records disagree with verified candidates')
    with RINDPack(path/'response.rindb',verify_file=True) as response,RINDPack(path/'solid.rindb',verify_file=True) as solid:
        if response.shape!=(1024,)*3 or solid.shape!=response.shape:raise ValueError('Production field shape invalid')
        if response.policy!=meta['numerical_policy_sha256'] or solid.policy!=response.policy:raise ValueError('Numerical policy mismatch')
        identity=hashlib.sha256(geometry.geometries[0].content_hash+truth.astype('<f8').tobytes()+bytes.fromhex(response.policy)).hexdigest()
        if response.identity!=identity or solid.identity!=identity:raise ValueError('Geometry/source/field identity mismatch')
        if meta.get('window_protocol')==WINDOW_PROTOCOL:
            from .information import InformationIndex
            InformationIndex.load(path/'information.rindi',response,solid)
            selection=pq.read_table(path/'window_selection.parquet').to_pylist()
            if len(selection)!=len(windows):raise ValueError('Selection provenance/window count mismatch')
            for i,(record,row) in enumerate(zip(windows,selection)):
                if row['window_id']!=i or row['origin']!=record[:3].tolist() or row['side']!=int(record[3]) or row['provenance'] not in ('uniform_accepted','guided_informative'):
                    raise ValueError('Selection provenance/window alignment mismatch')
        provenance=meta['generation']['numerical_policy']
        if hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()!=response.policy or provenance['physics']!=POLICY or provenance['oracle_limits']!=meta['numerical_policy']:raise ValueError('Incomplete numerical policy provenance')
        if full_index:response.verify_all();solid.verify_all()
        solid_fraction=sum(row[7] for row in solid.index)/(1024**3)
        tasks=[]
        domain=meta['candidate_domain']
        if domain not in ('world_free','world_outside_window'):raise ValueError('Unsupported candidate domain')
        for wid,(record,pair) in enumerate(zip(windows,witnesses)):
            w=Window(tuple(map(int,record[:3])),int(record[3]))
            if not excludes(w,truth,1024):raise ValueError('Observation contains true source')
            positions=bank[ids[wid]]
            if meta['true_source_included'] and not any(np.array_equal(p,truth) for p in positions):raise ValueError('Declared true source missing from candidate set')
            if domain=='world_outside_window' and any(not excludes(w,p,1024) for p in positions):raise ValueError('Candidate violates the declared outside-window domain')
            for point,expected in zip(pair,(True,False)):
                if np.any(point<record[:3]) or np.any(point>=record[:3]+int(record[3])):raise ValueError('Informative witness outside observation')
                tasks.append((tuple(map(int,point)),expected))
        for point,expected in sorted(tasks,key=lambda x:tuple(v//64 for v in x[0])):
            if solid.read_point(*point) or response.read_point(*point)!=expected:raise ValueError('Invalid informative-window witness')
        if physical:
            oracle=Oracle(geometry,meta['numerical_policy']);oracle.validate_source(truth)
            if oracle.policy_hash!=response.policy:raise ValueError('Physical verification requires the recorded numerical implementation')
            for wid,record in enumerate(windows):
                w=Window(tuple(map(int,record[:3])),int(record[3]))
                y=response.read_window(*w.origin,w.size);m=solid.read_window(*w.origin,w.size)
                if not excludes(w,truth,1024) or not informative(y,m)[0]:raise ValueError('Window invariant failed')
                for point in bank[ids[wid]]:
                    oracle.validate_source(point)
                    r=verify_grid_exact(oracle,truth,point,w,1024,Budget(300,50_000_000),observed=y)
                    if r['status']!='grid_exact':raise ValueError('Physical equivalence verification failed')
    row={'scene_id':meta['scene_id'],'split':meta['split'],'families':meta['families'],'status':meta['status'],
         'window_count':len(windows),'candidate_count_min':int(counts.min()),'candidate_count_max':int(counts.max()),
         'bytes':sum(x.stat().st_size for x in path.iterdir() if x.is_file()),'verification_sha256':sha256_file(path/'verification.json')}
    row['solid_fraction']=solid_fraction
    records=[{'scene_id':meta['scene_id'],'window_id':i,'size':int(r[3]),'equivalent_count':int(counts[i])} for i,r in enumerate(windows)]
    return {'scene_row':row,'window_rows':records}

def verify_dataset(root,scene=None,*,full_index=False,physical=False):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if manifest['format']!='RIND3D' or manifest['version'] not in (1,2) or manifest['config_sha256']!=config_hash(manifest['config']):raise ValueError('Invalid dataset manifest/config hash')
    from .profiles import validate_manifest_profile,LITE
    profile=validate_manifest_profile(manifest)
    index={r['scene_id']:r for r in pq.read_table(root/'scenes.parquet').to_pylist()}
    splits={r['scene_id']:r['split'] for r in pq.read_table(root/'splits.parquet').to_pylist()}
    if profile.name==LITE:
        groups={}
        for row in index.values():
            gid=row.get('geometry_id')
            if not gid or (gid in groups and groups[gid]!=row['split']):raise ValueError('Geometry split leakage')
            groups[gid]=row['split']
    paths=[root/'scenes'/scene] if scene else sorted(p for p in (root/'scenes').iterdir() if p.is_dir())
    rows=[]
    for path in paths:
        if path.parent!=root/'scenes':raise ValueError('Invalid scene path')
        checked=verify_scene(path,full_index=full_index,physical=physical);row=checked['scene_row']
        meta=json.loads((path/'verification.json').read_text())
        if meta['config_sha256']!=manifest['config_sha256'] or index.get(path.name)!=row or splits.get(path.name)!=row['split']:raise ValueError('Canonical scene and dataset indices disagree')
        actual=pq.read_table(root/'windows.parquet',filters=[('scene_id','=',path.name)]).to_pylist()
        if actual!=checked['window_rows']:raise ValueError('Queryable windows disagree with canonical scene records')
        rows.append(row)
    if not scene and set(index)!={p.name for p in paths}:raise ValueError('Scene index differs from published directories')
    return rows
