"""Prepare normalized-max multi-layer teachers and train the Phase-I energy field."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import queue
import random
import sys
import time
import numpy as np

HERE=Path(__file__).resolve(); REPO=HERE.parents[3]
PHASE1=REPO/'Phase1_Single_Source_Single_Intensity'; PHASE2=REPO/'Phase2_Multi_Source'
for item in (PHASE1/'src',PHASE2/'src',PHASE2/'vendor/rind-dataset',HERE.parent):
    if str(item) not in sys.path: sys.path.insert(0,str(item))

from layered_residual import combine_layer_probabilities,recover,save_run
from rind_phase2.data import Phase2Dataset


def _json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _atomic_checkpoint(path,value):
    import torch
    path=Path(path);temp=path.with_suffix('.tmp.pt')
    torch.save(value,temp);temp.replace(path)


def _progress(root,event,message,**fields):
    """Print one readable status line and append its structured form on disk."""
    stamp=time.strftime('%Y-%m-%d %H:%M:%S')
    row={'time':stamp,'event':event,**fields}
    print(f'[{stamp}] {message}',flush=True)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n')


def read_config(path):
    c=json.loads(Path(path).read_text()); required={'data_root','output_root','split_seed','scene_counts','record_counts','window_sizes','teacher','training'}
    if not required <= set(c): raise ValueError(f'missing config fields: {sorted(required-set(c))}')
    if set(c['scene_counts'])!={'train','validation','test'} or any(type(x) is not int or x<1 for x in c['scene_counts'].values()):
        raise ValueError('scene_counts requires positive train/validation/test counts')
    if (set(c['record_counts'])!={'train','validation','test'} or
            any(not ((type(x) is int and x>=1) or x=='all') for x in c['record_counts'].values())):
        raise ValueError('record_counts requires positive integers or "all" for train/validation/test')
    if not c['window_sizes'] or len(set(c['window_sizes']))!=len(c['window_sizes']) or any(x not in (16,32,64,128,256,512) for x in c['window_sizes']):
        raise ValueError('invalid window_sizes')
    if type(c['split_seed']) is not int or c['split_seed']<0: raise ValueError('split_seed must be nonnegative')
    t=c['training']
    if c['teacher'].get('combined_rule') != 'normalized_elementwise_max':
        raise ValueError('teacher.combined_rule must be normalized_elementwise_max')
    if t.get('normalization_support','geometry') not in ('geometry','world'):
        raise ValueError('normalization_support must be geometry or world')
    for key in ('epochs','patience','batch_size','hidden','num_threads','candidate_chunk'):
        if type(t[key]) is not int or t[key]<1: raise ValueError(f'{key} must be positive integer')
    if t.get('architecture','baseline') not in ('baseline','large'):
        raise ValueError('training.architecture must be baseline or large')
    heads=t.get('attention_heads',1)
    if type(heads) is not int or heads<1 or t['hidden']%heads:
        raise ValueError('training.attention_heads must be a positive divisor of hidden')
    if t.get('architecture','baseline')=='large' and t['hidden']<64:
        raise ValueError('large architecture requires hidden >= 64')
    if type(t.get('progress_every_batches',10)) is not int or t.get('progress_every_batches',10)<1:
        raise ValueError('progress_every_batches must be positive integer')
    if type(t.get('prefetch_batches',0)) is not int or t.get('prefetch_batches',0)<0:
        raise ValueError('training.prefetch_batches must be a nonnegative integer')
    if type(t['fourier_frequencies']) is not int or t['fourier_frequencies']<0:
        raise ValueError('fourier_frequencies must be nonnegative integer')
    if t.get('amp_dtype','none') not in ('none','bfloat16','float16'):
        raise ValueError('training.amp_dtype must be none, bfloat16 or float16')
    for key,default in (('deterministic_algorithms',True),('cudnn_benchmark',False),
                        ('fused_optimizer',False)):
        if type(t.get(key,default)) is not bool:
            raise ValueError(f'training.{key} must be an explicit boolean')
    for key in ('learning_rate','grad_clip'):
        if not np.isfinite(t[key]) or t[key]<=0: raise ValueError(f'{key} must be positive')
    if not np.isfinite(t['weight_decay']) or t['weight_decay']<0: raise ValueError('weight_decay must be nonnegative')
    if t.get('device') not in ('cpu','cuda'): raise ValueError('device must be cpu or cuda')
    if any(type(t.get(key)) is not bool for key in ('use_obstacle','pixel_coordinates')):
        raise ValueError('use_obstacle and pixel_coordinates must be explicit booleans')
    if not t['seeds'] or len(set(t['seeds']))!=len(t['seeds']) or any(type(x) is not int or x<0 for x in t['seeds']):
        raise ValueError('training seeds must be distinct nonnegative integers')
    teacher=c['teacher']
    if teacher.get('backend','auto') not in ('auto','cuda','numba','reference'):
        raise ValueError('teacher.backend must be auto, cuda, numba or reference')
    if type(teacher.get('shadow_screening')) is not bool:
        raise ValueError('teacher.shadow_screening must be an explicit boolean')
    if type(teacher.get('save_plots',True)) is not bool:
        raise ValueError('teacher.save_plots must be an explicit boolean when provided')
    if type(teacher.get('max_layers',32)) is not int or teacher.get('max_layers',32)<1:
        raise ValueError('teacher.max_layers must be a positive integer')
    for key,default in (('gpu_batch_size',64),('gpu_pixel_chunk',4096)):
        if type(teacher.get(key,default)) is not int or teacher.get(key,default)<1:
            raise ValueError(f'teacher.{key} must be a positive integer')
    gpu_min_default=min(1024,teacher.get('gpu_batch_size',64))
    if (type(teacher.get('gpu_min_batch_size',gpu_min_default)) is not int or
            teacher.get('gpu_min_batch_size',gpu_min_default)<1 or
            teacher.get('gpu_min_batch_size',gpu_min_default)>teacher.get('gpu_batch_size',64)):
        raise ValueError('teacher.gpu_min_batch_size must be positive and <= gpu_batch_size')
    if type(teacher.get('gpu_pair_budget',134217728)) is not int or teacher.get('gpu_pair_budget',134217728)<1:
        raise ValueError('teacher.gpu_pair_budget must be a positive integer')
    if (type(teacher.get('gpu_verify_candidates',2)) is not int or
            teacher.get('gpu_verify_candidates',2)<0):
        raise ValueError('teacher.gpu_verify_candidates must be a nonnegative integer')
    devices=teacher.get('gpu_devices',[0])
    if (not isinstance(devices,list) or not devices or
            any(type(device) is not int or device<0 for device in devices) or
            len(set(devices))!=len(devices)):
        raise ValueError('teacher.gpu_devices must be a nonempty list of distinct nonnegative integers')
    if type(teacher.get('parallel_window_prefetch',4)) is not int or teacher.get('parallel_window_prefetch',4)<1:
        raise ValueError('teacher.parallel_window_prefetch must be a positive integer')
    for key in ('spacing','temperature','boundary_sigma'):
        if not np.isfinite(teacher[key]) or teacher[key]<=0: raise ValueError(f'{key} must be positive')
    for key in ('alpha','beta','boundary_lambda'):
        if not np.isfinite(teacher[key]) or teacher[key]<0: raise ValueError(f'{key} must be nonnegative')
    if teacher.get('backend','auto')=='cuda' and teacher.get('beta',0.)!=0:
        raise ValueError('teacher.backend=cuda currently requires beta=0 for the GPU-only candidate pipeline')
    return c


def context(c):
    data=Path(c['data_root']); output=Path(c['output_root'])
    if not data.is_absolute(): data=PHASE2/data
    if not output.is_absolute(): output=PHASE2/output
    dataset=Phase2Dataset(data)
    world=int(dataset.manifest['global_size']);spacing=c['teacher']['spacing']
    if not float(spacing).is_integer() or world%int(spacing):
        raise ValueError('teacher spacing must be an integer divisor of world size')
    from rind_phase1.sampling import require_resolved_sampling
    require_resolved_sampling(world,spacing,c['training']['fourier_frequencies'])
    return dataset,output


def _window_class(dataset,scene,view,tolerance):
    o=dataset.get_observation(scene,view);x,y,L=map(int,o['window'])
    response=np.asarray(dataset.get_region(scene,x,y,L)['response'],dtype=np.float64)
    free=response[~o['obstacle']]
    if not len(free): return 'no_free_pixels'
    if np.all(free>tolerance): return 'all_lit'
    if not np.any(free>tolerance): return 'all_dark'
    return 'eligible'


def _protocol(c,dataset,output):
    total=sum(c['scene_counts'].values())
    if total>dataset.num_scenes: raise ValueError('scene counts exceed dataset')
    rng=np.random.default_rng(c['split_seed']);order=rng.permutation(dataset.num_scenes)
    splits={};offset=0
    for name in ('train','validation','test'):
        count=c['scene_counts'][name];splits[name]=order[offset:offset+count].astype(int).tolist();offset+=count
    source_files=(HERE.parent/'layered_residual.py',HERE.parent/'train_combined.py',
                  PHASE1/'src/rind_phase1/model.py',PHASE1/'src/rind_phase1/train.py',
                  PHASE2/'src/rind_phase2/regions.py',PHASE2/'src/rind_phase2/physics.py',
                  PHASE2/'src/rind_phase2/gpu_visibility.py')
    hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    protocol={'config':c,'config_sha256':_digest(c),'scene_splits':splits,
              'selection':'scene split first; deterministically shuffle real windows within each requested size and scan sizes round-robin; integer record_count stops at that many completed teachers, while "all" exhausts every real window in the split; filtered and failed windows do not count',
              'teacher_target':'normalized elementwise maximum of all recovered layer distributions',
              'teacher_execution':'decompose and peel every residual layer first; after complete reconstruction, traverse one shared uniform full-field candidate grid once and compare each rendered visibility mask with all layers',
              'physical_sampling_count':'one uniform candidate visibility pass per completed observation; layer count changes score comparisons but not physical render passes',
              'student_input':'original response, obstacle option, window and candidate coordinate; no residual/layer inputs',
              'source_sha256':hashes}
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('output_root contains another protocol; choose a new output_root')
    _json(path,protocol);return protocol


def _compute_teacher(dataset,output,teacher,split,scene,view,size,gpu_device):
    """Compute and atomically save one independent teacher window."""
    directory=output/'teachers'/split/f'scene{scene}-view{view}'
    started=time.perf_counter()
    result=recover(
        dataset,scene,view,spacing=teacher['spacing'],temperature=teacher['temperature'],
        tolerance=float(teacher.get('tolerance',1e-12)),alpha=teacher.get('alpha',1.),
        beta=teacher.get('beta',0.),boundary_lambda=teacher.get('boundary_lambda',0.),
        boundary_sigma=teacher.get('boundary_sigma',1.),max_layers=teacher.get('max_layers',32),
        backend=teacher.get('backend','auto'),shadow_screening=teacher['shadow_screening'],
        gpu_batch_size=teacher.get('gpu_batch_size',64),
        gpu_pixel_chunk=teacher.get('gpu_pixel_chunk',4096),
        gpu_verify_candidates=teacher.get('gpu_verify_candidates',2),gpu_device=gpu_device,
        gpu_min_batch_size=teacher.get('gpu_min_batch_size',min(1024,teacher.get('gpu_batch_size',64))),
        gpu_pair_budget=teacher.get('gpu_pair_budget',134217728))
    report=dict(result[0]);report['window']=dataset.get_observation(scene,view)['window'].tolist()
    report['world_size']=int(dataset.manifest['global_size'])
    save_run(directory,report,*result[1:],plots=teacher.get('save_plots',True))
    return report,time.perf_counter()-started


def prepare(c,dataset,output):
    teacher=c['teacher'];backend=teacher.get('backend','auto')
    device_ids=list(teacher.get('gpu_devices',[0])) if backend=='cuda' else []
    device_names={}
    if backend=='cuda':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('teacher.backend=cuda, but torch.cuda.is_available() is false')
        available=torch.cuda.device_count()
        invalid=[device for device in device_ids if device>=available]
        if invalid:
            raise RuntimeError(f'teacher.gpu_devices contains unavailable devices {invalid}; visible CUDA device count={available}')
        device_names={device:torch.cuda.get_device_name(device) for device in device_ids}

    started=time.perf_counter();protocol=_protocol(c,dataset,output);rng=np.random.default_rng(c['split_seed'])
    tol=float(teacher.get('tolerance',1e-12))
    records={name:[] for name in ('train','validation','test')};audit=[]
    split_statistics={}
    scanned=teacher_attempts=0
    desired=dict(c['record_counts'])
    workers=len(device_ids) if backend=='cuda' else 1
    prefetch=int(teacher.get('parallel_window_prefetch',4))
    _progress(output,'prepare_start',
        f'prepare start | record targets train={desired["train"]}, validation={desired["validation"]}, test={desired["test"]} | "all" scans the complete raw-window pool; numeric raw totals are printed for each split below | completed training records exclude filtered windows and incomplete teachers | spacing={teacher["spacing"]}, temperature={teacher["temperature"]} | teacher backend={backend} | parallel windows={workers}'
        +(f' | GPUs={device_names}' if device_names else ''),
        requested=desired,spacing=teacher['spacing'],temperature=teacher['temperature'],
        teacher_backend=backend,gpu_devices=device_ids,gpu_device_names=device_names,
        parallel_teacher_windows=workers,gpu_batch_size=teacher.get('gpu_batch_size',64),
        gpu_min_batch_size=teacher.get('gpu_min_batch_size',min(1024,teacher.get('gpu_batch_size',64))),
        gpu_pair_budget=teacher.get('gpu_pair_budget',134217728),
        gpu_pixel_chunk=teacher.get('gpu_pixel_chunk',4096),
        gpu_verify_candidates=teacher.get('gpu_verify_candidates',2))

    gpu_tokens=queue.Queue()
    for device in device_ids:gpu_tokens.put(device)

    def execute(item):
        split,scene,view,size=item
        if backend!='cuda':
            return _compute_teacher(dataset,output,teacher,split,scene,view,size,None)
        device=gpu_tokens.get()
        try:
            return _compute_teacher(dataset,output,teacher,split,scene,view,size,f'cuda:{device}')
        finally:
            gpu_tokens.put(device)

    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='rind-teacher') as executor:
        for split,scenes in protocol['scene_splits'].items():
            split_started=time.perf_counter()
            pools={int(size):[] for size in c['window_sizes']}
            for scene in scenes:
                views=dataset._base.local_views[scene,:int(dataset._base.view_counts[scene])]
                for size in pools:
                    pools[size].extend((split,int(scene),int(view),size)
                                       for view in np.flatnonzero(views[:,2]==size))
            for pool in pools.values():rng.shuffle(pool)
            raw_by_size={str(size):len(pool) for size,pool in pools.items()}
            raw_total=sum(raw_by_size.values())
            exhaustive=desired[split]=='all'
            target_text=('all raw windows; final completed-record count is determined after filtering'
                         if exhaustive else f'{desired[split]} completed training records')
            _progress(output,'prepare_split_start',
                      f'prepare {split} inventory | scenes={len(scenes)}, raw windows={raw_total}, by size={raw_by_size} | target={target_text}',
                      split=split,scenes=len(scenes),requested=desired[split],
                      exhaustive_scan=exhaustive,raw_windows=raw_total,
                      raw_windows_by_size=raw_by_size)
            offsets={size:0 for size in pools}
            split_scanned=split_eligible=split_incomplete=0
            split_reused=split_computed=0
            split_skips={key:0 for key in ('all_lit','all_dark','no_free_pixels')}

            def ordered_windows():
                while True:
                    progressed=False
                    for size,pool in pools.items():
                        offset=offsets[size]
                        if offset<len(pool):
                            progressed=True;offsets[size]=offset+1;yield pool[offset]
                    if not progressed:return

            candidates=iter(ordered_windows());exhausted=False
            while not exhausted and (exhaustive or len(records[split])<desired[split]):
                remaining=(workers*prefetch if exhaustive else
                           min(workers*prefetch,desired[split]-len(records[split])))
                batch=[]
                while len(batch)<remaining:
                    try:item=next(candidates)
                    except StopIteration:
                        exhausted=True;break
                    _,scene,view,size=item;scanned+=1;split_scanned+=1
                    kind=_window_class(dataset,scene,view,tol)
                    if kind!='eligible':
                        split_skips[kind]+=1
                        audit.append({'split':split,'scene_id':scene,'view_id':view,'size':size,
                                      'status':'skipped_'+kind})
                    else:
                        split_eligible+=1
                        batch.append(item)
                    if split_scanned%100==0:
                        _progress(output,'prepare_scan',
                            f'prepare {split} scan | raw checked={split_scanned}/{raw_total} ({100*split_scanned/max(raw_total,1):.1f}%) | eligible seen={split_eligible}, completed training records={len(records[split])}, incomplete teachers={split_incomplete}, skips={split_skips}',
                            split=split,checked=split_scanned,raw_windows=raw_total,
                            percent=100*split_scanned/max(raw_total,1),eligible_seen=split_eligible,
                            completed_records=len(records[split]),incomplete_teachers=split_incomplete,
                            skips=dict(split_skips))
                if not batch:continue

                results={};missing=[]
                for item in batch:
                    _,scene,view,size=item
                    directory=output/'teachers'/split/f'scene{scene}-view{view}'
                    existing=directory/'report.json'
                    reusable=False
                    if existing.exists():
                        report=json.loads(existing.read_text())
                        reusable=(report.get('status')!='complete' or (directory/'combined.npz').exists())
                    if reusable:
                        split_reused+=1
                        results[item]=(report,0.,True)
                    else:
                        split_computed+=1
                        teacher_attempts+=1
                        results[item]=teacher_attempts
                        missing.append(item)
                        _progress(output,'teacher_start',
                            f'teacher window {teacher_attempts} queued | {split} scene={scene} view={view} size={size} | raw checked={split_scanned}/{raw_total}, eligible seen={split_eligible}, completed training records={len(records[split])}',
                            attempt=teacher_attempts,split=split,scene_id=scene,view_id=view,
                            size=size,selected=len(records[split]),requested=desired[split],
                            checked=split_scanned,raw_windows=raw_total,eligible_seen=split_eligible)

                computed=list(executor.map(execute,missing))
                for item,(report,seconds) in zip(missing,computed):
                    results[item]=(report,seconds,False)

                for item in batch:
                    _,scene,view,size=item;report,seconds,reused=results[item]
                    if reused:
                        _progress(output,'teacher_reused',
                            f'teacher reused | {split} scene={scene} view={view} size={size} status={report["status"]} layers={report["layers_recovered"]}',
                            split=split,scene_id=scene,view_id=view,size=size,
                            status=report['status'],layers=report['layers_recovered'])
                    else:
                        _progress(output,'teacher_done',
                            f'teacher done | {split} scene={scene} view={view} | status={report["status"]}, layers={report["layers_recovered"]}/{report["layers_attempted"]}, MAE={report["final_mae"]:.6g}, {seconds:.1f}s, backend={report.get("teacher_backend",backend)}, device={report.get("gpu_device")}, gpu batches={report.get("gpu_render_batches",0)}, effective batch={report.get("gpu_effective_batch_size")}, OOM retries={report.get("gpu_oom_retries",0)}'
                            +(f', reason={report.get("failure_reason")}' if report['status']!='complete' else ''),
                            split=split,scene_id=scene,view_id=view,size=size,status=report['status'],
                            layers_recovered=report['layers_recovered'],layers_attempted=report['layers_attempted'],
                            final_mae=report['final_mae'],failure_reason=report.get('failure_reason'),
                            seconds=seconds,teacher_backend=report.get('teacher_backend',backend),
                            gpu_device=report.get('gpu_device'),gpu_render_batches=report.get('gpu_render_batches',0),
                            gpu_requested_batch_size=report.get('gpu_requested_batch_size'),
                            gpu_effective_batch_size=report.get('gpu_effective_batch_size'),
                            gpu_oom_retries=report.get('gpu_oom_retries',0),
                            gpu_validation_cpu_renders=report.get('gpu_validation_cpu_renders',0),
                            gpu_scored_candidate_layers=report.get('gpu_scored_candidate_layers',0))
                    directory=output/'teachers'/split/f'scene{scene}-view{view}'
                    row={'split':split,'scene_id':scene,'view_id':view,'size':size,
                         'status':report['status'],'failure_reason':report.get('failure_reason'),
                         'layers_recovered':report['layers_recovered'],
                         'directory':str(directory.relative_to(output))}
                    audit.append(row)
                    if report['status']=='complete' and report['layers_recovered']>0:
                        records[split].append(row)
                    else:
                        split_incomplete+=1

            if not exhaustive and len(records[split])<desired[split]:
                audit.append({'split':split,'status':'record_quota_unfilled',
                              'selected':len(records[split]),'requested':desired[split],
                              'candidate_windows':sum(map(len,pools.values()))})
            split_statistics[split]={
                'scene_count':len(scenes),'requested_completed_records':desired[split],
                'exhaustive_scan':exhaustive,'raw_windows':raw_total,
                'raw_windows_by_size':raw_by_size,'checked_windows':split_scanned,
                'eligible_windows_seen':split_eligible,
                'completed_training_records':len(records[split]),
                'incomplete_teachers':split_incomplete,
                'reused_teacher_reports':split_reused,'computed_teacher_reports':split_computed,
                'skipped_windows':dict(split_skips)}
            _progress(output,'prepare_split_done',
                f'prepare {split} done | raw checked={split_scanned}/{raw_total}, eligible seen={split_eligible}, completed training records={len(records[split])}, incomplete teachers={split_incomplete}, skips={split_skips}, elapsed={time.perf_counter()-split_started:.1f}s',
                split=split,selected=len(records[split]),requested=desired[split],
                **split_statistics[split],seconds=time.perf_counter()-split_started)

    manifest={'target':'normalized_elementwise_max','records':records,'audit':audit,
              'requested_counts':desired,'counts':{k:len(v) for k,v in records.items()},
              'split_statistics':split_statistics,
              'skip_counts':{kind:sum(x['status']==kind for x in audit) for kind in
                             ('skipped_all_lit','skipped_all_dark','skipped_no_free_pixels')}}
    _json(output/'records.json',manifest)
    failures=sum(x.get('status')=='incomplete' for x in audit)
    raw_counts={name:stats['raw_windows'] for name,stats in split_statistics.items()}
    _progress(output,'prepare_done',
        f'prepare done | exact completed training records={manifest["counts"]}, raw windows={raw_counts}, skips={manifest["skip_counts"]}, incomplete teachers={failures}, elapsed={time.perf_counter()-started:.1f}s',
        counts=manifest['counts'],skip_counts=manifest['skip_counts'],incomplete_teachers=failures,
        checked_windows=scanned,teacher_attempts=teacher_attempts,seconds=time.perf_counter()-started)
    short={split:{'selected':len(records[split]),'requested':desired[split]}
           for split in records if desired[split]!='all' and len(records[split])!=desired[split]}
    if short:
        raise ValueError(f'completed record quota not reached: {short}; enlarge scene_counts or adjust window_sizes')
    return manifest

class CombinedRecords:
    def __init__(self,entries,dataset,root,normalization_support='geometry',response_tolerance=1e-12):
        self.entries=list(entries);self.dataset=dataset;self.root=Path(root)
        self.normalization_support=normalization_support
        self.response_tolerance=float(response_tolerance)
        self.window_sizes=[int(x['size']) for x in self.entries]
        self.scene_ids={int(x['scene_id']) for x in self.entries}
        # These invariants do not change between epochs.  Validate once here
        # instead of reopening report.json and rerendering the window for every
        # sample in every epoch.
        for row in self.entries:
            directory=self.root/row['directory']
            report=json.loads((directory/'report.json').read_text())
            if report['status']!='complete' or report.get('partial_combined_distribution'):
                raise ValueError('training requires a complete teacher book')
            kind=_window_class(self.dataset,row['scene_id'],row['view_id'],self.response_tolerance)
            if kind!='eligible':
                raise ValueError(f'{kind.replace("_","-")} window leaked into training records')
    def __len__(self): return len(self.entries)
    def __getitem__(self,index):
        row=self.entries[index];directory=self.root/row['directory']
        observation=self.dataset.get_observation(row['scene_id'],row['view_id'])
        report=json.loads((directory/'report.json').read_text())
        probabilities=[];candidate=valid=None
        for detail in report.get('layers',[]):
            path=directory/f"layer-{int(detail['step']):02d}.npz"
            with np.load(path,allow_pickle=False) as layer:
                if 'probability' not in layer.files:
                    continue
                current_probability=np.asarray(layer['probability'],dtype=np.float64)
                current_candidate=np.asarray(layer['candidate_xy'],dtype=np.float64)
                current_valid=np.asarray(layer['valid'],dtype=bool)
            if candidate is None:
                candidate=current_candidate;valid=current_valid
            elif (not np.array_equal(candidate,current_candidate) or
                  not np.array_equal(valid,current_valid)):
                raise ValueError('saved layers do not share one candidate grid')
            probabilities.append(current_probability)
        if not probabilities:
            raise ValueError('complete teacher book has no saved layer probabilities')
        probability=combine_layer_probabilities(probabilities)
        if probability.shape!=(len(candidate),) or valid.shape!=probability.shape or not np.isclose(probability.sum(),1,atol=1e-12):
            raise ValueError('malformed combined teacher target')
        support=valid if self.normalization_support=='geometry' else np.ones_like(valid)
        return {'response':observation['response'],'obstacle':observation['obstacle'],'window':observation['window'],
                'candidate_xy':candidate,'valid':valid,'support':support,'teacher_prob':probability,
                'scene_id':row['scene_id'],'view_id':row['view_id']}


def train(c,dataset,output):
    import torch
    from rind_phase1.interfaces import INTERFACE_VERSION
    from rind_phase1.model import SourceEnergyField
    from rind_phase1.train import run_epoch,load_model
    manifest=json.loads((output/'records.json').read_text());t=c['training']
    actual={name:int(manifest.get('counts',{}).get(name,-1))
            for name in ('train','validation','test')}
    required=dict(c['record_counts'])
    mismatched={name:{'required':required[name],'found':actual[name]} for name in actual
                if ((required[name]=='all' and actual[name]<1) or
                    (required[name]!='all' and actual[name]!=required[name]))}
    if mismatched:
        raise ValueError(f'training record requirements are not satisfied: {mismatched}')
    torch.set_num_threads(t['num_threads'])
    support=t.get('normalization_support','geometry')
    response_tolerance=float(c['teacher'].get('tolerance',1e-12))
    splits={name:CombinedRecords(manifest['records'][name],dataset,output,support,response_tolerance)
            for name in ('train','validation','test')}
    if splits['train'].scene_ids & splits['validation'].scene_ids or splits['train'].scene_ids & splits['test'].scene_ids or splits['validation'].scene_ids & splits['test'].scene_ids:
        raise ValueError('scene leakage between splits')
    device=torch.device(t.get('device','cpu'))
    if device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    training_gpu_ids=(list(c['teacher'].get('gpu_devices',[0])) if device.type=='cuda' else [])
    if training_gpu_ids and max(training_gpu_ids)>=torch.cuda.device_count():
        raise ValueError(f'training requested GPUs {training_gpu_ids}, but only {torch.cuda.device_count()} are visible')
    if training_gpu_ids:
        device=torch.device(f'cuda:{training_gpu_ids[0]}')
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=t.get('cudnn_benchmark',False)
    all_reports={}
    for seed in t['seeds']:
        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        if training_gpu_ids:torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(t.get('deterministic_algorithms',True))
        rng=np.random.default_rng(seed)
        model_config={'hidden':t['hidden'],'fourier_frequencies':t['fourier_frequencies'],
                      'coordinate_scale':float(dataset.manifest['global_size']),
                      'use_obstacle':t.get('use_obstacle',True),'pixel_coordinates':t.get('pixel_coordinates',False),
                      'normalization_support':support,
                      'architecture':t.get('architecture','baseline'),
                      'attention_heads':t.get('attention_heads',1)}
        base_model=SourceEnergyField(**model_config).to(device)
        model=(torch.nn.DataParallel(base_model,device_ids=training_gpu_ids,
                                     output_device=training_gpu_ids[0])
               if len(training_gpu_ids)>1 else base_model)
        fused=bool(t.get('fused_optimizer',False) and device.type=='cuda')
        optimizer=torch.optim.AdamW(model.parameters(),lr=t['learning_rate'],
                                    weight_decay=t['weight_decay'],fused=fused)
        directory=output/'models'/f'seed_{seed}';directory.mkdir(parents=True,exist_ok=True)
        protocol=json.loads((output/'protocol.json').read_text())
        signature=_digest({'protocol':_digest(protocol),'records':_digest(manifest),'seed':seed})
        history=[];best=float('inf');bad=0;start=0;steps=0;started=time.perf_counter()
        if (directory/'last.pt').exists():
            checkpoint=torch.load(directory/'last.pt',map_location='cpu',weights_only=True)
            if checkpoint.get('run_signature')!=signature:
                raise ValueError('resume checkpoint protocol/targets/seed mismatch')
            base_model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
            torch.set_rng_state(checkpoint['torch_rng']);rng.bit_generator.state=checkpoint['numpy_rng']
            history=checkpoint['history'];best=checkpoint['best_validation_ce'];bad=checkpoint['bad_epochs']
            start=checkpoint['epoch'];steps=checkpoint['global_step']
        epoch_steps=sum(int(np.ceil(sum(size==s for size in splits['train'].window_sizes)/t['batch_size']))
                        for s in sorted(set(splits['train'].window_sizes)))
        validation_steps=sum(int(np.ceil(sum(size==s for size in splits['validation'].window_sizes)/t['batch_size']))
                             for s in sorted(set(splits['validation'].window_sizes)))
        test_steps=sum(int(np.ceil(sum(size==s for size in splits['test'].window_sizes)/t['batch_size']))
                       for s in sorted(set(splits['test'].window_sizes)))
        parameters=sum(p.numel() for p in base_model.parameters())
        _progress(output,'train_start',
            f'train seed={seed} | records train/validation/test={len(splits["train"])}/{len(splits["validation"])}/{len(splits["test"])} | architecture={model_config["architecture"]}, hidden={t["hidden"]}, heads={model_config["attention_heads"]}, parameters={parameters:,}, AMP={t.get("amp_dtype","none")}, fused AdamW={fused} | GPUs={training_gpu_ids or ["cpu"]} | resume epoch={start}',
            seed=seed,train_records=len(splits['train']),validation_records=len(splits['validation']),
            test_records=len(splits['test']),model_parameters=parameters,resume_epoch=start,
            epochs=t['epochs'],batch_size=t['batch_size'],learning_rate=t['learning_rate'],
            weight_decay=t['weight_decay'],training_gpu_ids=training_gpu_ids,
            architecture=model_config['architecture'],hidden=t['hidden'],
            attention_heads=model_config['attention_heads'],fourier_frequencies=t['fourier_frequencies'],
            amp_dtype=t.get('amp_dtype','none'),fused_optimizer=fused,
            deterministic_algorithms=t.get('deterministic_algorithms',True),
            cudnn_benchmark=t.get('cudnn_benchmark',False),
            prefetch_batches=t.get('prefetch_batches',0))
        # DataParallel needs the complete forward call so it can scatter the
        # native-size batch before observation encoding and candidate scoring.
        epoch_candidate_chunk=(None if len(training_gpu_ids)>1 else t['candidate_chunk'])
        every=t.get('progress_every_batches',10)
        def batch_progress(phase,epoch,total_batches):
            def report(batch_index,batch_records,running_ce):
                if batch_index==1 or batch_index%every==0 or batch_index==total_batches:
                    _progress(output,'batch_progress',
                        f'{phase} | seed={seed}, epoch={epoch}/{t["epochs"]}, batch={batch_index}/{total_batches}, batch records={batch_records}, running CE={running_ce:.6f}',
                        phase=phase,seed=seed,epoch=epoch,total_epochs=t['epochs'],batch=batch_index,
                        total_batches=total_batches,batch_records=batch_records,running_cross_entropy=running_ce)
            return report
        for epoch in range(start+1,t['epochs']+1):
            if bad>=t['patience']:break
            tick=time.perf_counter()
            _progress(output,'epoch_start',f'epoch {epoch}/{t["epochs"]} start | seed={seed}',seed=seed,epoch=epoch,total_epochs=t['epochs'])
            train_ce=run_epoch(model,splits['train'],t['batch_size'],optimizer,rng,t['grad_clip'],epoch_candidate_chunk,
                               batch_progress('train',epoch,epoch_steps),amp_dtype=t.get('amp_dtype','none'),
                               prefetch_batches=t.get('prefetch_batches',0))
            val_ce=run_epoch(model,splits['validation'],t['batch_size'],candidate_chunk=epoch_candidate_chunk,
                             progress_callback=batch_progress('validation',epoch,validation_steps),
                             amp_dtype=t.get('amp_dtype','none'),prefetch_batches=t.get('prefetch_batches',0))
            steps+=epoch_steps;improved=val_ce<best
            best=min(best,val_ce);bad=0 if improved else bad+1
            history.append({'epoch':epoch,'train_cross_entropy':train_ce,'validation_cross_entropy':val_ce,
                            'seconds':time.perf_counter()-tick})
            common={'interface_version':INTERFACE_VERSION,'model_config':model_config,'model':base_model.state_dict(),
                    'epoch':epoch,'seed':seed,'records_manifest':str((output/'records.json').resolve()),
                    'target_semantics':'normalized elementwise maximum across complete recovered layer distributions',
                    'scene_splits':{k:sorted(v.scene_ids) for k,v in splits.items()},
                    'optimizer':optimizer.state_dict(),'history':history,'best_validation_ce':best,
                    'bad_epochs':bad,'global_step':steps,'run_signature':signature,
                    'torch_rng':torch.get_rng_state(),'numpy_rng':rng.bit_generator.state,
                    'software':{'torch':str(torch.__version__),'numpy':np.__version__,'device':str(device),
                                'gpu_ids':training_gpu_ids,'parallelism':('data_parallel' if len(training_gpu_ids)>1 else 'single_device'),
                                'num_threads':t['num_threads'],'amp_dtype':t.get('amp_dtype','none'),
                                'fused_optimizer':fused,
                                'deterministic_algorithms':t.get('deterministic_algorithms',True),
                                'cudnn_benchmark':t.get('cudnn_benchmark',False),
                                'prefetch_batches':t.get('prefetch_batches',0)}}
            if improved:_atomic_checkpoint(directory/'best.pt',common)
            _atomic_checkpoint(directory/'last.pt',common)
            _json(directory/'training.json',{'seed':seed,'history':history,'best_validation_cross_entropy':best,
                  'selection':'minimum validation cross entropy; test evaluated after training'})
            _progress(output,'epoch_done',
                f'epoch {epoch}/{t["epochs"]} done | train CE={train_ce:.6f}, validation CE={val_ce:.6f}, best={best:.6f}, no-improvement={bad}/{t["patience"]}, {history[-1]["seconds"]:.1f}s',
                seed=seed,epoch=epoch,total_epochs=t['epochs'],train_cross_entropy=train_ce,
                validation_cross_entropy=val_ce,best_validation_cross_entropy=best,bad_epochs=bad,
                patience=t['patience'],seconds=history[-1]['seconds'])
        if not (directory/'best.pt').exists(): raise ValueError('training produced no checkpoint')
        best_model,checkpoint=load_model(directory/'best.pt');best_model.to(device)
        test_model=(torch.nn.DataParallel(best_model,device_ids=training_gpu_ids,
                                          output_device=training_gpu_ids[0])
                    if len(training_gpu_ids)>1 else best_model)
        _progress(output,'test_start',f'test start | seed={seed}, best epoch={checkpoint["epoch"]}',seed=seed,best_epoch=checkpoint['epoch'])
        test_ce=run_epoch(test_model,splits['test'],t['batch_size'],candidate_chunk=epoch_candidate_chunk,
                          progress_callback=batch_progress('test',checkpoint['epoch'],test_steps),
                          amp_dtype=t.get('amp_dtype','none'),prefetch_batches=t.get('prefetch_batches',0))
        report={'seed':seed,'history':history,'best_epoch':checkpoint['epoch'],'best_validation_cross_entropy':checkpoint['best_validation_ce'],
                'test_cross_entropy':test_ce,'elapsed_seconds':time.perf_counter()-started,
                'records':manifest['counts'],'all_lit_windows_skipped':manifest['skip_counts']['skipped_all_lit']}
        _json(directory/'training.json',report);all_reports[str(seed)]=report
        _progress(output,'train_done',
            f'train done | seed={seed}, best epoch={checkpoint["epoch"]}, best validation CE={checkpoint["best_validation_ce"]:.6f}, test CE={test_ce:.6f}, elapsed={report["elapsed_seconds"]:.1f}s',
            seed=seed,best_epoch=checkpoint['epoch'],best_validation_cross_entropy=checkpoint['best_validation_ce'],
            test_cross_entropy=test_ce,seconds=report['elapsed_seconds'])
    _json(output/'training_summary.json',all_reports);return all_reports


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','train','all'])
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--gpus',type=int,default=None,
                   help='number of visible CUDA devices for parallel teachers and student training')
    a=p.parse_args(argv)
    c=read_config(a.config)
    if a.gpus is not None:
        if a.gpus<1:raise ValueError('--gpus must be a positive integer')
        if c['teacher'].get('backend','auto')!='cuda':
            raise ValueError('--gpus requires teacher.backend=cuda')
        c['teacher']['gpu_devices']=list(range(a.gpus))
    dataset,output=context(c);output.mkdir(parents=True,exist_ok=True)
    _progress(output,'run_start',f'command={a.command} | config={a.config.resolve()} | output={output} | gpus={a.gpus}',
              command=a.command,config=str(a.config.resolve()),output=str(output),gpus=a.gpus)
    try:
        if a.command in ('prepare','all'):
            m=prepare(c,dataset,output);print('Prepared:',m['counts'],'skips:',m['skip_counts'],flush=True)
        if a.command in ('train','all'):
            train(c,dataset,output);print('Ready:',output/'training_summary.json')
    except Exception as error:
        _progress(output,'run_failed',f'FAILED | {type(error).__name__}: {error}',
                  error_type=type(error).__name__,error=str(error))
        raise
    _progress(output,'run_done',f'command={a.command} completed | output={output}',command=a.command,output=str(output))

if __name__=='__main__':main()
