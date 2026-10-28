"""Reproducible Phase I experiment stages. Run from the repository checkout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import queue
import time

import numpy as np

from .data import PROJECT_ROOT, Phase1Dataset
from .interfaces import INTERFACE_VERSION, training_record
from .teacher import generate_uniform_teacher, load_teacher_record, save_teacher_record


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


def _cache_sources(raw_roots):
    """Resolve prior Phase-I response directories and their original datasets."""
    sources=[]
    for raw in raw_roots:
        root=Path(raw).expanduser()
        if not root.is_absolute():root=PROJECT_ROOT/root
        root=root.resolve()
        if (root/'response/teachers').is_dir():root=root/'response'
        if not (root/'teachers').is_dir():
            sources.append({'root':root,'dataset':None,'reason':'teachers_directory_missing'})
            continue
        data_candidates=[]
        protocol_path=root.parent/'protocol.json'
        if protocol_path.is_file():
            try:
                old_config=json.loads(protocol_path.read_text()).get('config',{})
                old_data=old_config.get('data_root')
                if old_data:data_candidates.append((PROJECT_ROOT/old_data).resolve())
            except (ValueError,OSError):pass
        try:
            first_record=next((root/'teachers').glob('s*_v*.npz'))
            with np.load(first_record,allow_pickle=False) as archive:
                metadata=json.loads(str(archive['metadata_json'].item()))
            stored_root=metadata.get('dataset_root')
            if stored_root:data_candidates.append(Path(stored_root).expanduser().resolve())
        except (StopIteration,ValueError,OSError,KeyError):pass
        # Individual records also carry their absolute installation path, but
        # this protocol path is portable when the old cache remains in repo.
        old_dataset=None
        for data_root in data_candidates:
            try:
                old_dataset=Phase1Dataset(data_root);break
            except (FileNotFoundError,ValueError):pass
        sources.append({'root':root,'dataset':old_dataset,
                        'reason':None if old_dataset is not None else 'original_dataset_missing'})
    return sources


def _same_scene_and_observation(old_dataset,new_dataset,scene,view):
    old=old_dataset.get_observation(scene,view);new=new_dataset.get_observation(scene,view)
    if any(not np.array_equal(old[key],new[key]) for key in ('window','response','obstacle')):
        return False
    old_scene=old_dataset.get_scene(scene);new_scene=new_dataset.get_scene(scene)
    if int(old_scene['obstacle_count'])!=int(new_scene['obstacle_count']):return False
    return all(np.array_equal(old_scene[key],new_scene[key])
               for key in ('obstacle_types','obstacle_params','obstacle_group_ids'))


def _import_compatible_teacher(cache_sources,dataset,scene,view,settings,protocol_digest):
    """Import exact old costs only when data, grid and physics are identical."""
    if not cache_sources:return None,[]
    from .search import CandidateDomain,uniform_grid
    sample=dataset.get_observation(scene,view)
    expected_grid=uniform_grid(CandidateDomain(tuple(sample['window']),
                                                int(dataset.manifest['global_size']),
                                                settings.get('quadtree_prior',False)),
                               settings['spacing'])
    wanted_physics={
        'alpha':float(settings.get('alpha',1.0)),'beta':float(settings.get('beta',0.0)),
        'boundary_lambda':float(settings.get('boundary_lambda',0.0)),
        'boundary_sigma':float(settings.get('boundary_sigma',1.0)),'source_intensity':1.0}
    rejections=[]
    for source in cache_sources:
        path=source['root']/'teachers'/f's{scene}_v{view}.npz'
        if not path.is_file():continue
        if source['dataset'] is None:
            rejections.append({'path':str(path),'reason':source['reason']});continue
        try:
            if not _same_scene_and_observation(source['dataset'],dataset,scene,view):
                rejections.append({'path':str(path),'reason':'scene_or_observation_differs'});continue
            record=load_teacher_record(path)
            physics=record['metadata']['settings']['physics']
            if any(not np.isclose(float(physics.get(key,np.nan)),value,rtol=0,atol=1e-12)
                   for key,value in wanted_physics.items()):
                rejections.append({'path':str(path),'reason':'physics_settings_differ'});continue
            if not np.isclose(float(record['temperature']),float(settings['temperature']),rtol=0,atol=1e-12):
                rejections.append({'path':str(path),'reason':'temperature_differs'});continue
            if (not np.array_equal(record['candidate_xy'],expected_grid.candidate_xy) or
                    not np.array_equal(record['grid_index'],expected_grid.grid_index)):
                rejections.append({'path':str(path),'reason':'candidate_grid_differs'});continue
            training_record(record,sample)
            record['metadata']['imported_from']=str(path)
            record['metadata']['experiment_protocol']=protocol_digest
            record['metadata']['dataset_root']=str(dataset.root)
            return record,rejections
        except (ValueError,KeyError,OSError,IndexError) as error:
            rejections.append({'path':str(path),'reason':f'{type(error).__name__}: {error}'})
    return None,rejections


def source_hashes():
    paths = list((PROJECT_ROOT/'src/rind_phase1').glob('*.py'))
    paths += list((PROJECT_ROOT/'part_e').glob('*.py'))
    return {str(p.relative_to(PROJECT_ROOT)):file_hash(p) for p in sorted(paths)}


def read_config(path):
    c = json.loads(Path(path).read_text())
    counts = c['scene_counts']
    if set(counts) != {'train','validation','test'} or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError('three positive scene counts required')
    sizes = c['window_sizes']
    if not sizes or len(set(sizes)) != len(sizes) or any(x not in (16,32,64,128,256,512) for x in sizes):
        raise ValueError('unique native window sizes required')
    if type(c['views_per_size']) is not int or not 1 <= c['views_per_size'] <= 3:
        raise ValueError('views_per_size must be 1..3')
    t = c['training']
    if not t['seeds'] or len(set(t['seeds'])) != len(t['seeds']):
        raise ValueError('distinct training seeds required')
    for k in ('epochs','patience','batch_size','num_threads','hidden'):
        if type(t[k]) is not int or t[k] <= 0:
            raise ValueError(f'positive integer {k} required')
    for k in ('learning_rate','grad_clip'):
        if not np.isfinite(t[k]) or t[k] <= 0:
            raise ValueError(f'positive {k} required')
    from .protocol import validate_research_config
    validate_research_config(c)
    architecture=t.get('architecture','baseline')
    if architecture not in ('baseline','large','multiscale_boundary'):
        raise ValueError('unknown training architecture')
    heads=t.get('attention_heads',1)
    if type(heads) is not int or heads<1 or t['hidden']%heads:
        raise ValueError('attention_heads must be a positive divisor of hidden')
    if architecture in ('large','multiscale_boundary') and t['hidden']<64:
        raise ValueError('large architectures require hidden >= 64')
    if architecture=='multiscale_boundary' and (heads!=8 or t['hidden']%16):
        raise ValueError('multiscale_boundary requires attention_heads=8 and hidden divisible by 16')
    devices=t.get('gpu_devices',[0])
    if (not isinstance(devices,list) or not devices or len(set(devices))!=len(devices) or
            any(type(device) is not int or device<0 for device in devices)):
        raise ValueError('training.gpu_devices must be distinct nonnegative integers')
    if t.get('amp_dtype','none') not in ('none','bfloat16','float16'):
        raise ValueError('training.amp_dtype must be none, bfloat16 or float16')
    for key,default in (('deterministic_algorithms',True),('cudnn_benchmark',False),
                        ('fused_optimizer',False),('cache_records_in_memory',False),
                        ('cache_appended_layers_in_memory',False),('channels_last',False)):
        if type(t.get(key,default)) is not bool:raise ValueError(f'training.{key} must be boolean')
    layer_roots=t.get('append_layer_cache_roots',[])
    if (not isinstance(layer_roots,list) or
            any(not isinstance(path,str) or not path for path in layer_roots)):
        raise ValueError('training.append_layer_cache_roots must be a list of nonempty paths')
    maximum=t.get('append_layer_maximum_cost',c['evaluation']['compatibility_threshold'])
    if not isinstance(maximum,(int,float)) or not np.isfinite(maximum) or maximum<0:
        raise ValueError('training.append_layer_maximum_cost must be finite and nonnegative')
    teacher=c['teacher']
    if (not isinstance(teacher.get('cache_roots',[]),list) or
            any(not isinstance(path,str) or not path for path in teacher.get('cache_roots',[]))):
        raise ValueError('teacher.cache_roots must be a list of nonempty paths')
    if teacher.get('backend','auto') not in ('auto','numba','reference','cuda'):
        raise ValueError('unknown teacher backend')
    if teacher.get('record_storage','npz') not in ('npz','memmap'):
        raise ValueError('teacher.record_storage must be npz or memmap')
    if (type(teacher.get('shard_commit_interval',128)) is not int or
            teacher.get('shard_commit_interval',128)<1):
        raise ValueError('teacher.shard_commit_interval must be a positive integer')
    if teacher.get('backend')=='cuda':
        for key,default in (('gpu_batch_size',16384),('gpu_pixel_chunk',16384),
                            ('gpu_pair_budget',134217728),('parallel_window_prefetch',2),
                            ('progress_every_records',32)):
            if type(teacher.get(key,default)) is not int or teacher.get(key,default)<1:
                raise ValueError(f'teacher.{key} must be a positive integer')
        gpu_devices=teacher.get('gpu_devices',devices)
        if (not isinstance(gpu_devices,list) or not gpu_devices or
                len(set(gpu_devices))!=len(gpu_devices) or
                any(type(device) is not int or device<0 for device in gpu_devices)):
            raise ValueError('teacher.gpu_devices must be distinct nonnegative integers')
        if c['variants']['response'].get('beta',0)!=0:
            raise ValueError('CUDA teacher requires beta=0')
        if type(teacher.get('compress_records',True)) is not bool:
            raise ValueError('teacher.compress_records must be boolean')
    return c


def context(c):
    root = (PROJECT_ROOT/c['output_root']).resolve()
    dataset = Phase1Dataset(PROJECT_ROOT/c['data_root'])
    return root,dataset


def plan(c, root, dataset):
    """Freeze selection without looking at responses, costs, sources, or test metrics."""
    counts = c['scene_counts']
    if sum(counts.values()) > dataset.num_scenes:
        raise ValueError('requested scene count exceeds release')
    rng = np.random.default_rng(c['split_seed'])
    shuffled = rng.permutation(dataset.num_scenes).tolist()
    splits, observations, offset = {}, {}, 0
    for split in ('train','validation','test'):
        splits[split] = shuffled[offset:offset+counts[split]]
        offset += counts[split]
        observations[split] = []
        for scene in splits[split]:
            # Stored window metadata only; no response, source, or geometry based selection.
            views = dataset._base.local_views[scene,:int(dataset._base.view_counts[scene])]
            for size in c['window_sizes']:
                available = np.flatnonzero(views[:,2] == size)
                if len(available) < c['views_per_size']:
                    raise ValueError(f'missing size {size} in scene {scene}')
                chosen = rng.choice(available,size=c['views_per_size'],replace=False)
                observations[split].extend(dict(scene_id=int(scene),view_id=int(v),size=int(size)) for v in sorted(chosen))
    split_manifest = None
    if c.get('protocol_version',1)>=2:
        from .protocol import fixed_selection
        splits,observations,split_manifest=fixed_selection(c,dataset)
    protocol = dict(interface_version=INTERFACE_VERSION,config=c,config_sha256=digest(c),
                    dataset_manifest_sha256=digest(dataset.manifest),source_sha256=source_hashes(),
                    scene_splits=splits,observations=observations,
                    selection='seeded scene permutation and seeded uniform selection within each size; no response filtering')
    if split_manifest is not None:
        protocol['split_manifest_sha256']=digest(split_manifest)
        protocol['selection']='fixed scene pools; stable per-scene/size windows, independent of training count'
    path = root/'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('experiment directory already has a different config/data/code/selection; use a new output_root')
    else:
        write_json(path,protocol)
    return protocol


def prepare(c, root, dataset, protocol, variant):
    settings = {**c['teacher'],**c['variants'][variant]}
    entries = {k:[] for k in protocol['observations']}
    rows = []
    directory = root/variant
    total = sum(map(len,protocol['observations'].values()))
    started = time.perf_counter()
    protocol_digest=digest(protocol)
    backend=settings.get('backend','auto')
    gpu_devices=list(settings.get('gpu_devices',c['training'].get('gpu_devices',[0]))) if backend=='cuda' else []
    workers=len(gpu_devices) if gpu_devices else 1
    record_storage=settings.get('record_storage','npz')
    if record_storage not in ('npz','memmap'):
        raise ValueError('teacher.record_storage must be npz or memmap')
    tokens=queue.Queue()
    for device in gpu_devices:tokens.put(device)
    environment_roots=[item for item in os.environ.get('RIND_PHASE1_TEACHER_CACHE_ROOTS','').split(os.pathsep)
                       if item]
    import_roots=list(settings.get('cache_roots',[]))+environment_roots
    cache_sources=_cache_sources(import_roots)
    call_settings={key:value for key,value in settings.items()
                   if key not in ('gpu_devices','parallel_window_prefetch',
                                  'progress_every_records','compress_records','cache_roots',
                                  'record_storage','shard_commit_interval')}
    compress_records=bool(settings.get('compress_records',True))
    shard_commit_interval=int(settings.get('shard_commit_interval',128))
    stores={};locations={}
    if record_storage=='memmap':
        from .search import CandidateDomain,uniform_grid
        from .teacher_store import TeacherShard
        physics={
            'alpha':float(settings.get('alpha',1.0)),
            'beta':float(settings.get('beta',0.0)),
            'boundary_lambda':float(settings.get('boundary_lambda',0.0)),
            'boundary_sigma':float(settings.get('boundary_sigma',1.0)),
            'source_intensity':1.0,
        }
        for split,samples in protocol['observations'].items():
            by_size={}
            for item in samples:
                by_size.setdefault(int(item['size']),[]).append(item)
            for size,group in by_size.items():
                first=dataset.get_observation(group[0]['scene_id'],group[0]['view_id'])
                grid=uniform_grid(CandidateDomain(tuple(first['window']),
                    int(dataset.manifest['global_size']),settings.get('quadtree_prior',False)),
                    settings['spacing'])
                shard=TeacherShard(
                    directory/'teacher_shards'/f'{split}_size{size}.rindshard',
                    capacity=len(group),candidate_count=len(grid.candidate_xy),
                    protocol_digest=protocol_digest,
                    settings=dict(world_size=int(dataset.manifest['global_size']),
                                  spacing=settings['spacing'],temperature=settings['temperature'],
                                  quadtree_prior=settings.get('quadtree_prior',False),physics=physics))
                stores[(split,size)]=shard
                for slot,item in enumerate(group):
                    locations[(split,item['scene_id'],item['view_id'])]=(shard,slot,shard.reference(slot))
    def compute(item,split):
        scene,view=item['scene_id'],item['view_id']
        if record_storage=='memmap':
            shard,slot,path=locations[(split,scene,view)]
        else:
            shard=slot=None
            path=directory/'teachers'/f's{scene}_v{view}.npz'
        if (shard.complete(slot) if shard is not None else path.exists()):
            record=load_teacher_record(path)
            if record['metadata'].get('experiment_protocol')!=protocol_digest:
                raise ValueError(f'teacher cache protocol mismatch: {path}')
            return item,path,record,'reused',[]
        imported,rejections=_import_compatible_teacher(
            cache_sources,dataset,scene,view,settings,protocol_digest)
        if imported is not None:
            if shard is None:
                temp=path.with_suffix('.tmp.npz');save_teacher_record(
                    imported,temp,compressed=compress_records);temp.replace(path)
            return item,path,imported,'imported',rejections
        device=tokens.get() if gpu_devices else None
        try:
            current=dict(call_settings)
            if device is not None:current['gpu_device']=f'cuda:{device}'
            record=generate_uniform_teacher(dataset,scene,view,**current)
        finally:
            # Persistent storage is unavoidable host I/O.  Formal server runs
            # disable NPZ compression, so no bulk compression work remains on CPU.
            if device is not None:tokens.put(device)
        record['metadata']['experiment_protocol']=protocol_digest
        if shard is None:
            temp=path.with_suffix('.tmp.npz');save_teacher_record(
                record,temp,compressed=compress_records);temp.replace(path)
        return item,path,record,'rendered',rejections
    # The Numba workqueue backend is not thread-safe.  Deep pipelining is a
    # CUDA-only execution optimization; CPU/reference preparation stays serial.
    pipeline_depth=(int(settings.get('parallel_window_prefetch',2))
                    if gpu_devices else 1)
    pipeline_workers=workers*pipeline_depth
    progress_every=int(settings.get('progress_every_records',32))
    print(f'prepare {variant}: observations={total}, backend={backend}, parallel GPUs={gpu_devices or ["cpu"]}, '
          f'render slots={workers}, pipeline workers={pipeline_workers}, '
          f'record storage={record_storage}, '
          f'configured candidate batch={settings.get("gpu_batch_size") if backend=="cuda" else "n/a"}, '
          f'append caches={[str(source["root"]) for source in cache_sources]}',flush=True)
    skipped=[];cache_counts={'reused':0,'imported':0,'rendered':0,'rejected_candidates':0}
    cache_rejections=[]
    with ThreadPoolExecutor(max_workers=pipeline_workers,thread_name_prefix='rind-phase1-teacher') as executor:
      for split, samples in protocol['observations'].items():
        chunk_size=pipeline_workers
        for chunk_start in range(0,len(samples),chunk_size):
          chunk=samples[chunk_start:chunk_start+chunk_size]
          eligible=[]
          for item in chunk:
            sample=dataset.get_observation(item['scene_id'],item['view_id'])
            free=~sample['obstacle'];values=sample['response'][free]
            constant=(not len(values) or np.all(values==values[0]))
            if c.get('exclude_constant_windows',False) and constant:
                skipped.append({**item,'split':split,'reason':('all_dark' if not len(values) or values[0]==0 else 'all_lit')})
            else:eligible.append(item)
          computed=list(executor.map(lambda item:compute(item,split),eligible))
          for item,path,record,status,rejections in computed:
            cache_counts[status]+=1;cache_counts['rejected_candidates']+=len(rejections)
            cache_rejections.extend(rejections)
            scene,view = item['scene_id'],item['view_id']
            if record_storage=='memmap' and status!='reused':
                shard,slot,_=locations[(split,scene,view)]
                shard.write(slot,record)
            training_record(record,dataset.get_observation(scene,view))
            valid,cost,q = record['valid'],record['physical_cost'],record['teacher_prob']
            if record_storage=='memmap':
                # Index statistics must match the persisted float32 payload so
                # a resumed prepare produces the identical records digest.
                cost=np.asarray(cost,dtype=np.float32).astype(np.float64)
                q=np.asarray(q,dtype=np.float32).astype(np.float64)
            low = valid & (cost <= c['evaluation']['compatibility_threshold'])
            rows.append(dict(scene_id=scene,view_id=view,split=split,size=item['size'],
                config_id=str(record['config_id']),
                candidate_count=int(len(record['candidate_xy'])),
                valid_candidates=int(valid.sum()),minimum_cost=float(cost[valid].min()),
                compatible_candidates=int(low.sum()),teacher_compatible_mass=float(q[low].sum()),
                teacher_entropy=float(-(q[q>0]*np.log(q[q>0])).sum()),
                sha256=(locations[(split,scene,view)][0].digest(locations[(split,scene,view)][1])
                        if record_storage=='memmap' else file_hash(path)),
                path=str(path.relative_to(directory))))
            entries[split].append(str(path.relative_to(directory)))
            if len(rows)%progress_every == 0 or len(rows)==total:
                print(f'prepare {variant}: completed={len(rows)}/{total}, skipped={len(skipped)}, '
                      f'cache={cache_counts}, '
                      f'{time.perf_counter()-started:.1f}s',flush=True)
          if record_storage=='memmap':
              for shard in {locations[(split,item['scene_id'],item['view_id'])][0]
                            for item in eligible}:
                  if shard.pending_count>=shard_commit_interval:
                      shard.flush()
        if record_storage=='memmap':
            for (store_split,_),shard in stores.items():
                if store_split==split:shard.flush()
    write_json(directory/'records.json',entries)
    # Hashes allow resume/evaluation to detect modified targets, not just a renamed manifest.
    write_json(directory/'teachers.json',dict(records=rows,records_sha256=digest(rows),
        protocol_sha256=digest(protocol),elapsed_seconds=time.perf_counter()-started,
        observations_without_compatible_candidates=sum(r['compatible_candidates']==0 for r in rows),
        skipped=skipped,teacher_backend=backend,gpu_devices=gpu_devices,
        record_storage=record_storage,
        append_cache_counts=cache_counts,append_cache_sources=[str(source['root']) for source in cache_sources],
        append_cache_rejections=cache_rejections[:1000],
        append_cache_rejections_truncated=max(0,len(cache_rejections)-1000)))
    return entries


def prepared(c, root, protocol, variant):
    directory=root/variant
    index=json.loads((directory/'teachers.json').read_text())
    if index['protocol_sha256'] != digest(protocol):
        raise ValueError('prepared targets belong to another protocol')
    entries=json.loads((directory/'records.json').read_text())
    expected={k:[] for k in ('train','validation','test')}
    record_storage=index.get('record_storage','npz')
    shard_cache={}
    for row in index['records']:
        if record_storage=='memmap':
            from .teacher_store import TeacherShard,is_shard_reference
            reference=directory/row['path']
            if not is_shard_reference(reference):
                raise ValueError('invalid teacher shard reference')
            if reference.parent not in shard_cache:
                shard_cache[reference.parent]=TeacherShard(
                    reference.parent,protocol_digest=digest(protocol),mode='r')
            shard=shard_cache[reference.parent];slot=int(reference.name)
            if not shard.complete(slot) or shard.digest(slot)!=row['sha256']:
                raise ValueError('teacher shard record is incomplete or its index digest changed')
            if (int(shard.arrays['scene_id'][slot])!=int(row['scene_id']) or
                    int(shard.arrays['view_id'][slot])!=int(row['view_id'])):
                raise ValueError('teacher shard identity mismatch')
        elif file_hash(directory/row['path']) != row['sha256']:
            raise ValueError('teacher record was modified after preparation')
        expected[row['split']].append(row['path'])
    if entries != expected:
        raise ValueError('records manifest changed after preparation')
    return directory,index


def atomic_checkpoint(path, checkpoint):
    import torch
    temp=path.with_suffix('.tmp.pt')
    torch.save(checkpoint,temp)
    temp.replace(path)


def _appended_layer_index(c,directory,variant):
    settings=c['training']
    roots=list(dict.fromkeys(list(settings.get('append_layer_cache_roots',[]))+
        [item for item in os.environ.get('RIND_PHASE2_LAYER_CACHE_ROOTS','').split(os.pathsep) if item]))
    if not roots:return None
    from .layer_cache import build_layer_cache_index
    teacher_settings={**c['teacher'],**c['variants'][variant]}
    return build_layer_cache_index(
        roots,directory/'appended_layer_cache.json',teacher_settings,
        settings.get('append_layer_maximum_cost',c['evaluation']['compatibility_threshold']))


def train(c,root,dataset,protocol,variant,seed,run_label=None,student_objective=None):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from .model import SourceEnergyField
    from .train import (load_splits,run_epoch,CompositeRecords,batch_count,
                        validate_student_objective)
    directory,index=prepared(c,root,protocol,variant)
    settings=c['training']
    student_objective=validate_student_objective(student_objective)
    if run_label is not None:
        import re
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',run_label):
            raise ValueError('run label must contain only letters, digits, underscores or hyphens')
    world_size=int(os.environ.get('WORLD_SIZE','1'))
    rank=int(os.environ.get('RANK','0'))
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    distributed=world_size>1
    runtime_batch_size=int(os.environ.get('RIND_RUNTIME_BATCH_SIZE',settings['batch_size']))
    accumulation_steps=int(os.environ.get('RIND_GRADIENT_ACCUMULATION_STEPS','1'))
    if runtime_batch_size<world_size:
        raise ValueError('runtime global batch size must be at least the distributed world size')
    if accumulation_steps<1:
        raise ValueError('gradient accumulation steps must be positive')
    if distributed and settings['device']!='cuda':
        raise ValueError('distributed training requires CUDA')
    torch.set_num_threads(max(1,settings['num_threads']//world_size))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(settings.get('deterministic_algorithms',True))
    rng=np.random.default_rng(seed)
    device=torch.device(settings['device'])
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if device.type=='mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS requested but unavailable')
    gpu_ids=list(settings.get('gpu_devices',[0])) if device.type=='cuda' else []
    if not distributed and gpu_ids and max(gpu_ids)>=torch.cuda.device_count():
        raise ValueError(f'requested GPUs {gpu_ids}, but only {torch.cuda.device_count()} are visible')
    if distributed:
        if local_rank>=torch.cuda.device_count():
            raise ValueError(f'local rank {local_rank} exceeds visible CUDA devices')
        device=torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    elif gpu_ids:
        device=torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    if device.type=='cuda':
        safe_cuda=(distributed and os.environ.get('RIND_SAFE_CUDA','1')!='0')
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=(False if safe_cuda else
                                        settings.get('cudnn_benchmark',False))
    else:
        safe_cuda=False
    if distributed and not dist.is_initialized():
        # Set the process-local device before NCCL initialization.  Avoid the
        # newer device_id keyword so the packaged PyTorch 2.4 runtime remains
        # supported.
        dist.init_process_group(backend='nccl',init_method='env://')
    # load_splits checks all IDs; test observations are not scored during training.
    splits=load_splits(directory/'records.json',dataset,
                       settings.get('normalization_support','geometry'),
                       settings.get('cache_records_in_memory',False))
    if distributed:
        layer_index=_appended_layer_index(c,directory,variant) if rank==0 else None
        dist.barrier()
        if rank:
            roots=list(dict.fromkeys(list(settings.get('append_layer_cache_roots',[]))+
                [item for item in os.environ.get('RIND_PHASE2_LAYER_CACHE_ROOTS','').split(os.pathsep) if item]))
            layer_index=(json.loads((directory/'appended_layer_cache.json').read_text())
                         if roots else None)
    else:
        layer_index=_appended_layer_index(c,directory,variant)
    base_train_count=len(splits['train'])
    if layer_index is not None:
        from .layer_cache import LayerCacheRecords
        if layer_index['entries']:
            appended=LayerCacheRecords(
                layer_index['entries'],settings.get('normalization_support','geometry'),
                settings.get('cache_appended_layers_in_memory',False))
            splits['train']=CompositeRecords([splits['train'],appended])
        if rank==0:
            print(f'append Phase2 layers {variant}: base={base_train_count}, '
                  f'accepted={len(layer_index["entries"])}, total={len(splits["train"])} | '
                  f'validation/test unchanged={len(splits["validation"])}/{len(splits["test"])}',flush=True)
    model_config=dict(hidden=settings['hidden'],fourier_frequencies=settings['fourier_frequencies'],
                      coordinate_scale=float(dataset.manifest['global_size']),
                      use_obstacle=settings.get('use_obstacle',True),pixel_coordinates=settings.get('pixel_coordinates',False),
                      normalization_support=settings.get('normalization_support','geometry'),
                      architecture=settings.get('architecture','baseline'),
                      attention_heads=settings.get('attention_heads',1))
    base_model=SourceEnergyField(**model_config).to(device)
    if settings.get('channels_last',False):
        base_model.to(memory_format=torch.channels_last)
    model=(DistributedDataParallel(base_model,device_ids=[local_rank],output_device=local_rank,
                                   broadcast_buffers=False,gradient_as_bucket_view=True,
                                   static_graph=True)
           if distributed else base_model)
    fused=bool(settings.get('fused_optimizer',False) and device.type=='cuda' and not safe_cuda)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],
                                weight_decay=settings['weight_decay'],fused=fused)
    run=directory/(f'seed_{seed}' if run_label is None else f'seed_{seed}_{run_label}')
    run.mkdir(parents=True,exist_ok=True)
    append_digest=digest(layer_index) if layer_index is not None else None
    signature_fields=dict(protocol=digest(protocol),targets=index['records_sha256'],
                          appended_layers=append_digest,seed=seed)
    # Preserve the exact historical signature for the existing CE baseline.
    if student_objective is not None:signature_fields['student_objective']=student_objective
    signature=digest(signature_fields)
    history,best,bad_epochs,start,steps=[],float('inf'),0,0,0
    if (run/'last.pt').exists():
        cp=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if cp['run_signature'] != signature:
            raise ValueError('resume checkpoint protocol/targets/seed mismatch')
        prior_software=cp.get('software',{})
        if ('global_batch_size' in prior_software and
                int(prior_software['global_batch_size'])!=runtime_batch_size):
            raise ValueError('resume checkpoint runtime batch size differs from this launch')
        if ('gradient_accumulation_steps' in prior_software and
                int(prior_software['gradient_accumulation_steps'])!=accumulation_steps):
            raise ValueError('resume checkpoint gradient accumulation differs from this launch')
        base_model.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        distributed_rng=cp.get('distributed_rng')
        local_rng=(distributed_rng[rank] if distributed_rng and rank<len(distributed_rng) else None)
        torch.set_rng_state(local_rng['torch'] if local_rng else cp['torch_rng'])
        rng.bit_generator.state=local_rng['numpy'] if local_rng else cp['numpy_rng']
        history,best,bad_epochs,start,steps=cp['history'],cp['best_validation_ce'],cp['bad_epochs'],cp['epoch'],cp['global_step']
        if device.type=='cuda' and local_rng and 'cuda' in local_rng:
            torch.cuda.set_rng_state(local_rng['cuda'],device)
        elif device.type=='cuda' and 'cuda_rng' in cp:
            if distributed:
                states=cp['cuda_rng']
                torch.cuda.set_rng_state(states[rank%len(states)],device)
            else:
                torch.cuda.set_rng_state_all(cp['cuda_rng'])
        if device.type=='mps' and 'mps_rng' in cp:
            torch.mps.set_rng_state(cp['mps_rng'])
    epoch_steps=batch_count(splits['train'],runtime_batch_size)
    parameters=sum(parameter.numel() for parameter in base_model.parameters())
    if rank==0:
        local_batch=(runtime_batch_size+world_size-1)//world_size
        print(f'train {variant} seed={seed}: {len(splits["train"])} train / {len(splits["validation"])} validation | '
              f'architecture={model_config["architecture"]}, parameters={parameters:,}, '
              f'parallelism={"DDP" if distributed else "single_device"}, GPUs={world_size if distributed else gpu_ids or ["cpu"]}, '
              f'AMP={"none" if safe_cuda else settings.get("amp_dtype","none")}, safe_cuda={safe_cuda}, '
              f'global batch={runtime_batch_size} (configured={settings["batch_size"]}), '
              f'accumulation={accumulation_steps}, effective batch~={runtime_batch_size*accumulation_steps}, '
              f'local batch~={local_batch}, objective={student_objective or "canonical_ce"}, '
              f'run={run.name}, resume epoch={start}',flush=True)
    # Chunked scoring calls custom module methods directly and therefore
    # bypasses DDP.forward.  H20 96G has enough memory for the full local
    # candidate grid; one standard forward/backward is both safer and faster.
    candidate_chunk=(None if distributed else settings.get('candidate_chunk'))
    effective_amp=('none' if safe_cuda else settings.get('amp_dtype','none'))
    every=settings.get('progress_every_batches',25)
    def progress(phase,epoch,total_batches):
        def report(batch_index,batch_records,running_ce):
            if batch_index==1 or batch_index%every==0 or batch_index==total_batches:
                print(f'{phase} seed={seed} epoch={epoch}/{settings["epochs"]} '
                      f'batch={batch_index}/{total_batches} records={batch_records} CE={running_ce:.6f}',flush=True)
        return report
    validation_steps=batch_count(splits['validation'],runtime_batch_size)
    for epoch in range(start,settings['epochs']):
        if bad_epochs >= settings['patience']:
            break
        tick=time.perf_counter()
        train_ce=run_epoch(model,splits['train'],runtime_batch_size,optimizer,rng,
                           grad_clip=settings['grad_clip'],candidate_chunk=candidate_chunk,
                           progress_callback=(progress('train',epoch+1,epoch_steps) if rank==0 else None),
                           amp_dtype=effective_amp,
                           prefetch_batches=settings.get('prefetch_batches',0),
                           distributed_rank=rank,distributed_world_size=world_size,
                           accumulation_steps=accumulation_steps,
                           student_objective=student_objective)
        val_ce=run_epoch(model,splits['validation'],runtime_batch_size,candidate_chunk=candidate_chunk,
                         progress_callback=(progress('validation',epoch+1,validation_steps) if rank==0 else None),
                         amp_dtype=effective_amp,
                         prefetch_batches=settings.get('prefetch_batches',0),
                         distributed_rank=rank,distributed_world_size=world_size)
        steps+=int(np.ceil(epoch_steps/accumulation_steps))
        improved=val_ce < best
        bad_epochs=0 if improved else bad_epochs+1
        best=min(best,val_ce)
        row=dict(epoch=epoch+1,train_ce=train_ce,validation_ce=val_ce,seconds=time.perf_counter()-tick)
        history.append(row)
        local_rng_state=dict(torch=torch.get_rng_state(),numpy=rng.bit_generator.state)
        if device.type=='cuda':local_rng_state['cuda']=torch.cuda.get_rng_state(device)
        distributed_rng=[None]*world_size if rank==0 else None
        if distributed:
            dist.gather_object(local_rng_state,distributed_rng,dst=0)
        else:
            distributed_rng=[local_rng_state]
        cp=dict(interface_version=INTERFACE_VERSION,model_config=model_config,model=base_model.state_dict(),
            optimizer=optimizer.state_dict(),epoch=epoch+1,global_step=steps,seed=seed,run_signature=signature,
            protocol_sha256=digest(protocol),records_sha256=index['records_sha256'],
            appended_layer_index_sha256=append_digest,
            student_objective=student_objective,run_label=run_label,
            training_records=dict(base_phase1=base_train_count,
                                  appended_phase2_layers=len(layer_index['entries']) if layer_index else 0,
                                  total=len(splits['train'])),
            scene_splits=protocol['scene_splits'],teacher_configs=sorted(splits['train'].config_ids),
            best_validation_ce=best,bad_epochs=bad_epochs,history=history,
            torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,
            software=dict(torch=str(torch.__version__),numpy=np.__version__,device=str(device),
                          gpu_ids=gpu_ids,parallelism=('distributed_data_parallel' if distributed else 'single_device'),
                          world_size=world_size,global_batch_size=runtime_batch_size,
                          configured_batch_size=settings['batch_size'],
                          gradient_accumulation_steps=accumulation_steps,
                          num_threads=settings['num_threads'],amp_dtype=effective_amp,safe_cuda=safe_cuda,
                          fused_optimizer=fused,channels_last=settings.get('channels_last',False),
                          model_parameters=parameters))
        cp['distributed_rng']=distributed_rng
        if device.type=='cuda' and not distributed: cp['cuda_rng']=torch.cuda.get_rng_state_all()
        if device.type=='mps': cp['mps_rng']=torch.mps.get_rng_state()
        if rank==0:
            if improved: atomic_checkpoint(run/'best.pt',cp)
            atomic_checkpoint(run/'last.pt',cp)
            write_json(run/'training.json',dict(seed=seed,history=history,best_validation_ce=best,
                       training_records=dict(base_phase1=base_train_count,
                                             appended_phase2_layers=len(layer_index['entries']) if layer_index else 0,
                                             total=len(splits['train'])),
                       selection='minimum validation cross entropy; test not evaluated during training'))
            print(f'{variant} seed={seed} epoch={epoch+1}: train={train_ce:.5f} val={val_ce:.5f} best={best:.5f} ({row["seconds"]:.1f}s)',flush=True)
        if distributed:dist.barrier()
    if rank==0 and not (run/'best.pt').exists():
        raise ValueError('training produced no checkpoint')
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return run


def evaluation(c,root,dataset,protocol,variant,seed,run_label=None):
    import torch
    # Part E lives in the checkout; no second copy of scientific metrics.
    from part_e.evaluate import evaluate_record,aggregate,window_summary,plot_record,unseen_scene_check,unseen_obstacle_combinations
    from .train import load_model
    from .predict import predict
    directory,index=prepared(c,root,protocol,variant)
    run=directory/(f'seed_{seed}' if run_label is None else f'seed_{seed}_{run_label}')
    model,cp=load_model(run/'best.pt')
    layer_index=_appended_layer_index(c,directory,variant)
    signature_fields=dict(protocol=digest(protocol),targets=index['records_sha256'],
                          appended_layers=(digest(layer_index) if layer_index is not None else None),seed=seed)
    if cp.get('student_objective') is not None:
        signature_fields['student_objective']=cp['student_objective']
    signature=digest(signature_fields)
    if cp.get('run_signature') != signature:
        raise ValueError('checkpoint does not match frozen protocol and targets')
    last=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
    if last['epoch'] < c['training']['epochs'] and last['bad_epochs'] < c['training']['patience']:
        raise ValueError('training is not finished; test evaluation is locked')
    torch.set_num_threads(c['training']['num_threads'])
    model.to(c['training']['device'])
    rows,baselines,records,common_response,assisted=[],[],[],[],[]
    entries=json.loads((directory/'records.json').read_text())
    for i,filename in enumerate(entries['test']):
        record=load_teacher_record(directory/filename)
        sample=dataset.get_observation(record['scene_id'],record['view_id'])
        prediction=predict(model,record,sample,chunk=c['evaluation']['prediction_chunk'])
        row=evaluate_record(record,prediction,threshold=c['evaluation']['compatibility_threshold'],spacing=c['teacher']['spacing'])
        # Uniform over exactly the same declared student normalization support.
        support=prediction.get('support',record['valid'])
        uniform={**prediction,'student_prob':support.astype(float)/support.sum()}
        baseline=evaluate_record(record,uniform,threshold=c['evaluation']['compatibility_threshold'],spacing=c['teacher']['spacing'])
        response=sample['response']
        row['response_group']='uniform' if np.all(response==response.flat[0]) else 'nonuniform'
        rows.append(row);baselines.append(baseline);records.append(record)
        from .sampling import probability_from_energy
        assisted_prediction={**prediction,'support':record['valid'],'normalization_support':'geometry',
                             'student_prob':probability_from_energy(prediction['energy'],record['valid'])}
        assisted.append(evaluate_record(record,assisted_prediction,threshold=c['evaluation']['compatibility_threshold'],spacing=c['teacher']['spacing']))
        if variant != 'response':
            reference=load_teacher_record(root/'response'/filename)
            if not all(np.array_equal(reference[k],record[k]) for k in ('candidate_xy','valid','window')):
                raise ValueError('edge ablation requires identical response-baseline support')
            # Re-evaluation against an explicitly declared common response teacher.
            reference_prediction={**prediction,'config_id':reference['config_id']}
            common_response.append(evaluate_record(reference,reference_prediction,
                threshold=c['evaluation']['compatibility_threshold'],spacing=c['teacher']['spacing']))
        path=run/'predictions'/f's{record["scene_id"]}_v{record["view_id"]}.npz'
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**prediction)
        if i < c['evaluation']['plot_count']:
            plot_record(record,prediction,run/f'example_{i}.svg',response)
    # Categorical obstacle-type multisets are a limited, explicitly named diagnostic.
    signatures={}
    for scene in protocol['scene_splits']['train']+protocol['scene_splits']['test']:
        signatures[str(scene)]=dataset.get_scene(scene)['obstacle_types'].astype(int).tolist()
    report=dict(variant=variant,seed=seed,checkpoint_sha256=file_hash(run/'best.pt'),best_epoch=cp['epoch'],
        run_signature=signature,observations=rows,aggregate=aggregate(rows),by_window_size=window_summary(rows),
        by_response_group={g:aggregate([r for r in rows if r['response_group']==g]) for g in ('uniform','nonuniform')},
        normalization_support=model.normalization_support,use_obstacle=model.use_obstacle,
        geometry_assisted_secondary=aggregate(assisted),common_response_reference=aggregate(common_response or rows),
        uniform_baseline=aggregate(baselines),uniform_baseline_by_window_size=window_summary(baselines),
        unseen_scene=unseen_scene_check(records,protocol['scene_splits'],cp['scene_splits']['train']),
        obstacle_combinations=unseen_obstacle_combinations(rows,signatures,cp['scene_splits']['train']),
        interpretation='Discrete common grid; normalization_support declares whether hidden geometry masks student probabilities. Invalid mass and penalized cost are explicit. Constant/nonconstant response is not a count of independent geometric boundaries.')
    write_json(run/'test_metrics.json',report)
    print(f'evaluated {variant} seed={seed}: {len(rows)} observations',flush=True)
    return report


def summarize(c,root,variant):
    reports=[json.loads((root/variant/f'seed_{seed}'/'test_metrics.json').read_text()) for seed in c['training']['seeds']]
    common=set.intersection(*(set(r['aggregate']['metrics']) for r in reports))
    metrics={}
    for key in sorted(common):
        values=[r['aggregate']['metrics'][key]['scene_mean'] for r in reports]
        metrics[key]=dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)) if len(values)>1 else 0.,per_seed=values)
    result=dict(variant=variant,seeds=c['training']['seeds'],scene_balanced_metrics=metrics,
                best_epochs=[r['best_epoch'] for r in reports],
                uniform_baseline=reports[0]['uniform_baseline'],
                note='Standard deviation across training seeds, not a confidence interval over independent datasets.')
    write_json(root/variant/'summary.json',result)
    lines=[f'# {variant}: first-round experiment','',f'Seeds: {c["training"]["seeds"]}; best epochs: {result["best_epochs"]}.','',
           '| Metric (scene mean) | Mean | Seed std |','|---|---:|---:|']
    for key in ('forward_kl','jensen_shannon','l1_distance','student_compatible_mass','student_expected_physical_cost','student_invalid_mass','student_expected_penalized_cost','student_entropy','teacher_entropy'):
        if key in metrics:
            m=metrics[key];lines.append(f'| {key} | {m["mean"]:.6f} | {m["std"]:.6f} |')
    lines+=['','These results apply only to the frozen scene subset, window sizes and candidate lattice. See per-seed reports for uniform baseline, window groups, and grid coverage.']
    (root/variant/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result


def compare_search(c,root,dataset,protocol,variant):
    """Compare adaptive retention against the exact same uniform reference costs."""
    from .physics import PhysicalEvaluator
    from .search import CandidateDomain,uniform_grid
    from part_e.adaptive import adaptive_search
    from part_e.evaluate import search_comparison
    directory,_=prepared(c,root,protocol,variant)
    paths=json.loads((directory/'records.json').read_text())['test']
    settings=c['search_comparison']
    rows=[]
    for filename in paths[:settings['max_observations']]:
        reference=load_teacher_record(directory/filename)
        scene,view=reference['scene_id'],reference['view_id']
        domain=CandidateDomain(tuple(reference['window']),int(dataset.manifest['global_size']),False)
        grid=uniform_grid(domain,c['teacher']['spacing'])
        kwargs={k:v for k,v in {**c['teacher'],**c['variants'][variant]}.items() if k not in ('spacing','temperature')}
        evaluator=PhysicalEvaluator(dataset,scene,view,**kwargs)
        result=adaptive_search(grid,evaluator,budget=len(grid.candidate_xy),
            coarse_stride=settings['coarse_stride'],retain_fraction=settings['retain_fraction'],
            exploration_fraction=settings['exploration_fraction'],seed=settings['seed'])
        row=dict(scene_id=scene,view_id=view,complete=result.complete)
        if result.complete:
            row.update(search_comparison(reference,result.covered,
                threshold=c['evaluation']['compatibility_threshold'],
                adaptive_evaluations=result.num_physics_evaluations,adaptive_seconds=result.elapsed_seconds))
        rows.append(row)
    report=dict(observations=rows,protocol_sha256=digest(protocol),
        note='Held-out fixed subset. Render-count and compatible-mass coverage comparison; one-run timing is not a controlled speed benchmark. Adaptive probabilities are conditional on retained support; no area weighting.')
    write_json(directory/'search_comparison.json',report)
    return report


def sampling_probe(c,root,dataset,protocol,variant,seed):
    """Validation-only physical check on an interleaved, finer candidate grid."""
    from .train import load_model
    from .predict import predict
    from part_e.evaluate import evaluate_record,aggregate
    spacing=c['evaluation'].get('probe_spacing')
    if spacing is None: raise ValueError('set evaluation.probe_spacing')
    if spacing>=c['teacher']['spacing']: raise ValueError('probe spacing must be finer than training spacing')
    directory,_=prepared(c,root,protocol,variant)
    model,_=load_model(directory/f'seed_{seed}'/'best.pt')
    rows=[]
    for item in protocol['observations']['validation'][:c['evaluation'].get('probe_observations',16)]:
        settings={**c['teacher'],**c['variants'][variant],'spacing':spacing}
        record=generate_uniform_teacher(dataset,item['scene_id'],item['view_id'],**settings)
        sample=dataset.get_observation(item['scene_id'],item['view_id'])
        result=predict(model,record,sample,chunk=c['evaluation']['prediction_chunk'])
        rows.append(evaluate_record(record,result,threshold=c['evaluation']['compatibility_threshold'],spacing=spacing))
    write_json(directory/f'seed_{seed}'/'sampling_probe.json',dict(split='validation',training_spacing=c['teacher']['spacing'],probe_spacing=spacing,
        observations=rows,aggregate=aggregate(rows),note='New physical targets on a finer grid. Does not establish pixel-scale convergence.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','prepare','train','evaluate','summarize','search','probe','all'])
    parser.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/experiment_consistent.json')
    parser.add_argument('--variant',default='response')
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    c=read_config(args.config)
    if args.variant not in c['variants']:
        parser.error('unknown variant')
    root,dataset=context(c)
    protocol=plan(c,root,dataset)
    seeds=c['training']['seeds'] if args.seed is None else [args.seed]
    if not set(seeds)<=set(c['training']['seeds']):
        parser.error('seed must be declared in the frozen config')
    if args.stage=='plan':
        print(f'Frozen protocol: {root/"protocol.json"}',flush=True)
    if args.stage=='probe':
        for seed in seeds: sampling_probe(c,root,dataset,protocol,args.variant,seed)
    if args.stage=='search': compare_search(c,root,dataset,protocol,args.variant)
    if args.stage in ('prepare','all'): prepare(c,root,dataset,protocol,args.variant)
    if args.stage in ('train','all'):
        for seed in seeds: train(c,root,dataset,protocol,args.variant,seed)
    if args.stage in ('evaluate','all'):
        for seed in seeds: evaluation(c,root,dataset,protocol,args.variant,seed)
    if args.stage in ('summarize','all') and all((root/args.variant/f'seed_{seed}'/'test_metrics.json').exists() for seed in c['training']['seeds']):
        summarize(c,root,args.variant)
    if args.stage=='all' and c['evaluation'].get('run_search_comparison',False):
        compare_search(c,root,dataset,protocol,args.variant)


if __name__=='__main__': main()
