"""Append actual R>=3 windows, keeping the existing corpus/evaluation intact.

No source candidate rendering. Ordered multiprocessing prevents worker timing
from selecting the dataset. Checkpointable chunks never count rejected/repeated
windows toward the requested number. Uses the unchanged local label function.
"""
from collections import Counter
import copy
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import time

import numpy as np

import mask_only_data as labels
from mask_validation import validate_mask_cache
from search_cache import immutable_json
from expand_masks import _digest_arrays
from extend_evaluation import _ordered_results


def _configuration_key(dataset, scene, memo):
    if scene not in memo:
        geo = dataset.geometry_group(scene)
        sources = np.asarray(dataset.get_source_params(scene), dtype='<f8')
        sources = sources[np.lexsort((sources[:, 2], sources[:, 1], sources[:, 0]))]
        sources = np.where(sources == 0, 0., sources)
        memo[scene] = (geo, hashlib.sha256(geo.encode() + sources.tobytes()).hexdigest())
    return memo[scene]


def _fingerprint(configuration, window):
    return hashlib.sha256(configuration.encode() + np.asarray(window, dtype='<i8').tobytes()).hexdigest()


def _tasks(dataset, scenes, old_ids, options, cursor):
    order = np.array(scenes, dtype=np.int64)
    np.random.default_rng(options['seed']).shuffle(order)
    views, serial = {}, 0
    for round_number in range(options['max_views_per_scene']):
        for s in order:
            scene = int(s)
            if scene not in views:
                available = [(int(v), int(side)) for v, side in dataset.iter_scene_views(scene)
                             if int(side) in options['window_sizes']]
                np.random.default_rng(np.random.SeedSequence([options['seed'], scene])).shuffle(available)
                views[scene] = available[:options['max_views_per_scene']]
            if round_number >= len(views[scene]):
                continue
            view, side = views[scene][round_number]
            if (scene, view) in old_ids:
                continue
            if serial >= cursor:
                yield serial, scene, view, side
            serial += 1


def _assemble(source, output, old, chunks, signature, options, stats, hashes):
    cache = output / 'mask_cache'
    cache.mkdir(exist_ok=True)
    records = copy.deepcopy(old['records'])
    fields = (('response','f32','response_offset',4), ('obstacle','u8','obstacle_offset',1),
              ('masks','u8','mask_offset',1))
    for split in ('validation', 'test'):
        for key, suffix, _, _ in fields:
            destination = cache / f'{split}_{key}.{suffix}'
            original = (source/'mask_cache'/destination.name).resolve()
            if destination.is_symlink() and destination.resolve() != original:
                raise ValueError('existing evaluation cache link has a different source')
            if destination.exists() and not destination.is_symlink():
                raise ValueError('refusing to overwrite an existing evaluation file')
            if not destination.is_symlink():
                destination.symlink_to(original)
    streams, offsets = {}, {}
    try:
        for key, suffix, _, itemsize in fields:
            name = f'train_{key}.{suffix}'
            destination = cache / (name + '.tmp')
            shutil.copyfile(source/'mask_cache'/name, destination)
            streams[key] = destination.open('ab')
            offsets[key] = destination.stat().st_size // itemsize
        for path in chunks:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z['metadata']))
                payload = {key: np.asarray(z[key]) for key, *_ in fields}
            if _digest_arrays(payload.values()) != meta['arrays_sha256']:
                raise ValueError('complex append chunk checksum mismatch')
            for row in meta['records']:
                row = copy.deepcopy(row)
                for key, _, field, _ in fields:
                    row[field] += offsets[key]
                records['train'].append(row)
            for key, value in payload.items():
                value.tofile(streams[key]); offsets[key] += value.size
    finally:
        for stream in streams.values(): stream.close()
    for key, suffix, _, _ in fields:
        (cache/f'train_{key}.{suffix}.tmp').replace(cache/f'train_{key}.{suffix}')
    index = {**copy.deepcopy(old), 'records': records,
             'counts': {k:len(v) for k,v in records.items()},
             'corpus_sha256': labels._digest({'source':old['corpus_sha256'], 'signature':signature,'chunks':hashes}),
             'training_expansion': {'signature':signature,'source_root':str(source),
                 'source_index_sha256': labels._digest(old), 'base_train_windows':old['counts']['train'],
                 'new_complex_windows':stats['accepted'], 'selection':options,
                 'frozen_evaluation_sha256':labels._digest({s:old['records'][s] for s in ('validation','test')}),
                 'selection_statistics':dict(stats)}}
    index.pop('cache_view',None)
    index['dataset_distributions'] = {s:{'windows':len(r),'scenes':len({x['scene_id'] for x in r}),
        'effective_layers':dict(Counter(str(len(x['intensities'])) for x in r)),
        'size':dict(Counter(str(x['size']) for x in r))} for s,r in records.items()}
    validate_mask_cache(index,cache,index['max_slots'])
    assert records['train'][:len(old['records']['train'])] == old['records']['train']
    assert all(records[s] == old['records'][s] for s in ('validation','test'))
    labels._write_json(output/'mask_records.json',index)
    return index


def _save_distribution(output, old, index):
    base=old['records']['train'];added=index['records']['train'][len(base):]
    histogram=lambda rows: dict(sorted(Counter(str(len(r['intensities'])) for r in rows).items(),key=lambda kv:int(kv[0])))
    summary={'base_windows':len(base),'added_windows':len(added),'total_windows':index['counts']['train'],
             'base_effective_layers':histogram(base),'added_effective_layers':histogram(added),
             'total_effective_layers':histogram(index['records']['train']),
             'added_configurations':len({r['configuration_key'] for r in added}),
             'added_geometry_groups':len({r['geometry_group'] for r in added}),
             'train_unique_scene_view_ids':len({(r['scene_id'],r['view_id']) for r in index['records']['train']}),
             'replacement_used_for_data_generation':False,'candidate_renders':0}
    labels._write_json(output/'training_distribution.json',summary)
    print('Actual training R histogram:',summary['total_effective_layers'],flush=True)


def prepare_complex(config, dataset=None, *, stop_after_chunks=None):
    if config.get('training_expansion', {}).get('mode', 'append') == 'inventory':
        from inventory_complex import inventory_complex
        return inventory_complex(config, dataset, stop_after_chunks=stop_after_chunks)
    import fcntl
    from train_decomposition import resolve
    source, output = resolve(config['cache_source_root']).resolve(), resolve(config['output_root']).resolve()
    if source == output:
        raise ValueError('complex expansion needs a new output_root')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.complex_expansion.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError('another complex expansion is running') from e
        return _prepare(config,source,output,dataset,stop_after_chunks)


def _prepare(config,source,output,dataset,stop_after_chunks):
    from train_decomposition import resolve
    old = json.loads((source/'mask_records.json').read_text())
    validate_mask_cache(old,source/'mask_cache',config['model']['max_slots'])
    options = dict(config['training_expansion'])
    protocol_root = resolve(options.pop('label_protocol_root'))
    protocol = json.loads((protocol_root/'mask_protocol.json').read_text())
    target = options['additional_complex_windows']
    if target != 'all' and (type(target) is not int or target < 1):
        raise ValueError('additional_complex_windows must be a positive integer or all')
    for k in ('workers','chunk_size','max_views_per_scene','max_added_per_configuration','max_added_per_geometry'):
        if type(options[k]) is not int or options[k] < 1:raise ValueError(f'{k} must be positive')
    if type(options['min_layers']) is not int or not 2 <= options['min_layers'] <= config['model']['max_slots']:
        raise ValueError('min_layers must be between 2 and max_slots')
    if type(options['seed']) is not int or options['seed'] < 0:raise ValueError('seed must be nonnegative')
    if not options['window_sizes'] or any(type(s) is not int or s < 1 for s in options['window_sizes']):
        raise ValueError('window_sizes must be positive integer sizes')
    requested=config['label_preparation']['record_counts']
    if any(requested[s]!=old['counts'][s] for s in ('validation','test')):
        raise ValueError('validation/test counts must remain exactly frozen')
    expected=old['counts']['train']+target if target!='all' else 'all'
    if requested['train']!=expected:
        raise ValueError(f'label_preparation train count must be {expected}, equal to source plus new complex windows')
    inventory_root=options.pop('inventory_root',None)
    catalogue=None;catalogue_signature=None
    if inventory_root is not None:
        from inventory_selection import select_catalogue
        catalogue,catalogue_signature=select_catalogue(resolve(inventory_root),old,options,target)
    semantic = {k:v for k,v in options.items() if k not in ('workers','chunk_size')}
    contract = {'schema':'rind-complex-train-extension-v1', 'source_index_sha256':labels._digest(old),
                'label_protocol_signature':protocol['signature'],'selection':semantic,
                'max_slots':config['model']['max_slots'],'dataset_manifest':protocol['dataset_manifest']}
    if catalogue is not None:
        contract['inventory_signature']=catalogue_signature
        contract['selected_catalogue_sha256']=labels._digest(catalogue)
    signature = labels._digest(contract)
    immutable_json(output/'complex_expansion_protocol.json',{'signature':signature,**contract})
    if (output/'mask_records.json').exists():
        index=json.loads((output/'mask_records.json').read_text())
        if index.get('training_expansion',{}).get('signature') != signature:
            raise ValueError('existing complex cache belongs to another protocol')
        validate_mask_cache(index,output/'mask_cache',config['model']['max_slots'])
        _save_distribution(output,old,index)
        print(f'Complex cache ready: {index["counts"]}; no label generation',flush=True)
        return index
    import rind_phase2.regions as regions
    import rind_phase2.region_geometry as geometry
    expected={'label_code_sha256':hashlib.sha256(Path(labels.__file__).read_bytes()).hexdigest(),
              'tracer_sha256':hashlib.sha256(inspect.getsource(labels.trace_contribution).encode()).hexdigest(),
              'regions_sha256':hashlib.sha256(Path(regions.__file__).read_bytes()).hexdigest(),
              'geometry_sha256':hashlib.sha256(Path(geometry.__file__).read_bytes()).hexdigest()}
    if any(protocol.get(k)!=v for k,v in expected.items()):
        raise ValueError('cached teacher label rule changed; do not mix labels')
    if dataset is None:
        from dataset_update import open_dataset
        dataset=open_dataset(resolve(config['data_root']))
    if dataset.manifest != protocol['dataset_manifest']:
        raise ValueError('raw corpus differs from original label protocol')
    splits=labels._splits(dataset,{'label_preparation':protocol['selection']})
    if labels._digest(splits) != protocol['split_sha256']:raise ValueError('frozen split changed')
    membership={int(s):name for name,ss in splits.items() for s in ss}
    if sum(map(len,splits.values())) != len(membership):raise ValueError('overlapping dataset splits')
    if any(membership.get(row['scene_id'])!=name for name,rs in old['records'].items() for row in rs):
        raise ValueError('source cache crosses the frozen scene split')
    heldout_geometries={dataset.geometry_group(int(s)) for name in ('validation','test') for s in splits[name]}
    memo, fingerprints, ids = {}, set(), set()
    for row in old['records']['train']:
        geo,key=_configuration_key(dataset,row['scene_id'],memo)
        if geo in heldout_geometries:raise ValueError('source training geometry occurs in a held-out pool')
        fingerprints.add(_fingerprint(key,row['window'])); ids.add((row['scene_id'],row['view_id']))
    directory=output/'complex_chunks';directory.mkdir(exist_ok=True)
    chunks=sorted(directory.glob('chunk-*.npz'))
    stats=Counter();cursor=0;hashes=[];by_configuration=Counter();by_geometry=Counter()
    for number,path in enumerate(chunks):
        if path.name!=f'chunk-{number:06d}.npz':raise ValueError('complex chunk sequence has a gap')
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            content={k:np.asarray(z[k]) for k in ('response','obstacle','masks')}
        if meta['signature']!=signature or _digest_arrays(content.values())!=meta['arrays_sha256']:
            raise ValueError('corrupt or incompatible complex chunk')
        if meta['next_serial']<cursor:raise ValueError('complex chunk cursor regressed')
        for row in meta['records']:
            identity=(row['scene_id'],row['view_id']); fingerprint=row['configuration_window_fingerprint']
            if identity in ids or fingerprint in fingerprints:raise ValueError('repeated complex training window')
            if len(row['intensities'])<options['min_layers']:raise ValueError('noncomplex append row')
            ids.add(identity);fingerprints.add(fingerprint)
            by_configuration[row['configuration_key']]+=1;by_geometry[row['geometry_group']]+=1
        cursor=meta['next_serial'];stats=Counter(meta['statistics']);hashes.append(meta['arrays_sha256'])
    if stats['accepted'] != sum(by_configuration.values()):raise ValueError('complex accepted counter mismatch')
    print('Local-label geometry preflight:',regions.check_geometry_backend(),flush=True)
    work={**options,'max_slots':config['model']['max_slots'],'tolerance':protocol['selection'].get('tolerance',1e-12)}
    pending=[];arrays={k:[] for k in ('response','obstacle','masks')};lengths=Counter();written=0
    started=time.perf_counter();committed=stats['checked']
    def enough():return target!='all' and stats['accepted']>=target
    def flush():
        nonlocal pending,arrays,lengths,written,committed
        payload={k:np.concatenate(v) if v else np.empty(0,dtype='<f4' if k=='response' else 'u1') for k,v in arrays.items()}
        h=_digest_arrays(payload.values())
        meta={'signature':signature,'records':pending,'next_serial':cursor,'statistics':dict(stats),'arrays_sha256':h}
        path=directory/f'chunk-{len(chunks):06d}.npz';temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as stream:np.savez_compressed(stream,metadata=np.asarray(json.dumps(meta)),**payload)
        temporary.replace(path);chunks.append(path);hashes.append(h)
        pending=[];arrays={k:[] for k in arrays};lengths=Counter();written+=1;committed=stats['checked']
        if stop_after_chunks is not None and written>=stop_after_chunks:raise InterruptedError('committed complex chunk')
    old_ids={(r['scene_id'],r['view_id']) for r in old['records']['train']}
    tasks=(_tasks(dataset,splits['train'],old_ids,options,cursor) if catalogue is None else
           ((i,r['scene_id'],r['view_id'],r['size']) for i,r in enumerate(catalogue) if i>=cursor))
    results=_ordered_results(tasks,dataset,resolve(config['data_root']),work,options['workers'])
    try:
        if not enough():
            for serial,result in results:
                cursor=serial+1;stats['checked']+=1
                if catalogue is not None:
                    expected=catalogue[serial]
                    if result['status']!='complete':raise ValueError('inventoried window no longer decomposes completely')
                    actual=result['row']
                    h=hashlib.sha256(result['arrays']['masks'].tobytes()+
                                     np.asarray(actual['intensities'],dtype='<f8').tobytes()).hexdigest()
                    if (h!=expected['teacher_label_sha256'] or actual['window']!=expected['window']):
                        raise ValueError('regenerated local label differs from inventory; refusing mixed labels')
                if result['status']!='complete':stats[result.get('reason','incomplete')]+=1
                elif len(result['row']['intensities'])<options['min_layers']:stats['too_few_layers']+=1
                else:
                    row=result['row'];geo,key=_configuration_key(dataset,row['scene_id'],memo)
                    if geo in heldout_geometries:raise ValueError('candidate training geometry leaks into held-out pool')
                    fingerprint=_fingerprint(key,row['window'])
                    if fingerprint in fingerprints:stats['duplicate_configuration_window']+=1
                    elif by_configuration[key]>=options['max_added_per_configuration']:stats['configuration_cap']+=1
                    elif by_geometry[geo]>=options['max_added_per_geometry']:stats['geometry_cap']+=1
                    else:
                        row.update(configuration_key=key,geometry_group=geo,configuration_window_fingerprint=fingerprint)
                        for field,name in (('response_offset','response'),('obstacle_offset','obstacle'),('mask_offset','masks')):
                            row[field]=lengths[name]
                        for k,v in result['arrays'].items():arrays[k].append(v);lengths[k]+=v.size
                        pending.append(row);fingerprints.add(fingerprint);ids.add((row['scene_id'],row['view_id']))
                        by_configuration[key]+=1;by_geometry[geo]+=1;stats['accepted']+=1
                if len(pending)>=options['chunk_size'] or stats['checked']-committed>=500 or enough():
                    flush()
                    print(f'complex train | new real R>={options["min_layers"]}: {stats["accepted"]:,}/{target} | '
                          f'checked={stats["checked"]:,} | source={old["counts"]["train"]:,} | '
                          f'elapsed={time.perf_counter()-started:.1f}s | candidate renders=0',flush=True)
                if enough():break
    finally:
        results.close()
    if pending or stats['checked']!=committed:flush()
    report={'requested_new':target,'accepted_new':stats['accepted'],'base_train_windows':old['counts']['train'],
            'total_train_windows':old['counts']['train']+stats['accepted'],'statistics':dict(stats),
            'added_scene_configurations':len(by_configuration),'added_geometry_groups':len(by_geometry),
            'requested_target_reached':enough() if target!='all' else True,
            'candidate_pool_exhausted':not enough(),'candidate_renders':0,'seconds_this_run':time.perf_counter()-started}
    labels._write_json(output/'complex_availability.json',report)
    if target!='all' and not enough():
        raise RuntimeError(f'only {stats["accepted"]:,}/{target:,} unique completed complex windows available; '
                           'under configured scan/caps; no training index published; see complex_availability.json; never fill by replacement')
    index=_assemble(source,output,old,chunks,signature,semantic,stats,hashes)
    _save_distribution(output,old,index)
    print(f'Complex dataset complete: {index["counts"]}; evaluation unchanged; candidate renders=0',flush=True)
    return index
