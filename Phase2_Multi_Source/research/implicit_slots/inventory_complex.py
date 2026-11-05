"""Read-only census of new complex windows, with resumable metadata catalogues.

Scans the frozen training scene pool. Does not construct a training cache, load
weights, call a model, or change any evaluation records. A complete inventory
is a census of the declared sizes/view limit, not a fixed-size random sample.
"""
from collections import Counter
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import time

import mask_only_data as labels
from mask_validation import validate_mask_cache
from search_cache import immutable_json
from expand_complex_masks import _configuration_key, _fingerprint, _tasks
from extend_evaluation import _ordered_results


def _load(path):
    return json.loads(path.read_text())


def _hist(counter, prefix):
    return {k[len(prefix):]:v for k,v in sorted(counter.items(),
        key=lambda item:int(item[0][len(prefix):]) if item[0].startswith(prefix) else -1)
        if k.startswith(prefix)}


def inventory_complex(config, dataset=None, *, stop_after_chunks=None):
    import fcntl
    from train_decomposition import resolve
    source, output = resolve(config['cache_source_root']).resolve(), resolve(config['output_root']).resolve()
    if source == output:
        raise ValueError('inventory output must be separate from the original experiment')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.inventory.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError('another inventory is writing this output') from e
        return _inventory(config,source,output,dataset,stop_after_chunks)


def _inventory(config, source, output, dataset, stop_after_chunks):
    from train_decomposition import resolve
    old = _load(source/'mask_records.json')
    validate_mask_cache(old,source/'mask_cache',config['model']['max_slots'])
    options = dict(config['training_expansion'])
    protocol_root = resolve(options.pop('label_protocol_root'))
    protocol = _load(protocol_root/'mask_protocol.json')
    if options.pop('mode',None) != 'inventory':
        raise ValueError('inventory mode must be explicit')
    for key in ('workers','chunk_size','max_views_per_scene','max_added_per_configuration','max_added_per_geometry'):
        if type(options[key]) is not int or options[key] < 1:raise ValueError(f'{key} must be positive')
    if (type(options['min_layers']) is not int or not 2<=options['min_layers']<=config['model']['max_slots']
            or type(options['seed']) is not int or options['seed']<0):
        raise ValueError('invalid inventory minimum layer count or seed')
    sizes=options['window_sizes']
    if not sizes or len(set(sizes))!=len(sizes) or any(type(s) is not int or s<1 for s in sizes):
        raise ValueError('inventory window_sizes must be distinct positive integers')
    # No quota applies to a census. Reject leftover fixed-target config rather
    # than silently interpreting "all" as permission to train the entire pool.
    if (options.pop('additional_complex_windows')!='all'
            or config['label_preparation']['record_counts']['train']!='all'):
        raise ValueError('inventory requires all, not a requested training size')
    if any(config['label_preparation']['record_counts'][s]!=old['counts'][s] for s in ('validation','test')):
        raise ValueError('inventory cannot change the fixed evaluation counts')
    semantic={k:v for k,v in options.items() if k not in ('workers','chunk_size')}
    contract={'schema':'rind-complex-inventory-v1','source_index_sha256':labels._digest(old),
              'label_protocol_signature':protocol['signature'],'dataset_manifest':protocol['dataset_manifest'],
              'max_slots':config['model']['max_slots'],'selection':semantic,
              'policy':'catalogue all unique complex windows; diversity caps are a separate diagnostic'}
    signature=labels._digest(contract)
    immutable_json(output/'inventory_protocol.json',{'signature':signature,**contract})
    directory=output/'inventory_chunks';directory.mkdir(exist_ok=True)
    chunks=sorted(directory.glob('chunk-*.json.gz'))
    hashes=[]
    for number,path in enumerate(chunks):
        if path.name!=f'chunk-{number:06d}.json.gz':raise ValueError('inventory chunk sequence has a gap')
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    final=output/'inventory_report.json'
    if final.exists():
        previous=_load(final)
        if previous['signature']!=signature:raise ValueError('inventory report protocol mismatch')
        if previous['scan_complete']:
            if previous['catalogue_chunks_sha256']!=hashes:
                raise ValueError('completed inventory catalogue changed or is incomplete')
            print(f"Inventory already complete: {previous['new_unique_complex_windows']:,} new unique complex windows; no training",flush=True)
            return previous
    import rind_phase2.regions as regions
    import rind_phase2.region_geometry as geometry
    expected={'label_code_sha256':hashlib.sha256(Path(labels.__file__).read_bytes()).hexdigest(),
              'tracer_sha256':hashlib.sha256(inspect.getsource(labels.trace_contribution).encode()).hexdigest(),
              'regions_sha256':hashlib.sha256(Path(regions.__file__).read_bytes()).hexdigest(),
              'geometry_sha256':hashlib.sha256(Path(geometry.__file__).read_bytes()).hexdigest()}
    if any(protocol.get(k)!=v for k,v in expected.items()):
        raise ValueError('inventory label algorithm differs from the source cache')
    if dataset is None:
        from dataset_update import open_dataset
        dataset=open_dataset(resolve(config['data_root']))
    if dataset.manifest!=protocol['dataset_manifest']:raise ValueError('raw data release differs from cached protocol')
    splits=labels._splits(dataset,{'label_preparation':protocol['selection']})
    if labels._digest(splits)!=protocol['split_sha256']:raise ValueError('frozen scene split changed')
    membership={int(s):name for name,scenes in splits.items() for s in scenes}
    if sum(map(len,splits.values()))!=len(membership):raise ValueError('dataset scene splits overlap')
    if any(membership.get(row['scene_id'])!=name for name,rs in old['records'].items() for row in rs):
        raise ValueError('source cache contains a scene outside its frozen split')
    heldout={dataset.geometry_group(int(s)) for name in ('validation','test') for s in splits[name]}
    memo={};seen=set();old_ids=set()
    for row in old['records']['train']:
        geo,key=_configuration_key(dataset,row['scene_id'],memo)
        if geo in heldout:raise ValueError('source training geometry also occurs in held-out pool')
        seen.add(_fingerprint(key,row['window']));old_ids.add((row['scene_id'],row['view_id']))
    base_geometries={geo for geo,key in memo.values()}
    base_configurations={key for geo,key in memo.values()}
    stats=Counter();cursor=0;elapsed_before=0.;unique_configurations=set();unique_geometries=set();joint=Counter()
    capped_configurations=Counter();capped_geometries=Counter();seen_ids=set(old_ids)
    for path in chunks:
        meta=json.loads(gzip.decompress(path.read_bytes()))
        if meta['signature']!=signature or meta['next_serial']<cursor:
            raise ValueError('inventory chunk protocol/cursor mismatch')
        if labels._digest(meta['records'])!=meta['records_sha256']:
            raise ValueError('inventory metadata checksum mismatch')
        for row in meta['records']:
            key,geo=row['configuration_key'],row['geometry_group'];fp=row['configuration_window_fingerprint']
            identity=(row['scene_id'],row['view_id'])
            if identity in seen_ids or fp in seen:raise ValueError('duplicate inventory window')
            if len(row['intensities'])<options['min_layers']:raise ValueError('inventory row is not complex')
            expected_cap=(capped_configurations[key]<options['max_added_per_configuration']
                          and capped_geometries[geo]<options['max_added_per_geometry'])
            if row['within_diversity_caps']!=expected_cap:raise ValueError('inventory cap-selection mismatch')
            if expected_cap:capped_configurations[key]+=1;capped_geometries[geo]+=1
            seen.add(fp);seen_ids.add(identity);unique_configurations.add(key);unique_geometries.add(geo)
            joint[(len(row['intensities']),row['size'])]+=1
        stats=Counter(meta['statistics']);cursor=meta['next_serial'];elapsed_before=meta['processing_seconds']
    if stats['unique_complex']!=len(seen_ids)-len(old_ids):raise ValueError('inventory count does not match its catalogue')
    if stats['within_caps']!=sum(capped_configurations.values()):raise ValueError('inventory capped count mismatch')
    print('Inventory only | no weights, no training, no train-cache assembly | candidate renders=0',flush=True)
    print('Local-label geometry preflight:',regions.check_geometry_backend(),flush=True)
    base_r=Counter(str(len(r['intensities'])) for r in old['records']['train'])
    started=time.perf_counter();pending=[];written=0;committed=stats['checked']
    def report(complete):
        value={'schema':'rind-complex-inventory-report-v1','signature':signature,'scan_complete':complete,
            'training_started':False,'training_cache_created':False,'candidate_renders':0,
            'physical_single_source_checked':False,
            'label_semantics':'minimum-positive/BFS effective layers; complete arithmetic reconstruction required',
            'scope':{'frozen_split':'train','training_scene_pool_size':len(splits['train']),'window_sizes':sizes,'max_views_per_scene':options['max_views_per_scene'],
                     'min_effective_layers':options['min_layers'],'seed':options['seed']},
            'base_training_windows':old['counts']['train'],'base_training_by_R':dict(sorted(base_r.items(),key=lambda kv:int(kv[0]))),
            'base_unique_configurations':len(base_configurations),'base_unique_geometry_groups':len(base_geometries),
            'frozen_evaluation_counts':{s:old['counts'][s] for s in ('validation','test')},
            'new_unique_complex_windows':stats['unique_complex'],
            'new_unique_complex_by_R':_hist(stats,'complex_R_'),
            'new_unique_complex_by_size':_hist(stats,'complex_size_'),
            'new_unique_configurations':len(unique_configurations),'new_unique_geometry_groups':len(unique_geometries),
            'new_geometry_groups_not_in_base':len(unique_geometries-base_geometries),
            'new_unique_complex_by_R_and_size':{str(r):{str(s):n for (rr,s),n in sorted(joint.items()) if rr==r}
                                               for r in sorted({r for r,s in joint})},
            'available_with_current_diversity_caps':stats['within_caps'],
            'capped_complex_by_R':_hist(stats,'capped_R_'),
            'capped_complex_by_size':_hist(stats,'capped_size_'),
            'capped_configurations':len(capped_configurations),'capped_geometry_groups':len(capped_geometries),
            'diversity_caps':{k:options[k] for k in ('max_added_per_configuration','max_added_per_geometry')},
            'possible_total_training_windows_without_caps':old['counts']['train']+stats['unique_complex'],
            'possible_total_training_windows_with_caps':old['counts']['train']+stats['within_caps'],
            'scanned_new_view_ids':stats['checked'],'raw_completed_candidates_by_R':_hist(stats,'raw_complete_R_'),
            'statistics':dict(stats),'catalogue_chunks':len(chunks),'catalogue_chunks_sha256':hashes,
            'next_serial':cursor,'processing_seconds':elapsed_before+time.perf_counter()-started,
            'note':'uncapped/capped availability is within the declared sizes/view limit; catalogue contains metadata only'}
        labels._write_json(final,value)
        return value
    def flush():
        nonlocal pending,written,committed
        meta={'signature':signature,'next_serial':cursor,'statistics':dict(stats),'records':pending,
              'records_sha256':labels._digest(pending),'processing_seconds':elapsed_before+time.perf_counter()-started}
        body=json.dumps(meta,sort_keys=True,allow_nan=False).encode()
        path=directory/f'chunk-{len(chunks):06d}.json.gz';temporary=path.with_suffix('.tmp')
        temporary.write_bytes(gzip.compress(body,mtime=0));temporary.replace(path)
        chunks.append(path);hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        pending=[];written+=1;committed=stats['checked'];report(False)
        print(f'inventory | checked={stats["checked"]:,} | new unique R>={options["min_layers"]}: '
              f'{stats["unique_complex"]:,} | within caps={stats["within_caps"]:,} | no training',flush=True)
        if stop_after_chunks is not None and written>=stop_after_chunks:raise InterruptedError('committed inventory metadata chunk')
    work={**options,'max_slots':config['model']['max_slots'],
          'tolerance':protocol['selection'].get('tolerance',1e-12),'metadata_only':True}
    results=_ordered_results(_tasks(dataset,splits['train'],old_ids,options,cursor),dataset,
                             resolve(config['data_root']),work,options['workers'])
    try:
        for serial,result in results:
            cursor=serial+1;stats['checked']+=1
            if result['status']!='complete':stats[result.get('reason','incomplete')]+=1
            else:
                row=result['row'];r=len(row['intensities']);stats[f'raw_complete_R_{r}']+=1
                if r<options['min_layers']:stats['too_few_layers']+=1
                else:
                    geo,key=_configuration_key(dataset,row['scene_id'],memo)
                    if geo in heldout:raise ValueError('inventory candidate geometry occurs in held-out pool')
                    fp=_fingerprint(key,row['window'])
                    if fp in seen:stats['duplicate_configuration_window']+=1
                    else:
                        config_ok=capped_configurations[key]<options['max_added_per_configuration']
                        geo_ok=capped_geometries[geo]<options['max_added_per_geometry']
                        within=config_ok and geo_ok
                        row.update(configuration_key=key,geometry_group=geo,configuration_window_fingerprint=fp,
                                   within_diversity_caps=within)
                        if not config_ok:stats['over_configuration_cap']+=1
                        if not geo_ok:stats['over_geometry_cap']+=1
                        if within:
                            capped_configurations[key]+=1;capped_geometries[geo]+=1;stats['within_caps']+=1
                            stats[f'capped_R_{r}']+=1;stats[f'capped_size_{row["size"]}']+=1
                        pending.append(row);seen.add(fp);seen_ids.add((row['scene_id'],row['view_id']))
                        unique_configurations.add(key);unique_geometries.add(geo)
                        joint[(r,row['size'])]+=1
                        stats['unique_complex']+=1;stats[f'complex_R_{r}']+=1;stats[f'complex_size_{row["size"]}']+=1
            if len(pending)>=options['chunk_size'] or stats['checked']-committed>=500:flush()
    finally:results.close()
    if pending or stats['checked']!=committed:flush()
    value=report(True)
    print(f'Inventory complete | new unique complex={stats["unique_complex"]:,} | within current caps={stats["within_caps"]:,} '
          '| no model training, no training cache created',flush=True)
    return value
