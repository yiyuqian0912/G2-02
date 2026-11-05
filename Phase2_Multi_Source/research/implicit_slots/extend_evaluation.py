"""Append natural held-out windows; reuse all training bytes and legacy eval.

Only inexpensive local response decomposition is run. No source candidate
rendering or Phase1 inference. Added windows use geometry groups not present in
the corresponding legacy evaluation set, within the SAME frozen scene split.
"""
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import inspect
import itertools
import json
import multiprocessing
from pathlib import Path
import shutil
import time

import numpy as np

import mask_only_data as labels
from mask_validation import validate_mask_cache
from expand_masks import _digest_arrays
from search_cache import immutable_json

_WORK_DATASET = None
_WORK_OPTIONS = None


def _initialize_worker(data_root, options):
    global _WORK_DATASET, _WORK_OPTIONS
    from dataset_update import open_dataset
    import numba
    numba.set_num_threads(1)
    _WORK_DATASET, _WORK_OPTIONS = open_dataset(data_root), options


def _process(tasks):
    output = []
    for serial, scene, view, side in tasks:
        geometry = _WORK_DATASET.get_scene(scene)
        window = np.asarray(geometry['local_views'][view], dtype=np.int64)
        exact = _WORK_DATASET.get_region(scene, *map(int, window))
        result = labels.local_decomposition(exact['response'], exact['obstacle'], window,
                                            geometry['obstacle_types'], geometry['obstacle_params'],
                                            max_layers=_WORK_OPTIONS['max_slots'], tolerance=_WORK_OPTIONS['tolerance'])
        if result['status'] != 'complete':
            output.append((serial, {'status': result['status'], 'reason': result['reason']}))
            continue
        masks = result['masks']
        if _WORK_OPTIONS.get('metadata_only', False):
            packed = np.packbits(masks.reshape(len(masks), -1), axis=1, bitorder='little')
            label_hash = hashlib.sha256(packed.tobytes() + np.asarray(result['intensities'], dtype='<f8').tobytes()).hexdigest()
            row = {'scene_id':scene, 'view_id':view, 'size':side, 'window':window.tolist(),
                   'layer_count':len(masks), 'intensities':result['intensities'].tolist(),
                   'conflict_regions':result['conflict_regions'], 'unknown_regions':result['unknown_regions'],
                   'reconstruction_max_error':result['reconstruction_max_error'], 'teacher_label_sha256':label_hash}
            output.append((serial, {'status':'complete','row':row}))
            continue
        arrays = {'response': np.asarray(exact['response'], dtype='<f4').ravel(),
                  'obstacle': np.packbits(np.asarray(exact['obstacle'], bool).ravel(), bitorder='little'),
                  'masks': np.packbits(masks.reshape(len(masks), -1), axis=1, bitorder='little').ravel()}
        row = {'scene_id': scene, 'view_id': view, 'size': side, 'window': window.tolist(),
               'layer_count': len(masks), 'intensities': result['intensities'].tolist()}
        output.append((serial, {'status': 'complete', 'row': row, 'arrays': arrays}))
    return output


def _ordered_results(tasks, dataset, data_root, options, workers):
    """Bounded lookahead; retain submission order so runtime cannot bias sampling."""
    global _WORK_DATASET, _WORK_OPTIONS
    if workers == 1:
        _WORK_DATASET, _WORK_OPTIONS = dataset, options
        for task in tasks:
            yield from _process([task])
        return
    def batches():
        iterator = iter(tasks)
        while True:
            part = list(itertools.islice(iterator, 8))
            if not part:
                return
            yield part
    iterator, queue = iter(batches()), deque()
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                   initializer=_initialize_worker, initargs=(str(data_root), options))
    try:
        for _ in range(workers * 2):
            batch = next(iterator, None)
            if batch is not None:
                queue.append(executor.submit(_process, batch))
        while queue:
            future = queue.popleft()
            batch = next(iterator, None)
            if batch is not None:
                queue.append(executor.submit(_process, batch))
            yield from future.result()
    finally:
        for future in queue:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _candidate_tasks(dataset, scenes, legacy_rows, options, split_number):
    legacy_geometries = {dataset.geometry_group(r['scene_id']) for r in legacy_rows}
    order = np.asarray(scenes, dtype=np.int64).copy()
    seed = options['seed']
    np.random.default_rng(seed + split_number).shuffle(order)
    order = [int(s) for s in order if dataset.geometry_group(int(s)) not in legacy_geometries]
    sizes, view_orders = set(options['window_sizes']), {}
    serial = 0
    for round_index in range(options['max_views_per_scene']):
        for scene in order:
            if scene not in view_orders:
                views = [(int(v), int(side)) for v, side in dataset.iter_scene_views(scene) if int(side) in sizes]
                np.random.default_rng(np.random.SeedSequence([seed, split_number, scene])).shuffle(views)
                view_orders[scene] = views[:options['max_views_per_scene']]
            views = view_orders[scene]
            if round_index < len(views):
                view, side = views[round_index]
                yield serial, scene, view, side
                serial += 1


def _append_split(dataset, data_root, source_rows, scenes, split, split_number, output, options,
                  signature, *, stop_after_chunks=None):
    directory = output / 'evaluation_append_chunks' / split
    directory.mkdir(parents=True, exist_ok=True)
    chunks = sorted(directory.glob('chunk-*.npz'))
    cursor, stats = 0, Counter()
    seen = {(r['scene_id'], r['view_id']) for r in source_rows}
    added = 0
    for i, path in enumerate(chunks):
        if path.name != f'chunk-{i:06d}.npz':
            raise ValueError('evaluation chunk sequence contains a gap')
        with np.load(path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored['metadata']))
        if metadata['signature'] != signature:
            raise ValueError('evaluation append protocol changed')
        for row in metadata['records']:
            identity = (row['scene_id'], row['view_id'])
            if identity in seen:
                raise ValueError('duplicate evaluation window in append chunks')
            seen.add(identity)
        added += len(metadata['records'])
        cursor, stats = metadata['cursor'], Counter(metadata['statistics'])
    needed = options['record_counts'][split] - len(source_rows)
    if added > needed:
        raise ValueError('existing append chunks exceed requested evaluation count')
    pending, arrays = [], {name: [] for name in ('response', 'obstacle', 'masks')}
    lengths = Counter()
    committed_checked, commits = stats['checked'], 0
    started = time.perf_counter()
    def flush():
        nonlocal pending, arrays, lengths, added, committed_checked, commits
        payload = {k: np.concatenate(v) if v else np.empty(0, dtype='<f4' if k == 'response' else 'u1')
                   for k, v in arrays.items()}
        metadata = {'signature': signature, 'records': pending, 'cursor': cursor,
                    'statistics': dict(stats), 'arrays_sha256': _digest_arrays(payload.values())}
        path = directory / f'chunk-{len(chunks):06d}.npz'
        with path.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata)), **payload)
        path.with_suffix('.tmp').replace(path)
        chunks.append(path)
        added += len(pending)
        pending, arrays, lengths = [], {name: [] for name in arrays}, Counter()
        committed_checked = stats['checked']
        commits += 1
        print(f'extend {split} | legacy={len(source_rows)} added={added}/{needed} checked={stats["checked"]} | {time.perf_counter()-started:.1f}s | candidate renders=0', flush=True)
        if stop_after_chunks is not None and commits >= stop_after_chunks:
            raise InterruptedError('test interruption after committed evaluation chunk')
    if added < needed:
        tasks = itertools.islice(_candidate_tasks(dataset, scenes, source_rows, options, split_number), cursor, None)
        iterator = _ordered_results(tasks, dataset, data_root, options, options['workers'])
        try:
            for serial, result in iterator:
                cursor = serial + 1
                stats['checked'] += 1
                if result['status'] == 'complete':
                    row = result['row']
                    identity = (row['scene_id'], row['view_id'])
                    if identity in seen:
                        raise ValueError('duplicate candidate in natural evaluation selection')
                    seen.add(identity)
                    for field, name in (('response_offset', 'response'), ('obstacle_offset', 'obstacle'), ('mask_offset', 'masks')):
                        row[field] = lengths[name]
                    for name, value in result['arrays'].items():
                        arrays[name].append(value)
                        lengths[name] += value.size
                    pending.append(row)
                    stats['generated'] += 1
                else:
                    stats[result['reason']] += 1
                enough = added + len(pending) == needed
                if len(pending) >= options['chunk_size'] or stats['checked'] - committed_checked >= 200 or enough:
                    flush()
                if enough:
                    break
        finally:
            iterator.close()
    if pending or stats['checked'] != committed_checked:
        flush()
    if added != needed:
        raise RuntimeError(f'{split}: only {len(source_rows)+added}/{options["record_counts"][split]} complete windows from new held-out geometries; increase max_views_per_scene')
    return chunks


def _assemble(source, output, old, chunks_by_split, contract, signature):
    cache = output / 'mask_cache'
    cache.mkdir(exist_ok=True)
    records = copy.deepcopy(old['records'])
    hashes, selection_statistics = {}, {}
    fields = (('response', 'f32', 4, 'response_offset'), ('obstacle', 'u8', 1, 'obstacle_offset'), ('masks', 'u8', 1, 'mask_offset'))
    for name, suffix, _, _ in fields:
        target, origin = cache / f'train_{name}.{suffix}', source / 'mask_cache' / f'train_{name}.{suffix}'
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != origin.resolve():
                raise ValueError('training cache must remain a read-only source link')
        else:
            target.symlink_to(origin.resolve())
    for split, chunks in chunks_by_split.items():
        streams, offsets = {}, {}
        hashes[split] = []
        for name, suffix, element_size, _ in fields:
            temporary = cache / f'{split}_{name}.{suffix}.tmp'
            shutil.copyfile(source / 'mask_cache' / f'{split}_{name}.{suffix}', temporary)
            streams[name] = temporary.open('ab')
            offsets[name] = temporary.stat().st_size // element_size
        try:
            for path in chunks:
                with np.load(path, allow_pickle=False) as stored:
                    metadata = json.loads(str(stored['metadata']))
                    payload = {name: stored[name] for name in streams}
                content = _digest_arrays(payload.values())
                if content != metadata['arrays_sha256'] or metadata['signature'] != signature:
                    raise ValueError('evaluation chunk checksum/protocol mismatch')
                hashes[split].append(content)
                selection_statistics[split] = metadata['statistics']
                for original in metadata['records']:
                    row = dict(original)
                    for name, _, _, field in fields:
                        row[field] += offsets[name]
                    records[split].append(row)
                for name, value in payload.items():
                    value.tofile(streams[name])
                    offsets[name] += value.size
        finally:
            for stream in streams.values():
                stream.close()
        for name, suffix, _, _ in fields:
            (cache / f'{split}_{name}.{suffix}.tmp').replace(cache / f'{split}_{name}.{suffix}')
    index = {**copy.deepcopy(old), 'records': records,
             'counts': {k: len(v) for k, v in records.items()},
             'corpus_sha256': labels._digest({'source': old['corpus_sha256'], 'chunks': hashes, 'contract': signature}),
             'evaluation_extension': {'signature': signature, 'contract': contract,
                 'source_root': str(source), 'source_index_sha256': labels._digest(old),
                 'legacy_counts': {s: len(old['records'][s]) for s in ('validation', 'test')},
                 'added_selection_statistics': selection_statistics,
                 'training_records_sha256': labels._digest(old['records']['train'])}}
    index.pop('cache_view', None)
    for split in ('validation', 'test'):
        index['statistics'][split] = {'legacy': old.get('statistics', {}).get(split, {}),
                                      'appended': selection_statistics.get(split, {})}
    if 'expansion' in index:
        index['source_expansion_metadata'] = index.pop('expansion')
    index['dataset_distributions'] = {s: {'windows': len(rows), 'scenes': len({r['scene_id'] for r in rows}),
        'effective_layers': dict(Counter(str(len(r['intensities'])) for r in rows)),
        'size': dict(Counter(str(r['size']) for r in rows))} for s, rows in records.items()}
    validate_mask_cache(index, cache, index['max_slots'])
    for split in ('validation', 'test'):
        assert records[split][:len(old['records'][split])] == old['records'][split]
    assert records['train'] == old['records']['train']
    labels._write_json(output / 'mask_records.json', index)
    return index


def prepare_extended_evaluation(config, dataset=None, *, stop_after_chunks=None):
    import fcntl
    from train_decomposition import resolve
    source, output = resolve(config['cache_source_root']).resolve(), resolve(config['output_root']).resolve()
    if source == output:
        raise ValueError('fine-tuning output must differ from the original experiment')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.evaluation_extension.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another process is preparing this evaluation cache') from error
        old = json.loads((source / 'mask_records.json').read_text())
        validate_mask_cache(old, source / 'mask_cache', config['model']['max_slots'])
        expected_train = config.get('label_preparation', {}).get('record_counts', {}).get('train')
        if isinstance(expected_train, int) and old['counts']['train'] != expected_train:
            raise ValueError(f'expected {expected_train} existing training windows, found {old["counts"]["train"]}; select the correct source cache')
        options = dict(config['evaluation_expansion'])
        protocol_root = resolve(options.pop('label_protocol_root'))
        protocol = json.loads((protocol_root / 'mask_protocol.json').read_text())
        for key in ('workers', 'chunk_size', 'max_views_per_scene'):
            if type(options.get(key)) is not int or options[key] < 1:
                raise ValueError(f'evaluation_expansion.{key} must be positive')
        for split in ('validation', 'test'):
            count = options.get('record_counts', {}).get(split)
            if type(count) is not int or count < old['counts'][split]:
                raise ValueError('expanded evaluation cannot drop legacy windows')
        semantic = {k: v for k, v in options.items() if k not in ('workers', 'chunk_size')}
        contract = {'schema': 'rind-natural-evaluation-extension-v1', 'source_index_sha256': labels._digest(old),
                    'label_protocol_signature': protocol['signature'], 'selection': semantic,
                    'max_slots': config['model']['max_slots'], 'dataset_manifest': protocol['dataset_manifest'],
                    'added_geometry_policy': 'exclude legacy geometry groups within each frozen split'}
        signature = labels._digest(contract)
        immutable_json(output / 'evaluation_extension_protocol.json', {'signature': signature, **contract})
        final = output / 'mask_records.json'
        if final.exists():
            index = json.loads(final.read_text())
            if index.get('evaluation_extension', {}).get('signature') != signature:
                raise ValueError('existing cache belongs to a different evaluation extension')
            validate_mask_cache(index, output / 'mask_cache', config['model']['max_slots'])
            print(f'Evaluation cache already complete: {index["counts"]}; no data reread or candidate renders', flush=True)
            return index
        import rind_phase2.regions as regions
        import rind_phase2.region_geometry as geometry
        expected = {'label_code_sha256': hashlib.sha256(Path(labels.__file__).read_bytes()).hexdigest(),
                    'tracer_sha256': hashlib.sha256(inspect.getsource(labels.trace_contribution).encode()).hexdigest(),
                    'regions_sha256': hashlib.sha256(Path(regions.__file__).read_bytes()).hexdigest(),
                    'geometry_sha256': hashlib.sha256(Path(geometry.__file__).read_bytes()).hexdigest()}
        if any(protocol.get(k) != v for k, v in expected.items()):
            raise ValueError('source teacher algorithm differs; existing and new evaluation labels cannot be mixed')
        if dataset is None:
            from dataset_update import open_dataset
            try:
                dataset = open_dataset(resolve(config['data_root']))
            except FileNotFoundError as error:
                raise FileNotFoundError('Expanding held-out windows needs the original RIND dataset; existing training labels are reused, but new observations are not in the old 2000-window cache.') from error
        if dataset.manifest != protocol['dataset_manifest']:
            raise ValueError('raw dataset release differs from cached label protocol')
        splits = labels._splits(dataset, {'label_preparation': protocol['selection']})
        if labels._digest(splits) != protocol['split_sha256']:
            raise ValueError('frozen scene/geometry split changed')
        membership = {int(scene): split for split, scenes in splits.items() for scene in scenes}
        if any(membership.get(row['scene_id']) != split for split, rows in old['records'].items() for row in rows):
            raise ValueError('source rows no longer belong to the frozen dataset split')
        print('Local-label geometry preflight:', regions.check_geometry_backend(), flush=True)
        work = {**options, 'max_slots': config['model']['max_slots'],
                'tolerance': protocol['selection'].get('tolerance', 1e-12)}
        chunks = {split: _append_split(dataset, resolve(config['data_root']), old['records'][split], splits[split],
                    split, i, output, work, signature, stop_after_chunks=stop_after_chunks)
                  for i, split in enumerate(('validation', 'test'), 1)}
        index = _assemble(source, output, old, chunks, contract, signature)
        print(f'Natural evaluation extension complete: {index["counts"]}; training cache unchanged; candidate renders=0', flush=True)
        return index
