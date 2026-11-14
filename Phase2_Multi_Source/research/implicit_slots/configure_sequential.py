"""Freeze a v5 warm-start and bind its COMPLETE large cache; never regenerate."""
import argparse,copy,json,re,sys,math
from collections import Counter
from pathlib import Path
from train_decomposition import load_config,resolve
from search_cache import immutable_json
from mask_validation import validate_mask_cache

PHASE2=Path(__file__).resolve().parents[2]


def configure(source_session='rind_masks_disentangled_more',source_root=None,run_name=None,expected_source_epoch=None):
    if expected_source_epoch is not None and (type(expected_source_epoch) is not int or expected_source_epoch<0):
        raise ValueError('expected source epoch must be a nonnegative integer')
    if source_root is None:
        pointer=PHASE2/'outputs/disentangled_v5_launcher'/f'{source_session}.txt'
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',source_session):raise ValueError('invalid source session')
        if not pointer.is_file():raise FileNotFoundError(f'Missing source session pointer: {pointer}; pass --source-root outputs/YOUR_EXISTING_RUN')
        old=load_config(Path(pointer.read_text().strip()))
        source=resolve(old['output_root']).resolve()
    else:source=resolve(source_root).resolve()
    index_path=source/'mask_records.json'
    if not index_path.is_file():raise FileNotFoundError(f'Completed large cache missing: {index_path}; no small fallback is permitted')
    index=json.loads(index_path.read_text())
    validate_mask_cache(index,source/'mask_cache',index['max_slots'])
    template=load_config(PHASE2/'configs/masks_sequential.json')
    name=run_name or (source.name+'_implicit_v6')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',name):raise ValueError('invalid run name')
    output=PHASE2/'outputs'/name
    if output.resolve()==source:raise ValueError('source and destination must differ')
    destination=PHASE2/'outputs/launch_configs'/f'{name}.json'
    if destination.is_file():
        existing=load_config(destination)
        if resolve(existing['cache_source_root']).resolve()!=source:raise ValueError('existing run uses another dataset')
        if (existing['model'].get('residual_gradient')!='straight_through' or
                existing['training'].get('loss_normalization')!='optimizer_step'):
            raise ValueError('earlier v6 optimization protocol; choose a new --run-name to preserve its checkpoint')
        protocol=existing.get('source_training_protocol',{})
        frozen_epoch=protocol.get('warm_start',{}).get('epoch')
        if expected_source_epoch is not None and frozen_epoch!=expected_source_epoch:
            raise ValueError(f'existing run froze epoch {frozen_epoch}, expected {expected_source_epoch}; choose a new --run-name and --session')
        if protocol.get('corpus_sha256')!=index['corpus_sha256'] or protocol.get('counts')!=index['counts']:
            raise ValueError('source cache changed since launch; choose a new run name')
        print(f'Resume config: {destination}; complete cached windows={index["counts"]}',file=sys.stderr)
        return destination
    checkpoint=source/'models/seed_0/best.pt'
    if not checkpoint.is_file():raise FileNotFoundError(f'Warm-start best checkpoint missing: {checkpoint}')
    from evaluation_snapshot import create_snapshot
    meta=create_snapshot(checkpoint,output/'warm_start')
    if expected_source_epoch is not None and meta['epoch']!=expected_source_epoch:
        raise ValueError(f'source best is epoch {meta["epoch"]}, expected {expected_source_epoch}; training not started')
    import torch
    saved=torch.load(meta['snapshot_path'],map_location='cpu',weights_only=True)
    old=saved['config']
    if old['model'].get('decomposition_architecture')!='disentangled_slots_v5':
        raise ValueError('v6 warm-start expects the current v5 architecture')
    config=copy.deepcopy(template)
    # Keep the exact source encoder width/capacity and every original record.
    config['model']={**copy.deepcopy(old['model']),'decomposition_architecture':'sequential_count_shape_v6','roundoff_ulps':8,
                     'residual_gradient':'straight_through'}
    config['data_root']=old.get('data_root',config['data_root'])
    config['output_root']=str(output);config['cache_source_root']=str(source)
    config['name']=name;config['label_preparation']=copy.deepcopy(old['label_preparation'])
    config['label_preparation']['record_counts']=copy.deepcopy(index['counts'])
    # Weights use training metadata only. Retain every cache record; regular
    # DDP tail padding may repeat a record, without replacement sampling or
    # validation/test frequency leakage.
    group=lambda r:min(r,5)
    frequencies=Counter(group(len(row['intensities'])) for row in index['records']['train'])
    layers=Counter()
    for row in index['records']['train']:layers[group(len(row['intensities']))]+=len(row['intensities'])
    def weights(freq):
        total=sum(freq.values());present=len(freq)
        return [1.]+[min(4.,max(.25,math.sqrt(total/(present*freq[group(r)])))) if freq[group(r)] else 1.
                     for r in range(1,index['max_slots']+1)]
    config['loss']['count_class_weights']=weights(frequencies)
    config['loss']['shape_class_weights']=weights(layers)
    config['training']['initial_decomposition_checkpoint']=meta['snapshot_path']
    config['source_training_protocol']={'source_root':str(source),'counts':index['counts'],
        'corpus_sha256':index['corpus_sha256'],'warm_start':meta,'raw_data_required':False,
        'new_teacher_rendering':False,'all_records_preserved':True,
        'requested_warm_start_epoch':expected_source_epoch}
    immutable_json(destination,config)
    print(f'Full cached train/validation/test: {index["counts"]}; no filtering/subsampling/re-rendering',file=sys.stderr)
    print(f'Frozen v5 best epoch={meta["epoch"]}; new output={output}',file=sys.stderr)
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-session',default='rind_masks_disentangled_more')
    p.add_argument('--source-root');p.add_argument('--run-name')
    p.add_argument('--expected-source-epoch',type=int)
    a=p.parse_args();print(configure(a.source_session,a.source_root,a.run_name,a.expected_source_epoch))
