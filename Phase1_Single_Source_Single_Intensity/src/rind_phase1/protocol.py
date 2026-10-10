"""Fixed scene pools and per-scene window choices independent of training size."""
import json
from pathlib import Path
import numpy as np
from .data import PROJECT_ROOT
from .sampling import require_resolved_sampling


def fixed_selection(config,dataset):
    path=PROJECT_ROOT/config['split_manifest']
    manifest=json.loads(path.read_text())
    if manifest['num_scenes'] != dataset.num_scenes:
        raise ValueError('split manifest dataset size mismatch')
    pools=manifest['scene_pools']
    seen=set()
    for name in ('train','validation','test'):
        ids=pools[name]
        if len(ids)!=len(set(ids)) or any(type(x) is not int or not 0<=x<dataset.num_scenes for x in ids) or seen.intersection(ids):
            raise ValueError('invalid or overlapping fixed scene pools')
        seen.update(ids)
    if len(seen)!=dataset.num_scenes: raise ValueError('fixed scene pools must cover the release')
    counts=config['scene_counts']
    splits,observations={},{}
    for name in ('train','validation','test'):
        if counts[name]>len(pools[name]):
            raise ValueError(f'{name} count exceeds its fixed pool of {len(pools[name])}')
        splits[name]=pools[name][:counts[name]]
        observations[name]=[]
        for scene in splits[name]:
            views=dataset._base.local_views[scene,:int(dataset._base.view_counts[scene])]
            for size in config['window_sizes']:
                available=np.flatnonzero(views[:,2]==size).tolist()
                rng=np.random.default_rng(np.random.SeedSequence([manifest['window_seed'],scene,size]))
                available=[available[i] for i in rng.permutation(len(available))]
                preferred=manifest.get('preferred_views',{}).get(f'{scene}:{size}')
                if preferred is not None:
                    if preferred not in available: raise ValueError('preferred window absent from dataset')
                    available.remove(preferred);available.insert(0,preferred)
                if len(available)<config['views_per_size']: raise ValueError('not enough windows')
                observations[name].extend(dict(scene_id=scene,view_id=int(v),size=size) for v in available[:config['views_per_size']])
    return splits,observations,manifest


def validate_research_config(config,world_size=1024):
    if config.get('protocol_version',1)<2:return
    if 'split_manifest' not in config: raise ValueError('v2 requires a frozen split_manifest')
    t=config['training']
    spacing=config['teacher']['spacing']
    if spacing <= 0 or not float(spacing).is_integer() or world_size % spacing:
        raise ValueError('v2 teacher spacing must be a positive integer dividing world_size')
    chunk=t.get('candidate_chunk')
    if chunk is not None and (type(chunk) is not int or chunk < 1):
        raise ValueError('candidate_chunk must be a positive integer')
    require_resolved_sampling(world_size,config['teacher']['spacing'],t['fourier_frequencies'])
    if t.get('normalization_support') not in ('world','geometry'):
        raise ValueError('explicit normalization_support required')
    for k in ('use_obstacle','pixel_coordinates'):
        if type(t.get(k)) is not bool: raise ValueError(f'explicit boolean {k} required')
