"""Freeze an existing Phase1 model and inherit its independent scene splits."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from rind_phase1.data import Phase1Dataset,PROJECT_ROOT


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    tmp.replace(path)


def resolve(path):
    p=Path(path).expanduser();return p.resolve() if p.is_absolute() else (PROJECT_ROOT/p).resolve()


def geometry_hash(dataset,scene):
    n=int(dataset._base.obstacle_counts[scene])
    types=dataset._base.obstacle_types[scene,:n];params=dataset._base.obstacle_params[scene,:n]
    return hashlib.sha256(b''.join(sorted(bytes([int(t)])+np.asarray(p,dtype='<f8').tobytes() for t,p in zip(types,params)))).hexdigest()


def select_scenes(available, requested, rng):
    if requested != 'all' and (type(requested) is not int or requested < 1):
        raise ValueError('scene count must be a positive integer or all')
    count=len(available) if requested=='all' else requested
    if not count or count>len(available):
        raise ValueError(f'requires {requested}, available scene count is {len(available)}')
    return [int(s) for s in rng.choice(available,count,replace=False)]


def prepare(config,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=output/'protocol.json'
    if target.is_file():
        result=json.loads(target.read_text())
        schedule_keys={'updates','minimum_scene_visits'}
        if {k:v for k,v in result['config'].items() if k not in schedule_keys}!={k:v for k,v in config.items() if k not in schedule_keys}:raise ValueError('existing RL output uses another config; use a new --run-name')
        if sha(output/'phase1_frozen.pt')!=result['snapshot_sha256']:raise ValueError('frozen Phase1 weights changed')
        return result
    run=resolve(config['phase1_run']);source=run/config.get('variant','response')/'seed_0/best.pt'
    if not source.is_file():raise FileNotFoundError(f'Phase1 best missing: {source}; provide --phase1-run explicitly. No fallback.')
    origin=json.loads((run/'protocol.json').read_text())
    data_root=resolve(config.get('data_root') or origin['config']['data_root'])
    dataset=Phase1Dataset(data_root)
    if hashlib.sha256(json.dumps(dataset.manifest,sort_keys=True).encode()).hexdigest()!=origin['dataset_manifest_sha256']:raise ValueError('locator/data manifest mismatch')
    saved=torch.load(source,map_location='cpu',weights_only=True)
    splits=saved['scene_splits'];ownership={};ids={};geometry_groups={}
    print('RL available source splits: '+str({k:len(v) for k,v in splits.items()})+'; requested: '+str(config['scene_counts']),flush=True)
    excluded=set(config.get('exclude_test_scene_ids',[]));excluded_geometry=set()
    if any(type(s) is not int or s<0 for s in excluded):raise ValueError('invalid excluded test identity')
    for scene in excluded:
        if scene<dataset.num_scenes:excluded_geometry.add(geometry_hash(dataset,scene))
    flat=[s for rows in splits.values() for s in rows]
    if len(flat)!=len(set(flat)) or any(type(s) is not int or not 0<=s<dataset.num_scenes for s in flat):
        raise ValueError('invalid/overlapping pretrained scene identities')
    # Audit exact continuous primitive sets against ALL supplied locator splits,
    # including locator-training geometries outside the selected RL subset.
    for split,rows in splits.items():
        for j,scene in enumerate(rows):
            geometry=geometry_hash(dataset,scene);geometry_groups[str(scene)]=geometry
            previous=ownership.setdefault(geometry,split)
            if previous!=split:raise ValueError('exact geometry leaks between pretrained locator splits')
            if int(dataset._base.driver_counts[scene])!=1:raise ValueError('single-source scenes required')
            if float(dataset._base.driver_strengths[scene,0])!=1.:raise ValueError('unit-strength Phase1 source required')
            if j and j%5000==0:print(f'RL geometry audit {split}: {j}/{len(rows)}',flush=True)
        available=[s for s in rows if split!='test' or (s not in excluded and geometry_groups[str(s)] not in excluded_geometry)]
        want=config['scene_counts'][split]
        rng=np.random.default_rng(np.random.SeedSequence([config['seed'],('train','validation','test').index(split)]))
        try:ids[split]=select_scenes(available,want,rng)
        except ValueError as error:raise ValueError(f'{split}: {error} after exclusions') from error
    world=dataset.manifest['global_size'];tile=config['tile_size'];initial=config['initial_size']
    if world%tile or world%config['spacing'] or initial>world or max(config['evaluation_budgets']+[config['budget']])>world**2:
        raise ValueError('budget/grid outside dataset world')
    starts={}
    pixel=config.get('initial_pixel')
    if pixel is not None and (not isinstance(pixel,list) or len(pixel)!=2 or
            any(type(v) is not int or not 0<=v<world for v in pixel) or initial!=tile):
        raise ValueError('initial_pixel must be [integer x,y] in the world, with a one-tile initial window')
    for split,rows in ids.items():
        for scene in rows:
            rng=np.random.default_rng(np.random.SeedSequence([config['seed'],scene,419]))
            starts[str(scene)]=[int(rng.integers((world-initial)//tile+1))*tile,
                                int(rng.integers((world-initial)//tile+1))*tile,initial]
            if pixel is not None:starts[str(scene)]=[pixel[0]//tile*tile,pixel[1]//tile*tile,tile]
    snapshot={key:saved[key] for key in ('interface_version','model_config','model','epoch','scene_splits')}
    torch.save(snapshot,output/'phase1_frozen.pt')
    result=dict(schema='rind-single-source-active-rl-v1',config=config,data_root=str(data_root),world=world,
        source_checkpoint=str(source),source_sha256=sha(source),source_epoch=saved['epoch'],
        snapshot_sha256=sha(output/'phase1_frozen.pt'),data_manifest_sha256=sha(data_root/'manifest.json'),
        scene_splits=ids,initial_windows=starts,geometry_groups={str(s):geometry_groups[str(s)] for rows in ids.values() for s in rows},exact_geometry_groups_audited=len(ownership),
        locator_had_appended_phase2_layers=bool(origin['config']['training'].get('append_layer_cache_roots')),
        appended_geometry_audit='not independently verified; audit covers the Phase1 data manifest and supplied scene splits',
        actor_inputs=['observed_response','coverage','belief','window','budget','action_geometry'],
        excluded_test_scene_ids=sorted(excluded),
        source_scene_counts={k:len(v) for k,v in splits.items()},
        selected_scene_counts={k:len(v) for k,v in ids.items()},
        validation_selection_scenes=ids['validation'][:config.get('validation_probe_scenes',len(ids['validation']))],
        truth_usage='training reward and evaluation only; never policy inputs',
        inference_fusion=('normalized pointwise minimum over informative unique tiles; constant-response tiles neutral; not a posterior'
            if config.get('fusion_method')=='minimum' else 'fixed disjoint tiles, tempered log opinion pool; not an exact posterior'))
    write_json(target,result)
    print(f'RL prepared | source best={saved["epoch"]} | scenes='+str({k:len(v) for k,v in ids.items()}),flush=True)
    return result
