"""Model-facing observations, immutable family filters, explicit private oracle."""
from __future__ import annotations
from collections import OrderedDict,Counter
from pathlib import Path
import json,os
import numpy as np
import pyarrow.parquet as pq
from .pack import RINDPack,PackedWindow,sha256_file
from .geometry import load_geometry,Oracle,render_region,Budget
from .pipeline import verify_scene
from .candidates import read_references
from .window_contract import WINDOW_SIDES,require_window_side
from rind.storage import pack_arrays

def family_matches(scene,allowed,mode):
    scene,allowed=set(scene),set(allowed)
    if mode=='any':return bool(scene&allowed)
    if mode=='all':return allowed<=scene
    if mode=='subset':return scene<=allowed
    if mode=='exact':return scene==allowed
    raise ValueError('family_match must be any/all/subset/exact')

class RINDDataset:
    def __new__(cls,root=None,*args,**kwargs):
        if root is None:return super().__new__(cls)
        manifest=json.loads((Path(root)/'manifest.json').read_text())
        if manifest.get('catalog_format')=='rind3d-immutable-scene-catalog-v1':
            from rind3d_reader import RINDDataset as Reader
            return Reader(root,*args,**kwargs)
        return super().__new__(cls)
    def __init__(self,root,split='train',include_families=None,exclude_families=None,
                 family_match='subset',window_sizes=None,require_ready=True,
                 view='model',response_format='packed',equivalent_count=None,
                 occupancy_range=None,allow_shortfall=False,cache_scenes=2,cache_bricks=16,legacy_diagnostics=False):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        if self.manifest.get('format')!='RIND3D' or self.manifest.get('version') not in (1,2):raise ValueError('Unsupported dataset manifest')
        from .profiles import validate_manifest_profile,LITE
        profile=validate_manifest_profile(self.manifest)
        if profile.name==LITE:
            from .pilot import config_hash
            from .cli import validate_config
            from .profiles import profile_for
            cfg=self.manifest.get('config',{})
            if self.manifest.get('config_sha256')!=config_hash(cfg) or profile_for(cfg).name!=LITE:raise ValueError('Lite manifest/config identity mismatch')
            validate_config(cfg)
        if view not in ('model','candidate_conditioned'):raise ValueError('Use the explicit private_oracle() for private labels')
        if response_format not in ('packed','bool','torch.bool'):raise ValueError('Unknown response format')
        if legacy_diagnostics and require_ready:raise ValueError('Legacy windows require explicit require_ready=False')
        self.legacy_diagnostics=bool(legacy_diagnostics)
        if self.legacy_diagnostics:
            from .legacy.window_contract import require_window_side as side_validator,WINDOW_SIDES as allowed_sides
        else:side_validator,allowed_sides=require_window_side,WINDOW_SIDES
        if window_sizes is None:window_sizes=allowed_sides
        if not window_sizes:raise ValueError('Select at least one exact window side')
        self._side_validator=side_validator
        selected_sides={side_validator(side) for side in window_sizes}
        if family_match not in ('any','all','subset','exact'):raise ValueError('Invalid family_match')
        self.view=view;self.response_format=response_format;self.require_ready=require_ready
        self.cache_scenes=int(cache_scenes);self.cache_bricks=int(cache_bricks);self.cache=OrderedDict()
        self._pid=os.getpid()
        if self.cache_scenes<1 or self.cache_bricks<0:raise ValueError('Invalid reader cache limits')
        all_scenes=pq.read_table(self.root/'scenes.parquet').to_pylist()
        splits=pq.read_table(self.root/'splits.parquet').to_pylist();splitmap={s['scene_id']:s['split'] for s in splits}
        if len(splitmap)!=len(splits) or len({s['scene_id'] for s in all_scenes})!=len(all_scenes):raise ValueError('Duplicate scene or split assignment')
        if profile.name==LITE:
            groups={}
            for row in all_scenes:
                gid=row.get('geometry_id')
                if not gid:raise ValueError('Lite scene lacks geometry split group')
                if gid in groups and groups[gid]!=row['split']:raise ValueError('Geometry group leaks across splits')
                groups[gid]=row['split']
        selected={}
        for s in all_scenes:
            sid=s['scene_id']
            if Path(sid).name!=sid or sid in ('.','..'):raise ValueError('Invalid scene identifier')
            if splitmap.get(sid)!=s['split']:raise ValueError('Scene/split indices disagree')
            if split is not None and s['split']!=split:continue
            allowed=('READY_LITE',) if profile.name==LITE else ('READY','READY_SHORTFALL')
            if require_ready and s['status'] not in allowed:continue
            if s['status']=='READY_SHORTFALL' and not allow_shortfall:continue
            families=s['families']
            if include_families is not None and not family_matches(families,include_families,family_match):continue
            if exclude_families and set(families)&set(exclude_families):continue
            if occupancy_range is not None:
                if 'solid_fraction' not in s:raise ValueError('Occupancy label is missing; cannot silently ignore the filter')
                if not occupancy_range[0]<=s['solid_fraction']<=occupancy_range[1]:continue
            selected[sid]=s
        self.scenes=selected
        # Only the lightweight index is loaded. No geometry/response is expanded.
        table=pq.read_table(self.root/'windows.parquet',columns=['scene_id','window_id','size','equivalent_count'])
        self.records=[]
        for batch in table.to_batches(max_chunksize=65536):
            for row in batch.to_pylist():
                if row['scene_id'] not in selected:continue
                side=side_validator(row['size'])
                if side not in selected_sides:continue
                if equivalent_count is not None and row['equivalent_count']<equivalent_count:continue
                self.records.append((row['scene_id'],row['window_id'],row['size'],row['equivalent_count']))

    def __len__(self):return len(self.records)

    def _open(self,sid):
        if self._pid!=os.getpid():
            # fork() also inherits open file descriptions and their seek offsets.
            self.close();self._pid=os.getpid()
        if sid in self.cache:
            value=self.cache.pop(sid);self.cache[sid]=value;return value
        path=self.root/'scenes'/sid
        if self.require_ready:
            checked=verify_scene(path)
            if self.manifest.get('dataset_profile')=='rind3d-lite-v0.1':
                meta=json.loads((path/'verification.json').read_text())
                if meta.get('dataset_profile')!=self.manifest['dataset_profile'] or meta.get('config_sha256')!=self.manifest['config_sha256']:raise ValueError('Scene does not belong to this Lite manifest')
                if 'production_revision' in self.manifest and any(meta.get(k)!=self.manifest[k] for k in ('production_revision','quality_identity','production_mode')):raise ValueError('Core scene/manifest research certification differs')
                row=checked['scene_row']
                if any(row.get(k)!=self.scenes[sid].get(k) for k in ('geometry_id','source_id','status','window_count','families','split')):raise ValueError('Lite scene/catalog identity mismatch')
        else:
            meta=json.loads((path/'verification.json').read_text())
            for name,digest in meta['hashes'].items():
                if (path/name).parent!=path or sha256_file(path/name)!=digest:raise ValueError('Staging file hash mismatch')
        if self.scenes[sid].get('verification_sha256')!=sha256_file(path/'verification.json'):raise ValueError('Scene index verification hash mismatch')
        geometry=load_geometry(path/'geometry.npz')
        if geometry.scenes:raise ValueError('Model structure must not serialize private driver records')
        arrays={k:v for k,v in pack_arrays(geometry).items() if not k.startswith(('scene_','physics_'))}
        for array in arrays.values():array.setflags(write=False)
        windows=np.fromfile(path/'window_records.bin',dtype='<u2').reshape(-1,4)
        for side in windows[:,3]:self._side_validator(side)
        counts=np.fromfile(path/'candidate_counts.bin',dtype='<u2')
        if len(counts)!=len(windows):raise ValueError('Canonical window/count tables disagree')
        value={'path':path,'structure':{'format':'canonical-geometry-SoA-v1','arrays':arrays},'windows':windows,
               'counts':counts,
               'response':RINDPack(path/'response.rindb',cache_bricks=self.cache_bricks)}
        self.cache[sid]=value
        while len(self.cache)>self.cache_scenes:
            _,old=self.cache.popitem(last=False);old['response'].close()
        return value

    def __getitem__(self,index):
        sid,wid,side,count=self.records[index];item=self._open(sid)
        if not 0<=wid<len(item['windows']):raise ValueError('Queryable window identifier is out of range')
        x,y,z,length=map(int,item['windows'][wid])
        if length!=side or int(item['counts'][wid])!=count:raise ValueError('Queryable window index disagrees with canonical records')
        observed=item['response'].read_window(x,y,z,length,packed=self.response_format=='packed')
        if self.response_format=='torch.bool':
            import torch
            observed=torch.from_numpy(observed)
        result={'scene_id':sid,'window_id':wid,'window_origin':np.array([x,y,z],dtype=np.int64),
                'window_size':length,'structure':item['structure'],'observed_response':observed}
        if self.view=='candidate_conditioned':result['candidate_sources']=self.private_oracle().get_candidate_sources(sid,wid)
        return result

    def private_oracle(self):return PrivateOracle(self)

    def sampling_weights(self,balance='family'):
        """Opt-in index weights; never generates or rewrites any observations."""
        if balance=='family':keys=[tuple(sorted(self.scenes[s]['families'])) for s,_,_,_ in self.records]
        elif balance=='size':keys=[size for _,_,size,_ in self.records]
        else:raise ValueError('balance must be family or size')
        counts=Counter(keys);return np.array([1/counts[k] for k in keys],dtype=np.float64)

    def close(self):
        for value in self.cache.values():value['response'].close()
        self.cache.clear()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

    def __getstate__(self):
        # DataLoader workers independently reopen bounded reader caches.
        state=self.__dict__.copy();state['cache']=OrderedDict();return state


class PrivateOracle:
    def __init__(self,dataset):self.dataset=dataset
    def _path(self,sid):
        if sid not in self.dataset.scenes:raise KeyError(sid)
        return self.dataset._open(sid)['path']
    def get_true_source(self,sid):return np.fromfile(self._path(sid)/'driver.f64',dtype='<f8').reshape(3)
    def get_candidate_sources(self,sid,window_id):
        path=self._path(sid);meta=json.loads((path/'verification.json').read_text())
        bank=np.fromfile(path/'candidate_bank.bin',dtype='<f8').reshape(-1,3)
        counts=np.fromfile(path/'candidate_counts.bin',dtype='<u2')
        if not 0<=window_id<len(counts):raise IndexError(window_id)
        ids=read_references(path,meta,len(counts),counts)
        return bank[ids[window_id]].copy()
    def get_window_selection_metadata(self,sid,window_id):
        """Provenance, witnesses and scores are explicit private/evaluation data."""
        path=self._path(sid)/'window_selection.parquet'
        if not path.exists():raise KeyError('No stored window-selection provenance for this scene')
        rows=pq.read_table(path,filters=[('window_id','=',int(window_id))]).to_pylist()
        if len(rows)!=1:raise IndexError(window_id)
        return rows[0]
    def get_window_counts(self,sid,window_id):
        """Explicit evaluation-only exact counts; never part of model input.

        Reuses the compact index when present. Older scenes are read-only:
        a transient index is built and discarded without writing into them.
        For repeated ROI queries use CountIndex directly in a context manager.
        """
        from .counts import CountIndex
        path=self._path(sid);records=np.fromfile(path/'window_records.bin',dtype='<u2').reshape(-1,4)
        if not 0<=window_id<len(records):raise IndexError(window_id)
        row=records[window_id]
        with RINDPack(path/'response.rindb',verify_file=True) as y,RINDPack(path/'solid.rindb',verify_file=True) as m:
            p=path/'counts.rindc';index=CountIndex.load(p,y,m) if p.exists() else CountIndex.build(y,m)
            return index.query(tuple(map(int,row[:3])),int(row[3]))
    def read_global_brick(self,sid,brick_xyz,*,packed=False):
        with RINDPack(self._path(sid)/'response.rindb') as r:return r.read_global_brick(brick_xyz,packed=packed)
    def read_global_window(self,sid,origin,size,*,packed=False):
        with RINDPack(self._path(sid)/'response.rindb') as r:return r.read_window(*origin,size,packed=packed)
    def rerender(self,sid,source,origin,size,seconds=30):
        meta=json.loads((self._path(sid)/'verification.json').read_text())
        if any(not isinstance(v,(int,np.integer)) for v in (*origin,size)) or size<1 or min(origin)<0 or max(origin)+size>meta['resolution']:raise ValueError('Invalid rerender region')
        if meta.get('dataset_profile')=='rind3d-lite-v0.1':
            from .rendering.backends import BackendFactory
            from .lite_acceptance import DRIVER_RADIUS,verify_clearance
            with BackendFactory() as factory:
                bundle=load_geometry(self._path(sid)/'geometry.npz');context=factory.context(bundle,meta['numerical_policy'])
                verify_clearance(context,source,context.clearance(source,DRIVER_RADIUS))
                if context.policy_hash!=meta['numerical_policy_sha256']:raise ValueError('Recorded physical implementation is unavailable')
                renderer=context.bind(source);result=np.empty((size,)*3,dtype=bool);budget=Budget(seconds)
                for z in range(0,size,64):
                    for y in range(0,size,64):
                        for x in range(0,size,64):
                            # Generic public rerender accepts arbitrary ROI sizes.
                            extents=np.minimum(64,np.asarray([size-x,size-y,size-z]));side=int(max(extents))
                            requested=np.asarray(origin)+[x,y,z];actual=np.minimum(requested,meta['resolution']-side)
                            offsets=requested-actual;tile=renderer.render(tuple(actual),side,meta['resolution'],budget=budget)[0]
                            dx,dy,dz=extents;ox,oy,oz=offsets
                            result[z:z+dz,y:y+dy,x:x+dx]=tile[oz:oz+dz,oy:oy+dy,ox:ox+dx]
                return result
        oracle=Oracle(load_geometry(self._path(sid)/'geometry.npz'),meta['numerical_policy']);oracle.validate_source(source)
        if meta['status'] in ('READY','READY_SHORTFALL') and oracle.policy_hash!=meta['numerical_policy_sha256']:raise ValueError('Recorded physical implementation is unavailable')
        return render_region(oracle,np.asarray(source,dtype=np.float64),tuple(origin),size,meta['resolution'],budget=Budget(seconds))[0]

def dataset_stats(root):
    root=Path(root);scenes=pq.read_table(root/'scenes.parquet').to_pylist();windows=pq.read_table(root/'windows.parquet').to_pylist()
    return {'scenes':len(scenes),'windows':len(windows),'by_family':dict(Counter('+'.join(sorted(x['families'])) for x in scenes)),
            'by_window_size':dict(Counter(str(x['size']) for x in windows)),'statuses':dict(Counter(x['status'] for x in scenes)),
            'scene_splits':dict(Counter(x['split'] for x in scenes)),
            'candidate_count_distribution':dict(Counter(str(x['equivalent_count']) for x in windows)),'indexed_bytes':sum(x['bytes'] for x in scenes)}

def make_dataloader(dataset,batch_size=4,num_workers=0,seed=0):
    """Exact-side batches keep the three window shapes collatable; geometry stays a list."""
    if type(dataset).__module__.startswith('rind3d_reader'):
        from rind3d_reader import make_dataloader as reader_loader
        return reader_loader(dataset,batch_size,num_workers,seed)
    import torch
    from torch.utils.data import DataLoader
    if batch_size<1 or num_workers<0:raise ValueError('Invalid batch size or worker count')
    batches=[];rng=np.random.default_rng(seed)
    sizes=sorted({x[2] for x in dataset.records})
    for size in sizes:
        ids=np.array([i for i,x in enumerate(dataset.records) if x[2]==size]);rng.shuffle(ids)
        batches.extend(ids[i:i+batch_size].tolist() for i in range(0,len(ids),batch_size))
    rng.shuffle(batches)
    kwargs={'multiprocessing_context':'spawn','prefetch_factor':1} if num_workers else {}
    loader=DataLoader(dataset,batch_sampler=batches,num_workers=num_workers,collate_fn=collate_observations,**kwargs)
    return ObservationLoader(loader)


class ObservationLoader:
    """Tensorize in the parent; workers transfer plain NumPy batches.

    This also avoids the reproduced PyTorch 2.15 CPU tensor-IPC teardown abort
    when a consumer stops before the prefetched epoch is exhausted.
    """
    def __init__(self,loader):self.loader=loader
    @property
    def dataset(self):return self.loader.dataset
    @property
    def batch_sampler(self):return self.loader.batch_sampler
    def __len__(self):return len(self.loader)
    def __iter__(self):
        import torch
        for batch in self.loader:
            for name in ('window_origin','window_id','observed_response'):
                batch[name]=torch.from_numpy(batch[name])
            if 'candidate_sources' in batch:batch['candidate_sources']=[torch.from_numpy(x) for x in batch['candidate_sources']]
            yield batch

def collate_observations(examples):
    """No Tensor payloads in a multiprocessing queue; arrays are owned copies."""
    response=examples[0]['observed_response']
    packed=isinstance(response,PackedWindow)
    values=[x['observed_response'].data if packed else x['observed_response'] for x in examples]
    result={'scene_id':[x['scene_id'] for x in examples],
            'window_id':np.asarray([x['window_id'] for x in examples],dtype=np.int64),
            'window_origin':np.stack([x['window_origin'] for x in examples]),
            'window_size':examples[0]['window_size'],'structure':[x['structure'] for x in examples],
            'observed_response':np.stack(values),
            'response_format':'packed-LSB-X' if packed else 'bool'}
    if 'candidate_sources' in examples[0]:result['candidate_sources']=[x['candidate_sources'].copy() for x in examples]
    return result
