"""Two physical brick replays, persistent identity binding and exact coverage."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib,json,time
from pathlib import Path
import numpy as np
from .pack import RINDPack,sha256_file,atomic_json
from .stream import scene_identity

VERSION='two-physical-brick-replays-v1'

def bricks_for(window):
    o=window.origin;s=window.size
    return [(x,y,z) for z in range(o[2]//64,(o[2]+s-1)//64+1) for y in range(o[1]//64,(o[1]+s-1)//64+1) for x in range(o[0]//64,(o[0]+s-1)//64+1)]

def identity(context,source,n,primary,independent,yhash,mhash):
    if primary==independent or primary.get('kind')==independent.get('kind'):raise ValueError('Two different physical implementations required')
    return {'geometry_id':context.geometry_id,'source_bytes':np.asarray(source,dtype='<f8').tobytes().hex(),
            'scene_identity':scene_identity(context,source),'policy':context.policy_hash,'resolution':n,
            'primary_backend':primary,'independent_backend':independent,'response_sha256':yhash,'solid_sha256':mhash}

@dataclass(frozen=True)
class PhysicalEvidence:
    key:dict
    covered:frozenset
    def verify_window(self,window,context,source):
        if self.key['geometry_id']!=context.geometry_id or self.key['policy']!=context.policy_hash or self.key['source_bytes']!=np.asarray(source,dtype='<f8').tobytes().hex():raise ValueError('Physical evidence source/geometry/policy mismatch')
        if min(window.origin)<0 or max(window.origin)+window.size>self.key['resolution']:raise ValueError('Window outside evidence domain')
        if any(tuple(b) not in self.covered for b in bricks_for(window)):raise ValueError('Incomplete physical brick coverage')
        return {'status':'grid_exact','independent_status':'grid_exact','covered_voxels':window.size**3,'proof':VERSION}

def load_evidence(path,expected,windows):
    record=json.loads(Path(path).read_text())
    if record.get('version')!=VERSION or record.get('key')!=expected or not record.get('complete'):raise ValueError('Stale/mismatched physical evidence')
    rows=record.get('bricks',[]);coords=[tuple(r['coordinate']) for r in rows]
    if len(set(coords))!=len(coords) or not rows:raise ValueError('Duplicate/empty physical evidence')
    for r in rows:
        if r.get('primary_sha256')!=r.get('independent_sha256') or len(r.get('primary_sha256',''))!=64:raise ValueError('Physical replays did not agree')
    required={b for w in windows for b in bricks_for(w)}
    if set(coords)!=required:raise ValueError('Physical evidence coverage differs from requested windows')
    stages=record.get('stages',[])
    if len(stages)!=2 or any(s.get('rendered_bricks')!=len(rows) or not s.get('all_response_and_solid_bits_matched') for s in stages):raise ValueError('Incomplete physical evidence stages')
    return PhysicalEvidence(expected,frozenset(coords)),record

def verify_physical_bricks(scene,windows,context,source,*,resume=False,workers=2,seconds=300):
    scene=Path(scene);path=scene/'physical_evidence.private.json';primary=context.bind(source);independent=context.bind(source,independent=True)
    yhash=sha256_file(scene/'response.rindb');mhash=sha256_file(scene/'solid.rindb')
    with RINDPack(scene/'response.rindb',verify_file=True) as y,RINDPack(scene/'solid.rindb',verify_file=True) as m:
        if y.identity!=scene_identity(context,source) or m.identity!=y.identity or y.policy!=context.policy_hash or m.policy!=context.policy_hash or y.shape!=m.shape:raise ValueError('Stored field identity mismatch')
        n=y.shape[0]
    key=identity(context,source,n,primary.identity,independent.identity,yhash,mhash)
    if path.exists():
        if not resume:raise FileExistsError(path)
        return load_evidence(path,key,windows)
    required=sorted({b for w in windows for b in bricks_for(w)});digests={};stages=[];started=time.perf_counter()
    from .geometry import Budget
    for name,renderer in [('primary',primary),('independent',independent)]:
        budget=Budget(seconds,10_000_000_000);t=time.perf_counter();cpu=time.process_time();before=dict(renderer.work);read_seconds=0.
        with RINDPack(scene/'response.rindb',verify_file=True,cache_bricks=0) as y,RINDPack(scene/'solid.rindb',verify_file=True,cache_bricks=0) as m:
            def job(b):
                origin=tuple(a*64 for a in b);pair=renderer.packed_batch([origin],min(64,n),n,budget=budget)[0];return b,pair
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for pos in range(0,len(required),16):
                    for b,pair in pool.map(job,required[pos:pos+16]):
                        tick=time.perf_counter()
                        if pair[0]!=y.read_brick_packed(b) or pair[1]!=m.read_brick_packed(b):raise ValueError(f'Physical {name} mismatch at {b}')
                        read_seconds+=time.perf_counter()-tick;digest=hashlib.sha256(b''.join(pair)).hexdigest()
                        if name=='primary':digests[b]=[digest]
                        else:digests[b].append(digest)
        stages.append({'stage':name,'wall_seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-cpu,
                       'comparison_io_seconds':read_seconds,'rendered_bricks':len(required),'all_response_and_solid_bits_matched':True,
                       'work':{k:v-before.get(k,0) for k,v in renderer.work.items()}})
    record={'version':VERSION,'key':key,'complete':True,'private':True,'stages':stages,
            'bricks':[{'coordinate':list(b),'primary_sha256':v[0],'independent_sha256':v[1]} for b,v in digests.items()],
            'wall_seconds':time.perf_counter()-started,'hashes_are_not_physical_verification':True}
    atomic_json(path,record);return load_evidence(path,key,windows)
