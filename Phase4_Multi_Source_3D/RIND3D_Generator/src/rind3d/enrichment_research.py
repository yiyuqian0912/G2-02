"""Optional, immutable ambiguity enrichment. Core uses no discovery work.

All source-space constructions are proposals. Every published witness must
pass clearance, window exclusion, separation and complete independent A/B
physical comparison. Artifacts live outside frozen scenes.
"""
from collections import Counter,OrderedDict
from pathlib import Path
import hashlib,time
import numpy as np
from .geometry import Budget,BudgetExhausted
from .lite_acceptance import DRIVER_RADIUS,verify_clearance
from .pack import sha256_file,atomic_json
from .physical_evidence import bricks_for
from .windows import excludes,sobol_points
from .observation_research import sparse_observations,transition_observations
from .inverse_research import InverseConstraints,explore,local_proposals,silhouette_proposals
from .search import separated

VERSION='rind3d-private-enrichment-research-v1'

class SharedCandidateVerifier:
    def __init__(self,context,*,cache_bricks=256,independent_factory=None,cache_sources=32):
        self.context=context;self.cache=OrderedDict();self.capacity=cache_bricks;self.stats=Counter();self.independent_factory=independent_factory
        self.models=OrderedDict();self.source_limit=int(cache_sources)
        if self.capacity<1 or self.source_limit<1:raise ValueError('Positive candidate cache bounds required')
    def verify(self,source,window,response,solid,*,independent,budget):
        skey=np.asarray(source,dtype='<f8').tobytes();key=(skey,independent)
        if key not in self.models:
            self.models[key]=self.independent_factory(source) if independent and self.independent_factory else self.context.bind(source,independent=independent)
            while len(self.models)>self.source_limit:self.models.popitem(last=False)
        else:self.models.move_to_end(key)
        renderer=self.models[key];total=0;digests=[]
        for b in bricks_for(window):
            budget.check();cache_key=(self.context.geometry_id,self.context.policy_hash,skey,str(renderer.identity),b)
            if cache_key in self.cache:pair=self.cache.pop(cache_key);self.cache[cache_key]=pair;self.stats['brick_cache_hits']+=1
            else:
                pair=renderer.packed_batch([tuple(v*64 for v in b)],64,1024,budget=budget)[0];self.cache[cache_key]=pair;self.stats['physical_bricks']+=1
                while len(self.cache)>self.capacity:self.cache.popitem(last=False)
            lo=np.maximum(np.asarray(b)*64,window.origin);hi=np.minimum(np.asarray(b)*64+64,np.asarray(window.origin)+window.size)
            shape=tuple((hi-lo)[::-1]);bo=lo-np.asarray(b)*64
            actual=[np.unpackbits(np.frombuffer(v,dtype=np.uint8),bitorder='little').reshape((64,)*3)[bo[2]:bo[2]+shape[0],bo[1]:bo[1]+shape[1],bo[0]:bo[0]+shape[2]].astype(bool) for v in pair]
            # Compare only the observation intersection; differences outside W
            # never reject an otherwise equivalent source.
            wlo=lo-np.asarray(window.origin);expected=[v[wlo[2]:wlo[2]+shape[0],wlo[1]:wlo[1]+shape[1],wlo[0]:wlo[0]+shape[2]] for v in (response,solid)]
            if any(not np.array_equal(a,e) for a,e in zip(actual,expected)):return {'status':'mismatch','covered_voxels':total}
            total+=int(np.prod(shape));digests.append(hashlib.sha256(np.packbits(np.stack(actual),bitorder='little').tobytes()).hexdigest())
        if total!=window.size**3:raise AssertionError('Incomplete candidate physical coverage')
        return {'status':'grid_exact','covered_voxels':total,'backend':renderer.identity,'intersection_digests':digests}

def search(context,truth,window,y,m,*,seconds=0,method='random',seed=0,verifier=None,max_candidates=8,pool=None,probe_count=32,probe_strategy='spatial'):
    start=time.perf_counter();cpu=time.process_time();stats=Counter();retained=[np.array(truth,copy=True)];proofs=[];diagnostics={};verifier=verifier or SharedCandidateVerifier(context)
    if seconds<0 or max_candidates<1 or max_candidates>8:raise ValueError('Invalid enrichment budget/cap')
    if seconds==0 or max_candidates==1:return np.asarray(retained),[],{'optional_search_disabled':True,'actual_candidates':1,'elapsed_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'uniqueness_proven':False,'preparation_performed':False}
    budget=Budget(seconds,1_000_000_000);termination='proposal_pool_exhausted'
    try:
        t=time.perf_counter()
        if probe_strategy=='transition':points,labels,diagnostics['observations']=transition_observations(y,m,window,count=probe_count,seed=seed)
        elif probe_strategy=='spatial':points,labels=sparse_observations(y,m,window,count=probe_count,seed=seed)
        else:raise ValueError('Unknown sparse observation strategy')
        observed=y.read_window(*window.origin,window.size);mask=m.read_window(*window.origin,window.size)
        constraints=None
        if method in ('silhouette','inverse','hybrid'):
            # Sixteen class/spatially stratified necessary constraints; the full
            # 32-probe screen and complete A/B verification still follow.
            ids=np.arange(min(16,len(points))) if probe_strategy=='transition' else np.r_[np.flatnonzero(labels)[:8],np.flatnonzero(~labels)[:8]]
            constraints=InverseConstraints(context,points[ids],labels[ids])
            diagnostics['inverse_preparation']={'observations':len(ids),'wall_seconds':constraints.setup_seconds,'cpu_seconds':constraints.setup_cpu_seconds}
        base=sobol_points(seed,0,771,1024) if pool is None else np.asarray(pool)
        if method=='random':proposals=base
        elif method=='local':proposals=local_proposals(truth,count=1024,seed=seed)
        elif method=='silhouette':proposals,diagnostics['silhouette']=silhouette_proposals(constraints,truth,count=1024,seed=seed)
        elif method in ('inverse','hybrid'):
            remaining=max(0,seconds-(time.perf_counter()-start));proposals,diagnostics['inverse']=explore(constraints,truth,window,seconds=remaining*.4,max_regions=512,max_depth=5,seed=seed)
            if method=='hybrid':
                local=local_proposals(truth,count=256,seed=seed);chunks=[]
                for i in range(0,max(len(proposals),256),8):chunks.extend((proposals[i:i+8],local[i:i+8],base[i:i+8]))
                proposals=np.concatenate([a for a in chunks if len(a)],axis=0)
        else:raise ValueError('Unknown candidate construction')
        stats['preparation_microseconds']=round((time.perf_counter()-t)*1e6);used=set();budget.check()
        for p in proposals:
            budget.check();p=np.asarray(p,dtype='<f8');key=p.tobytes()
            if key in used:stats['duplicate_proposals']+=1;continue
            used.add(key)
            if not np.isfinite(p).all() or np.any(p<=0) or np.any(p>=1):stats['outside_world']+=1;continue
            if not excludes(window,p,1024):stats['inside_window']+=1;continue
            if any(not separated(p,q,32/1024) for q in retained):stats['too_close']+=1;continue
            stats['proposals_evaluated']+=1
            if context.contains(p[None],budget=budget)[0]:stats['solid']+=1;continue
            q=context.clearance(p,DRIVER_RADIUS)
            if q['status']!='GT':stats['clearance_LE']+=1;continue
            verify_clearance(context,p,q);renderer=context.bind(p);t=time.perf_counter()
            # Native batch screening; strict full A/B below is still required.
            predicted=renderer.response(p,points,budget=budget);stats['probe_points']+=len(points);stats['probe_microseconds']+=round((time.perf_counter()-t)*1e6)
            if not np.array_equal(predicted,labels):stats['probe_rejected']+=1;continue
            stats['probe_survivors']+=1;t=time.perf_counter();first=verifier.verify(p,window,observed,mask,independent=False,budget=budget);stats['primary_microseconds']+=round((time.perf_counter()-t)*1e6)
            if first['status']!='grid_exact':stats['full_mismatch']+=1;continue
            t=time.perf_counter();second=verifier.verify(p,window,observed,mask,independent=True,budget=budget);stats['independent_microseconds']+=round((time.perf_counter()-t)*1e6)
            if second['status']!='grid_exact':raise RuntimeError('Independent full-window candidate disagreement')
            budget.check();retained.append(p.copy());proofs.append({'source':p.tolist(),'clearance':q,'primary':first,'independent':second})
            if len(retained)==max_candidates:termination='max_candidates';break
    except BudgetExhausted as exc:termination='incomplete_optional_'+exc.kind+'_budget'
    return np.asarray(retained),proofs,{'method':method,'probe_strategy':probe_strategy,'probe_count_requested':probe_count,'seed':seed,'declared_seconds':seconds,'elapsed_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
            'actual_candidates':len(retained),'additional_sources':len(retained)-1,'unique_nontruth_coordinates':len({p.tobytes() for p in retained[1:]}),
            'stats':dict(stats),'termination':termination,'diagnostics':diagnostics,'verification_cache':dict(verifier.stats),
            'uniqueness_proven':False,'all_retained_additions_independently_full_window_verified':True,'source_domain':'world_outside_window'}

def save_artifact(path,scene,records):
    import os,tempfile,json
    path=Path(path);scene=Path(scene)
    if path.exists():raise FileExistsError('Enrichment artifacts are immutable; choose a new version')
    if scene.resolve() in path.resolve().parents:raise ValueError('Enrichment must not mutate the scene')
    hashes={name:sha256_file(scene/name) for name in ('geometry.npz','driver.f64','response.rindb','solid.rindb','window_records.bin','verification.json')}
    value={'format':VERSION,'private':True,'bound_scene_files':hashes,'scene_id':scene.name,'records':records,'not_a_uniqueness_certificate':True}
    path.parent.mkdir(parents=True,exist_ok=True);temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.'+path.name+'.',delete=False) as f:
            temporary=Path(f.name);json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        # Atomic no-replace publication, including concurrent writers. The
        # completed temporary inode is linked only if this version is absent.
        os.link(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)

def read_artifact(path,scene,*,physical_recheck=True,seconds=300):
    """Explicit private API. Default independently rechecks every extra source.

    This never updates a READY scene, its catalog, or the ordinary model input.
    """
    import json
    from .geometry import load_geometry
    from .rendering.context import GeometryContext
    from .pack import RINDPack
    from .windows import Window
    scene=Path(scene);record=json.loads(Path(path).read_text());meta=json.loads((scene/'verification.json').read_text())
    from .pipeline import verify_scene
    verify_scene(scene)
    if record.get('format')!=VERSION or record.get('private') is not True or record.get('scene_id')!=meta.get('scene_id') or meta.get('status')!='READY_LITE':raise ValueError('Unsupported enrichment/scene identity')
    required={'geometry.npz','driver.f64','response.rindb','solid.rindb','window_records.bin','verification.json'}
    if set(record.get('bound_scene_files',{}))!=required or any(sha256_file(scene/name)!=digest for name,digest in record['bound_scene_files'].items()):raise ValueError('Stale enrichment artifact')
    ctx=GeometryContext(load_geometry(scene/'geometry.npz'),meta['numerical_policy']);truth=np.fromfile(scene/'driver.f64',dtype='<f8');windows=np.fromfile(scene/'window_records.bin',dtype='<u2').reshape(-1,4);result={};budget=Budget(seconds,10_000_000_000);verifier=SharedCandidateVerifier(ctx)
    with RINDPack(scene/'response.rindb',verify_file=True) as y,RINDPack(scene/'solid.rindb',verify_file=True) as m:
        for item in record['records']:
            wid=item['window_id']
            if not isinstance(wid,int) or not 0<=wid<len(windows) or wid in result:raise ValueError('Duplicate/invalid enriched window')
            raw=windows[wid];w=Window(tuple(map(int,raw[:3])),int(raw[3]));points=np.asarray(item['positions'],dtype='<f8')
            if points.ndim!=2 or points.shape[1]!=3 or not 1<=len(points)<=8 or not np.array_equal(points[0],truth) or len(item['proofs'])!=len(points)-1:raise ValueError('Enrichment reference contract mismatch')
            for i,p in enumerate(points):
                if not np.isfinite(p).all() or np.any(p<=0) or np.any(p>=1) or not excludes(w,p,1024) or any(not separated(p,q,32/1024) for q in points[:i]):raise ValueError('Invalid enriched source domain/separation')
            observed=mask=None
            for p,proof in zip(points[1:],item['proofs']):
                if not np.array_equal(p,proof['source']):raise ValueError('Enrichment source/proof differs')
                verify_clearance(ctx,p,proof['clearance']);a,b=proof['primary'],proof['independent']
                if any(r.get('status')!='grid_exact' or r.get('covered_voxels')!=w.size**3 for r in (a,b)) or a['backend']['kind']==b['backend']['kind'] or a['intersection_digests']!=b['intersection_digests'] or len(a['intersection_digests'])!=len(bricks_for(w)):raise ValueError('Incomplete independent enrichment evidence')
                if physical_recheck:
                    if observed is None:observed=y.read_window(*w.origin,w.size);mask=m.read_window(*w.origin,w.size)
                    for independent in (False,True):
                        if verifier.verify(p,w,observed,mask,independent=independent,budget=budget)['status']!='grid_exact':raise ValueError('Enrichment physical mismatch')
            result[wid]=points.copy()
    return result
