"""A stays parametric clipping/GMP; only the existing subdivision leaf varies.

B is the unchanged certified projection implementation. No decision is shared.
"""
from collections import Counter,OrderedDict
from pathlib import Path
import hashlib,json,os,time
import numpy as np
from .geometry import load_geometry
from .pack import RINDPack,atomic_json,sha256_file
from .rendering.context import GeometryContext
from .projection_research import ProjectionSource
from .physical_evidence import identity as evidence_identity,bricks_for,load_evidence,VERSION as EVIDENCE_VERSION
from .stream import scene_identity
from .core_research import _windows
from .windows import Window
from rind3d_reader.durable import write_json
atomic_json=write_json
_worker_cache=OrderedDict()

class ParametricSource:
    def __init__(self,context,source,*,leaf=8):
        if leaf not in (1,2,4,8):raise ValueError('Unsupported primary leaf')
        self.base=context.bind(source);self.source=np.asarray(source);self.leaf=leaf;self.native=self.base.native;self.work=Counter();self.identity=dict(self.base.identity,subdivision_leaf=leaf,decision_algorithm='unchanged-native-parametric-clipping-GMP',traversal_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def packed_batch(self,origins,side,resolution,*,budget=None):
        result=[]
        for origin in origins:
            if budget:budget.check()
            a,b,stats=self.native.raster(list(origin),side,resolution,self.leaf,False,False);self.work.update(stats)
            result.append(tuple(np.packbits(np.pad(x,((0,64-side),)*3),bitorder='little').tobytes() for x in (a,b)))
        return result

def checkpoint(path,state):
    state.pop('journal_sha256',None);state['journal_sha256']=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest();atomic_json(path,state)

def _windows(desc):return [Window(tuple(w['origin']),w['side'],witnesses=tuple(map(tuple,w['witnesses']))) for w in desc['windows']]

def verify_quantum(desc,soft_seconds=30):
    """Resume identity-bound physical work; return COMPLETE or INCOMPLETE."""
    scene=Path(desc['stage']);cfg=desc['cfg'];start=time.perf_counter();cpu=time.process_time()
    key=(str(scene),desc['geometry_id'],desc['verifier'],desc.get('primary_leaf',8),sha256_file(scene/'geometry.npz'),sha256_file(scene/'driver.f64'),json.dumps(cfg['numerical_policy'],sort_keys=True))
    if key not in _worker_cache:
        ctx=GeometryContext(load_geometry(scene/'geometry.npz'),cfg['numerical_policy']);p=np.fromfile(scene/'driver.f64',dtype='<f8')
        b=ProjectionSource(ctx,p) if desc['verifier']=='silhouette' else ctx.bind(p,independent=True)
        _worker_cache[key]=(ctx,p,[ParametricSource(ctx,p,leaf=desc.get('primary_leaf',8)),b]);
        while len(_worker_cache)>4:_worker_cache.popitem(last=False)
    else:_worker_cache.move_to_end(key)
    ctx,p,models=_worker_cache[key];windows=_windows(desc);coords=sorted({b for w in windows for b in bricks_for(w)})
    if ctx.geometry_id!=desc['geometry_id'] or ('source' in desc and not np.array_equal(p,desc['source'])):raise ValueError('Physical job geometry/source identity differs')
    n=cfg['resolution'];ek=evidence_identity(ctx,p,n,models[0].identity,models[1].identity,sha256_file(scene/'response.rindb'),sha256_file(scene/'solid.rindb'))
    journal=scene/'physical-work.private.json';proof=scene/'physical_evidence.private.json'
    if proof.exists():
        _,r=load_evidence(proof,ek,windows);return {'status':'COMPLETE','evidence_seconds':r['wall_seconds'],'cached_complete_evidence':True}
    if journal.exists():
        state=json.loads(journal.read_text())
        checksum=state.pop('journal_sha256',None)
        if checksum!=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest():raise ValueError('Physical journal checksum mismatch')
        if state['key']!=ek or state['required']!=[list(b) for b in coords]:raise ValueError('Physical journal identity/coverage mismatch')
    else:state={'key':ek,'required':[list(b) for b in coords],'done':[[],[]],'metrics':[{'wall_seconds':0.,'cpu_seconds':0.,'work':{}},{'wall_seconds':0.,'cpu_seconds':0.,'work':{}}],
                'quantums':0,'setup_seconds':0.,'status':'INCOMPLETE','cpu_only_worker':True}
    state['quantums']+=1;processed=0;state['setup_seconds']+=time.perf_counter()-start
    with RINDPack(scene/'response.rindb',verify_file=True,cache_bricks=0) as y,RINDPack(scene/'solid.rindb',verify_file=True,cache_bricks=0) as m:
        if y.shape!=(n,)*3 or m.shape!=y.shape or y.policy!=ctx.policy_hash or m.policy!=y.policy or y.identity!=scene_identity(ctx,p) or m.identity!=y.identity:raise ValueError('Physical field shape/policy/identity differs')
        for backend,model in enumerate(models):
            marker=scene.parents[1]/'phase-markers'/f"{desc['attempt_id']:012d}-{'primary' if backend==0 else 'independent'}.json";write_json(marker,{'attempt_id':desc['attempt_id'],'phase':backend,'pid':os.getpid()})
            before=dict(model.work);t=time.perf_counter();c=time.process_time();position=len(state['done'][backend]);stop=False
            def accrue():
                nonlocal t,c,before
                metrics=state['metrics'][backend];now=time.perf_counter();used=time.process_time()
                metrics['wall_seconds']+=now-t;metrics['cpu_seconds']+=used-c
                for k,v in model.work.items():
                    if k.startswith('max_'):metrics['work'][k]=max(metrics['work'].get(k,0),v)
                    else:metrics['work'][k]=metrics['work'].get(k,0)+v-before.get(k,0)
                t=now;c=used;before=dict(model.work)
            if position>len(coords):raise ValueError('Physical journal count overflow')
            for i,b in enumerate(coords[:position]):
                if state['done'][backend][i]['coordinate']!=list(b):raise ValueError('Physical journal ordering mismatch')
            for b in coords[position:]:
                if processed and time.perf_counter()-start>=soft_seconds:stop=True;break
                pair=model.packed_batch([tuple(k*64 for k in b)],min(n,64),n)[0]
                if pair!=(y.read_brick_packed(b),m.read_brick_packed(b)):raise ValueError(f'Physical backend {backend} mismatch at {b}')
                digest=hashlib.sha256(b''.join(pair)).hexdigest();state['done'][backend].append({'coordinate':list(b),'digest':digest});processed+=1
                if processed%64==0:accrue();checkpoint(journal,state)
            accrue()
            if stop:break
    checkpoint(journal,state)
    complete=all(len(d)==len(coords) for d in state['done'])
    if complete:
        rows=[]
        for a,b in zip(*state['done']):
            if a['coordinate']!=b['coordinate'] or a['digest']!=b['digest']:raise ValueError('Independent physical replay differs')
            rows.append({'coordinate':a['coordinate'],'primary_sha256':a['digest'],'independent_sha256':b['digest']})
        stages=[dict(s,stage=name,rendered_bricks=len(coords),all_response_and_solid_bits_matched=True) for name,s in zip(('primary','independent'),state['metrics'])]
        record={'version':EVIDENCE_VERSION,'key':ek,'complete':True,'private':True,'stages':stages,'bricks':rows,
                'wall_seconds':sum(s['wall_seconds'] for s in stages)+state['setup_seconds'],'quantums':state['quantums'],
                'hashes_are_not_physical_verification':True,'cooperative_soft_seconds':soft_seconds,'setup_seconds':state['setup_seconds']}
        atomic_json(proof,record);load_evidence(proof,ek,windows)
    return {'status':'COMPLETE' if complete else 'INCOMPLETE','quantum_wall_seconds':time.perf_counter()-start,'quantum_cpu_seconds':time.process_time()-cpu,
            'processed_bricks':processed,'done_counts':list(map(len,state['done'])),'required_bricks':len(coords),'soft_threshold_exceeded':time.perf_counter()-start>=soft_seconds}

