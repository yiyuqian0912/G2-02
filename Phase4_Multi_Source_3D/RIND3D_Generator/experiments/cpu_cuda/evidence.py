"""Identity-bound physical replays reusable by overlapping windows.

These are audit records of exhaustive physical re-evaluation, not a claim that
file hashes alone prove physics. No response read is used as an independent
oracle. Each referenced brick is freshly rendered by two implementations.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
import numpy as np
from rind3d.pack import RINDPack, sha256_file, atomic_json
from rind3d.stream import scene_identity


def intersecting_bricks(w):
    o=w.origin; l=w.size
    return [(x,y,z) for z in range(o[2]//64,(o[2]+l-1)//64+1)
            for y in range(o[1]//64,(o[1]+l-1)//64+1)
            for x in range(o[0]//64,(o[0]+l-1)//64+1)]


@dataclass(frozen=True)
class BrickEvidence:
    identity: str
    policy: str
    resolution: int
    source_bytes: bytes
    response_sha256: str
    solid_sha256: str
    verified: frozenset
    primary_backend: str
    independent_backend: str

    def validate_files(self, path):
        # Required at the boundary of a new reuse session; hashes bind already
        # computed physical evidence to files, rather than proving the physics.
        for name,digest in [('response',self.response_sha256),('solid',self.solid_sha256)]:
            if sha256_file(Path(path)/(name+'.rindb'))!=digest:
                raise ValueError('Evidence file content changed')

    def verify_window(self, w, oracle, source):
        if self.identity!=scene_identity(oracle,source) or self.policy!=oracle.policy_hash:
            raise ValueError('Evidence geometry/source/policy identity mismatch')
        if np.asarray(source,dtype='<f8').tobytes()!=self.source_bytes:
            raise ValueError('Evidence source mismatch')
        if min(w.origin)<0 or any(v+w.size>self.resolution for v in w.origin):
            raise ValueError('Window outside evidence field')
        bricks=intersecting_bricks(w)
        if not all(b in self.verified for b in bricks): raise ValueError('Incomplete physical evidence coverage')
        return {'status':'grid_exact','covered_voxels':w.size**3,
                'proof':'two-physical-brick-replays-cover-every-window-voxel',
                'independent_status':'grid_exact','bricks':len(bricks)}


def build_evidence(path,windows,primary,independent,*,primary_backend,independent_backend,output=None):
    if primary_backend==independent_backend: raise ValueError('Separate physical implementations required')
    if not np.array_equal(primary.source,independent.source): raise ValueError('Verifier source mismatch')
    identity=scene_identity(primary,primary.source)
    if scene_identity(independent,independent.source)!=identity: raise ValueError('Verifier identity mismatch')
    wanted=sorted({b for w in windows for b in intersecting_bricks(w)})
    stages=[]; digests={}; n=None
    for label,oracle,backend in [('primary',primary,primary_backend),('independent',independent,independent_backend)]:
        t=time.perf_counter(); cpu=time.process_time(); render_cpu=0.; compare_seconds=0.; seen=set(); before=dict(oracle.work)
        with RINDPack(Path(path)/'response.rindb',verify_file=True,cache_bricks=0) as response, RINDPack(Path(path)/'solid.rindb',verify_file=True,cache_bricks=0) as solid:
            if response.identity!=identity or solid.identity!=identity or response.policy!=oracle.policy_hash or solid.policy!=oracle.policy_hash:
                raise ValueError('Evidence input identity mismatch')
            if response.shape!=solid.shape or len(set(response.shape))!=1: raise ValueError('Evidence shape mismatch')
            n=response.shape[0]
            def job(b):
                start=time.perf_counter(); origin=tuple(v*64 for v in b); side=min(64,n-min(origin))
                y,m,_=oracle.render(oracle.source,origin,side,n)
                pair=[]
                for a in (y,m):
                    padded=np.zeros((64,)*3,dtype=bool);padded[:side,:side,:side]=a
                    pair.append(np.packbits(padded,bitorder='little').tobytes())
                return b,pair,time.perf_counter()-start
            # map keeps comparisons deterministic and bounds in-flight memory.
            with ThreadPoolExecutor(max_workers=2) as executor:
                for start in range(0,len(wanted),16):
                    for b,pair,dt in executor.map(job,wanted[start:start+16]):
                        render_cpu+=dt; tick=time.perf_counter()
                        for reader,raw in zip((response,solid),pair):
                            if reader.read_brick_packed(b)!=raw: raise AssertionError(f'Physical replay mismatch {label} {b}')
                        compare_seconds+=time.perf_counter()-tick;seen.add(b)
                        digest=hashlib.sha256(b''.join(pair)).hexdigest()
                        if label=='primary':digests[','.join(map(str,b))]=digest
                        elif digests[','.join(map(str,b))]!=digest:raise AssertionError('Independent replay differs')
            if len(seen)!=len(wanted):raise AssertionError('Incomplete replay')
        stages.append({'stage':label,'backend':backend,'wall_seconds':time.perf_counter()-t,
                       'cpu_seconds':time.process_time()-cpu,'worker_seconds':render_cpu,'comparison_io_seconds':compare_seconds,
                       'freshly_rendered_bricks':len(seen),'verified_voxels':int(sum(min(64,n-min(np.array(b)*64))**3 for b in seen)),
                       'work':{k:v-before.get(k,0) for k,v in oracle.work.items()}})
    proof=BrickEvidence(identity,primary.policy_hash,n,primary.source.tobytes(),sha256_file(Path(path)/'response.rindb'),sha256_file(Path(path)/'solid.rindb'),frozenset(wanted),primary_backend,independent_backend)
    report={'identity':identity,'policy':proof.policy,'geometry_id':primary.shared.geometry_id,'source_id':primary.source_id,
            'resolution':n,'response_sha256':proof.response_sha256,'solid_sha256':proof.solid_sha256,
            'stages':stages,'brick_digests':digests,'all_physical_replays_passed':True,'independent_implementations':True,
            'stored_response_was_not_an_oracle':True,'private':True,'production_ready':False}
    if output:atomic_json(output,report)
    return proof,report


def demo():
    import json,shutil
    from rind3d.geometry import load_geometry
    from rind3d.windows import Window
    from rind3d import RINDDataset
    from .cpu import GeometryGroup
    from .benchmark import Control,ROOT,OLD
    from experiments.shadow_volumes.cases import SOURCES
    root=ROOT/'lite-evidence';root.mkdir(exist_ok=False)
    path=OLD/'lite-compatibility/scenes/scene_box_source0'
    records=np.fromfile(path/'window_records.bin',dtype='<u2').reshape(-1,4)
    windows=[Window(tuple(map(int,r[:3])),int(r[3])) for r in records]
    t=time.perf_counter();cpu=time.process_time();g=GeometryGroup(load_geometry(path/'geometry.npz'))
    primary=g.bind(SOURCES[0]);independent=Control(g,SOURCES[0],'B_native');setup=time.perf_counter()-t
    proof,report=build_evidence(path,windows,primary,independent,primary_backend='new-cached-parametric-v1',independent_backend='unchanged-native-filter-v1',output=root/'physical-evidence.json')
    tick=time.perf_counter();proof.validate_files(path)
    evidence=[proof.verify_window(w,primary,SOURCES[0]) for w in windows];query_seconds=time.perf_counter()-tick
    assert len(evidence)==256 and all(e['covered_voxels']==w.size**3 for e,w in zip(evidence,windows))
    validation_wall=time.perf_counter()-t;validation_cpu=time.process_time()-cpu
    tick=time.perf_counter();shutil.copytree(OLD/'lite-compatibility',root/'dataset')
    target=root/'dataset/scenes/scene_box_source0';shutil.copy2(root/'physical-evidence.json',target/'physical-evidence.private.json')
    # Keep INCOMPLETE: this research pass does not add the missing Lite READY gate.
    meta=json.loads((target/'verification.json').read_text());meta.update(physical_evidence='physical-evidence.private.json',publication_ready=False)
    meta['hashes']['physical-evidence.private.json']=sha256_file(target/'physical-evidence.private.json');atomic_json(target/'verification.json',meta)
    # Update index hash to keep the copied diagnostic dataset internally consistent.
    import pyarrow.parquet as pq
    from rind3d.pipeline import parquet_write
    rows=pq.read_table(root/'dataset/scenes.parquet').to_pylist()
    rows[0]['verification_sha256']=sha256_file(target/'verification.json');parquet_write(root/'dataset/scenes.parquet',rows)
    with RINDDataset(root/'dataset',split=None,require_ready=False) as ds:
        assert len(ds)==256
        sample=ds[0];assert not set(sample)&{'candidate_sources','true_source','witnesses','source_id','physical_evidence'}
        assert ds.private_oracle().get_candidate_sources(sample['scene_id'],sample['window_id']).shape==(1,3)
    with RINDDataset(root/'dataset',split=None) as ds:assert len(ds)==0
    atomic_json(root/'result.json',{'windows':256,'side_counts':{'64':85,'128':85,'256':86},'actual_candidates_each':1,'max_candidates':8,
                'minimum_source_separation':32/1024,'setup_seconds':setup,'evidence_stages':report['stages'],
                'window_evidence_queries_seconds':query_seconds,'validation_wall_seconds':validation_wall,
                'validation_cpu_seconds':validation_cpu,'metadata_copy_loader_seconds':time.perf_counter()-tick,
                'complete_compatibility_check':True,'strict_loader_excludes_demo':True,'private_labels_hidden':True,'production_ready':False})


if __name__=='__main__':demo()
