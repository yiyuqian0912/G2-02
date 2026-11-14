"""Measured space/latency, native BVH comparisons, and bounded streaming audit."""
from pathlib import Path
import hashlib,json,time,resource
import numpy as np
from rind.contracts import Geometry,RootKind,make_leaf
from rind.storage import CanonicalBundle
from .geometry import Oracle,Budget,centres,POLICY_HASH,render_region
from .stream import stream_response
from .pack import RINDPack,RINDPackWriter,atomic_json
from .zarr_store import create_zarr,seal_zarr,ZarrBits
from .pilot import source_fingerprint
from .window_contract import WINDOW_SIDES

def bytes_on_disk(path):return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())

def fixture_box():
    b=CanonicalBundle(3);leaf=make_leaf('affine_box_v1',3,center=[.5,.5,.5],matrix=np.diag([.125,.25,.25]));b.add_leaf(leaf);b.add_geometry(Geometry(3,RootKind.REF,leaf.content_hash));return b

def integer_box_reference(source,points,resolution):
    """Independent exact test fixture: dyadic values become bounded integers;
    cross products compare slab fractions without floating-point division.
    """
    scale=2**20;s=np.rint(np.asarray(source)*scale).astype(np.int64)
    q=np.rint(points*scale).astype(np.int64)
    if not np.array_equal(q/scale,points) or not np.array_equal(s/scale,source):raise ValueError('Fixture is not exactly representable on this dyadic lattice')
    lower=np.rint(np.array([.375,.25,.25])*scale).astype(np.int64);upper=np.rint(np.array([.625,.75,.75])*scale).astype(np.int64)
    en=np.zeros(len(q),dtype=np.int64);ed=np.ones(len(q),dtype=np.int64);xn=np.ones(len(q),dtype=np.int64);xd=np.ones(len(q),dtype=np.int64);miss=np.zeros(len(q),bool)
    for a in range(3):
        d=q[:,a]-s[a];den=np.abs(d);parallel=den==0;safe=np.where(parallel,1,den)
        low=np.where(d>=0,lower[a]-s[a],s[a]-upper[a]);high=np.where(d>=0,upper[a]-s[a],s[a]-lower[a])
        update=(~parallel)&(low*ed>en*safe);en=np.where(update,low,en);ed=np.where(update,safe,ed)
        update=(~parallel)&(high*xd<xn*safe);xn=np.where(update,high,xn);xd=np.where(update,safe,xd)
        miss|=parallel&((s[a]<lower[a])|(s[a]>upper[a]))
    return miss|(en*xd>xn*ed)

def compare_storage(output,resolution=256,reads=24):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);results=[]
    for name in ['uniform','boundary-slabs','checkerboard','random']:
        p=output/(name+'.rindb');zp=output/(name+'.zarr')
        if p.exists() or zp.exists():raise FileExistsError('Benchmark output exists; use a new directory')
        identity=hashlib.sha256(name.encode()).hexdigest();writer=RINDPackWriter(p,(resolution,)*3,identity=identity,policy=POLICY_HASH)
        zarr=create_zarr(zp,(resolution,)*3);rng=np.random.default_rng(12);times={'rindpack_write_seconds':0.,'zarr_write_seconds':0.}
        if resolution!=256:raise ValueError('Storage comparison buffers exactly one 256^3 packed outer shard')
        zbuffer=np.zeros((256,256,32),dtype=np.uint8)
        g=resolution//64
        for z in range(g):
            for y in range(g):
                for x in range(g):
                    zz,yy,xx=np.indices((64,)*3);xx+=x*64;yy+=y*64;zz+=z*64
                    if name=='uniform':a=np.zeros((64,)*3,dtype=bool)
                    elif name=='boundary-slabs':a=(xx+yy//3+zz//7)>(resolution//2)
                    elif name=='checkerboard':a=(xx+yy+zz)%2==0
                    else:a=rng.integers(0,2,(64,)*3,dtype=np.uint8).astype(bool)
                    t=time.perf_counter();writer.write((z*g+y)*g+x,np.packbits(a,bitorder='little').tobytes());times['rindpack_write_seconds']+=time.perf_counter()-t
                    t=time.perf_counter();zbuffer[x*64:x*64+64,y*64:y*64+64,z*8:z*8+8]=np.packbits(a.transpose(2,1,0),axis=-1,bitorder='little');times['zarr_write_seconds']+=time.perf_counter()-t
        t=time.perf_counter();zarr[:]=zbuffer;seal_zarr(zp,zarr);times['zarr_write_seconds']+=time.perf_counter()-t
        t=time.perf_counter();writer.finish();times['rindpack_write_seconds']+=time.perf_counter()-t
        sizes=[];origins=[];rng=np.random.default_rng(21)
        for _ in range(reads):
            size=int(rng.choice([s for s in WINDOW_SIDES if s<=resolution]));sizes.append(size);origins.append(tuple(rng.integers(0,resolution-size+1,3).tolist()))
        readers=[RINDPack(p,cache_bricks=0),ZarrBits(zp)];latency=[[],[]]
        for origin,size in zip(origins,sizes):
            values=[]
            for j,r in enumerate(readers):
                t=time.perf_counter();v=r.read_window(*origin,size,packed=True);latency[j].append(time.perf_counter()-t);values.append(v)
            if not np.array_equal(values[0].unpack(),values[1].unpack()):raise AssertionError('RINDPack/Zarr bit mismatch')
        for r in readers:r.close()
        results.append({'pattern':name,'resolution':resolution,'rindpack_bytes':p.stat().st_size,'zarr_bytes':bytes_on_disk(zp),
                        'zarr_files':sum(x.is_file() for x in zp.rglob('*')),'reads':reads,
                        'rindpack_median_read_ms':float(np.median(latency[0])*1000),'zarr_median_read_ms':float(np.median(latency[1])*1000),
                        'rindpack_p95_read_ms':float(np.quantile(latency[0],.95)*1000),'zarr_p95_read_ms':float(np.quantile(latency[1],.95)*1000),
                        'bit_mismatches':0,'zarr_write_strategy':'one bounded 2 MiB packed outer-shard buffer; independent inner Zstd+CRC32C',**times})
        print('storage benchmark:',name,results[-1]['rindpack_bytes'],results[-1]['zarr_bytes'],flush=True)
    return results

def compare_backends(output,cfg):
    from ._native import rindray_reference
    result=[];source=np.array([65317,193993,337117],dtype=np.float64)/(2**20);oracle=Oracle(fixture_box())
    for resolution in [32,64,128,256,512]:
        exact_mismatch=0;bvh_mismatch=0;filter_seconds=0.;reference_seconds=0.;bvh_seconds=0.;tested=0
        # Full small grids, selected bounded 64^3 bricks on 256/512.
        origins=[(0,0,0)] if resolution<=64 else [(x,y,z) for z in range(0,resolution,64) for y in range(0,resolution,64) for x in range(0,resolution,64)]
        if resolution>=256:
            rng=np.random.default_rng(9);origins=[origins[i] for i in sorted(rng.choice(len(origins),min(8,len(origins)),replace=False))]
        for origin in origins:
            size=min(64,resolution);points=centres(origin,size,resolution)
            t=time.perf_counter();expected=integer_box_reference(source,points,resolution);reference_seconds+=time.perf_counter()-t
            t=time.perf_counter();actual=oracle.response(source,points,budget=Budget(300,5_000_000));filter_seconds+=time.perf_counter()-t
            t=time.perf_counter();bvh=rindray_reference(oracle.primitives,source[None],points,1);bvh_seconds+=time.perf_counter()-t
            exact_mismatch+=int(np.count_nonzero(actual!=expected));bvh_mismatch+=int(np.count_nonzero(actual!=bvh));tested+=len(points)
        result.append({'resolution':resolution,'scope':'full-grid' if resolution<=128 else '8-selected-bricks',
                       'points':tested,'canonical_vs_integer_mismatches':exact_mismatch,'raw_rindray_mismatches':bvh_mismatch,
                       'filtered_exact_seconds':filter_seconds,'independent_integer_seconds':reference_seconds,'rindray_bvh_seconds':bvh_seconds,
                       'rindray_policy':'float32/open endpoints; measured baseline, not canonical truth'})
        if exact_mismatch:raise AssertionError('Canonical filter disagrees with exact integer fixture')
        print('backend benchmark:',resolution,result[-1],flush=True)
        atomic_json(Path(output)/'backend-progress.json',result)
    return result

def benchmark(cfg,output,codecs=True,backends=True,full_1024=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);result={'format':'rind3d-benchmark-v1','config':cfg,'source_sha256':source_fingerprint()}
    if codecs:
        result['storage']=compare_storage(output/'formats');atomic_json(output/'report.json',result)
    if backends:result['backends']=compare_backends(output,cfg)
    if full_1024:
        b=CanonicalBundle(3);b.add_geometry(Geometry.empty(3));oracle=Oracle(b);local=dict(cfg,resolution=1024,generation_seconds=300)
        result['full1024_empty_stream']=stream_response(oracle,np.array([.1,.2,.3]),output/'empty1024',local)
        result['full1024_empty_stream']['scope']='Complete exact empty-world response/solid; non-informative, never READY'
    result['max_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    atomic_json(output/'report.json',result);return result
