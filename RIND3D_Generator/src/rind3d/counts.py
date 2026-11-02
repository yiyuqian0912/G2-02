"""Exact counts for unaligned stored windows using a 16³ block prefix.

Only a 65³ (for 1024³ fields) coarse prefix is allocated. Boundary slabs are
disjoint and scanned from actual packed bits. Solid zeros never count as shadow.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,numbers,os,time
import numpy as np
from .pack import RINDPack,sha256_file,atomic_text
from . import _research

VERSION='RIND-count-index-v1'

@dataclass(frozen=True)
class Counts:
    lit:int
    shadow:int
    solid:int
    @property
    def free(self):return self.lit+self.shadow
    @property
    def informative(self):return self.lit>0 and self.shadow>0
    @property
    def lit_ratio(self):return self.lit/self.free if self.free else None
    @property
    def ratio_bin(self):
        if not self.informative:raise ValueError('Exact ratios zero/one or no free voxels are invalid observations')
        return min(9,10*self.lit//self.free)

def _index(reader):return [[r[0],r[1],r[2],r[5],r[6],r[7]] for r in reader.index]

class CountIndex:
    def __init__(self,response,solid,leaves=None,*,cache_bricks=128,metadata=None):
        if response.shape!=solid.shape or response.identity!=solid.identity or response.policy!=solid.policy:raise ValueError('Count fields identity mismatch')
        self.response=response;self.solid=solid;self.shape=response.shape
        self.native=_research.PackedFields(str(response.path),str(solid.path),list(self.shape[::-1]),_index(response),_index(solid),cache_bricks)
        if leaves is None:leaves=self.native.leaves()
        self.leaves=np.ascontiguousarray(leaves,dtype=np.uint16)
        expected=(*tuple((n+15)//16 for n in self.shape),2)
        if self.leaves.shape!=expected:raise ValueError('Invalid count leaf shape')
        if np.any(self.leaves.astype(np.uint32).sum(axis=-1)>4096):raise ValueError('Invalid count leaf populations')
        prefix=self.leaves.astype(np.uint32)
        for axis in (0,1,2):prefix=prefix.cumsum(axis=axis,dtype=np.uint32)
        self.prefix=np.pad(prefix,((1,0),(1,0),(1,0),(0,0)))
        self.metadata=metadata or {'format':VERSION,'block_side':16,'shape_zyx':list(self.shape),'identity':response.identity,'policy':response.policy,
                                  'response_sha256':sha256_file(response.path),'solid_sha256':sha256_file(solid.path)}
        self.build_report={}
    @property
    def nbytes(self):return self.leaves.nbytes+self.prefix.nbytes
    @classmethod
    def build(cls,response,solid,**kwargs):
        start=time.perf_counter();cpu=time.process_time();result=cls(response,solid,**kwargs)
        result.build_report={'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'index_ram_bytes':result.nbytes,
                             'coarse_prefix_shape':list(result.prefix.shape),'native':result.native.stats(),'dense_world_array':False}
        return result
    def _validate(self,origin,size):
        if len(origin)!=3 or any(isinstance(x,bool) or not isinstance(x,numbers.Integral) for x in (*origin,size)):raise TypeError('Integer ROI required')
        o=tuple(map(int,origin));size=int(size)
        if size<1 or min(o)<0 or any(v+size>n for v,n in zip(o,self.shape[::-1])):raise ValueError('Count ROI outside field')
        return o,size
    def direct(self,origin,size):
        o,size=self._validate(origin,size);hi=tuple(v+size for v in o)
        lit,solid=map(int,self.native.boxes(np.array([[*o,*hi]],dtype=np.int32))[0]);return Counts(lit,size**3-lit-solid,solid)
    def query(self,origin,size):
        o,size=self._validate(origin,size);e=tuple(v+size for v in o);a=tuple((v+15)//16 for v in o);b=tuple(v//16 for v in e)
        if any(x>=y for x,y in zip(a,b)):return self.direct(o,size)
        total=[0,0]
        for mask in range(8):
            x,y,z=[b[k] if mask&(1<<k) else a[k] for k in range(3)];sign=1 if mask.bit_count()%2 else -1
            for c in (0,1):total[c]+=sign*int(self.prefix[z,y,x,c])
        aa=tuple(v*16 for v in a);bb=tuple(v*16 for v in b);rectangles=[]
        # Full YZ X-slabs; middle X Y-slabs; middle XY Z-slabs.
        for axis in range(3):
            for lower,upper in ((o[axis],aa[axis]),(bb[axis],e[axis])):
                lo=list(o);hi=list(e)
                for k in range(axis):lo[k]=aa[k];hi[k]=bb[k]
                lo[axis]=lower;hi[axis]=upper
                if all(x<y for x,y in zip(lo,hi)):rectangles.append([*lo,*hi])
        if rectangles:
            values=self.native.boxes(np.asarray(rectangles,dtype=np.int32)).sum(axis=0,dtype=np.uint64)
            total=[x+int(y) for x,y in zip(total,values)]
        lit,solid=total
        if min(lit,solid)<0 or lit+solid>size**3:raise RuntimeError('Invalid exact count aggregation')
        return Counts(lit,size**3-lit-solid,solid)
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
        metadata=dict(self.metadata,leaves_sha256=hashlib.sha256(self.leaves.tobytes()).hexdigest())
        with tmp.open('wb') as f:
            np.savez_compressed(f,leaves=self.leaves,metadata=np.frombuffer(json.dumps(metadata,sort_keys=True).encode(),dtype=np.uint8));f.flush();os.fsync(f.fileno())
        os.replace(tmp,path);atomic_text(path.with_name(path.name+'.sha256'),sha256_file(path)+'  '+path.name+'\n');return path.stat().st_size
    @classmethod
    def load(cls,path,response,solid,**kwargs):
        path=Path(path)
        if sha256_file(path)!=path.with_name(path.name+'.sha256').read_text().split()[0]:raise ValueError('Count index file SHA mismatch')
        with np.load(path,allow_pickle=False) as z:
            leaves=z['leaves'];meta=json.loads(z['metadata'].tobytes())
        if meta.get('format')!=VERSION or meta.get('block_side')!=16 or meta.get('shape_zyx')!=list(response.shape) or meta.get('identity')!=response.identity or meta.get('policy')!=response.policy:raise ValueError('Count index contract/identity mismatch')
        if meta.get('response_sha256')!=sha256_file(response.path) or meta.get('solid_sha256')!=sha256_file(solid.path) or meta.get('leaves_sha256')!=hashlib.sha256(leaves.tobytes()).hexdigest():raise ValueError('Stale/corrupt count index')
        return cls(response,solid,leaves,metadata=meta,**kwargs)
