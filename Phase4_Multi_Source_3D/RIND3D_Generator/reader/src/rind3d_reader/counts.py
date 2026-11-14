"""Pure Python exact 16³ coarse-prefix counts; no generator/native oracle."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json
import numpy as np
from .pack import sha256_file

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
  if not self.informative:raise ValueError('Uninformative response')
  return 10*self.lit//self.free

class CountIndex:
 def __init__(self,y,m,leaves):
  if y.shape!=m.shape or y.identity!=m.identity or y.policy!=m.policy:raise ValueError('Field identity mismatch')
  self.response=y;self.solid=m;self.leaves=np.asarray(leaves,dtype=np.uint16)
  if self.leaves.shape!=(*( (n+15)//16 for n in y.shape),2):raise ValueError('Count leaf shape mismatch')
  if np.any(self.leaves.astype(np.uint32).sum(-1)>4096):raise ValueError('Invalid leaf count')
  p=self.leaves.astype(np.uint32)
  for axis in range(3):p=p.cumsum(axis=axis,dtype=np.uint32)
  self.prefix=np.pad(p,((1,0),(1,0),(1,0),(0,0)))
 @property
 def nbytes(self):return self.leaves.nbytes+self.prefix.nbytes
 @classmethod
 def load(cls,path,y,m):
  path=Path(path);expected=path.with_name(path.name+'.sha256')
  if expected.exists() and sha256_file(path)!=expected.read_text().split()[0]:raise ValueError('Count index checksum mismatch')
  with np.load(path,allow_pickle=False) as f:
   leaves=f['leaves'];meta=json.loads(bytes(f['metadata']).decode())
  if meta.get('format')!='RIND-count-index-v1' or meta.get('identity')!=y.identity or meta.get('policy')!=y.policy:raise ValueError('Count metadata identity mismatch')
  if meta.get('leaves_sha256')!=hashlib.sha256(leaves.tobytes()).hexdigest() or meta.get('response_sha256')!=sha256_file(y.path) or meta.get('solid_sha256')!=sha256_file(m.path):raise ValueError('Count/field checksum mismatch')
  return cls(y,m,leaves)
 @classmethod
 def build(cls,y,m):
  if any(n%16 for n in y.shape):raise ValueError('16-aligned logical shape required')
  leaves=np.zeros((*tuple(n//16 for n in y.shape),2),dtype=np.uint16)
  for bz in range(y.grid[0]):
   for by in range(y.grid[1]):
    for bx in range(y.grid[2]):
     a=y.read_global_brick((bx,by,bz));b=m.read_global_brick((bx,by,bz))
     if np.any(a&b):raise ValueError('Lit solid bit')
     for iz in range(4):
      for iy in range(4):
       for ix in range(4):
        key=(bz*4+iz,by*4+iy,bx*4+ix)
        if any(v>=n for v,n in zip(key,leaves.shape)):continue
        sl=(slice(iz*16,iz*16+16),slice(iy*16,iy*16+16),slice(ix*16,ix*16+16));leaves[key]=(np.count_nonzero(a[sl]&~b[sl]),np.count_nonzero(b[sl]))
  return cls(y,m,leaves)
 def _box(self,lo,hi):
  total=np.zeros(2,dtype=np.int64)
  for bz in range(lo[2]//64,(hi[2]-1)//64+1):
   for by in range(lo[1]//64,(hi[1]-1)//64+1):
    for bx in range(lo[0]//64,(hi[0]-1)//64+1):
     a=self.response.read_global_brick((bx,by,bz));b=self.solid.read_global_brick((bx,by,bz));base=np.array([bx,by,bz])*64;l=np.maximum(lo-base,0);h=np.minimum(hi-base,64);sl=tuple(slice(int(l[k]),int(h[k])) for k in (2,1,0));total+=(np.count_nonzero(a[sl]&~b[sl]),np.count_nonzero(b[sl]))
  return total
 def query(self,origin,side):
  if any(not isinstance(v,(int,np.integer)) for v in (*origin,side)):raise TypeError('Integer ROI required')
  lo=np.asarray(origin,dtype=np.int64);hi=lo+side
  if len(lo)!=3 or side<1 or np.any(lo<0) or np.any(hi>self.response.shape[::-1]):raise ValueError('ROI out of bounds')
  a=((lo+15)//16)*16;b=(hi//16)*16
  if np.any(a>=b):r=self._box(lo,hi)
  else:
   r=np.zeros(2,dtype=np.int64);p=self.prefix
   for mask in range(8):
    v=[int((b[k] if mask&(1<<k) else a[k])//16) for k in range(3)];r+=(-1 if (3-mask.bit_count())%2 else 1)*p[v[2],v[1],v[0]].astype(np.int64)
   for x,y in ((lo,np.array([a[0],hi[1],hi[2]])),(np.array([b[0],lo[1],lo[2]]),hi),(np.array([a[0],lo[1],lo[2]]),np.array([b[0],a[1],hi[2]])),(np.array([a[0],b[1],lo[2]]),np.array([b[0],hi[1],hi[2]])),(np.array([a[0],a[1],lo[2]]),np.array([b[0],b[1],a[2]])),(np.array([a[0],a[1],b[2]]),np.array([b[0],b[1],hi[2]]))):
    if np.all(y>x):r+=self._box(x,y)
  return Counts(int(r[0]),int(side**3-r.sum()),int(r[1]))
