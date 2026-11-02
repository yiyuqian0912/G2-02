"""Sufficient exact certificates for existing observation-contract infeasibility.

This is about sampled observations, not a voxel approximation of geometry.
Only a complete certified response/mask may supply these counts. Unproved
sampler failures remain incomplete and never enter this rejection path.
"""
import numpy as np

class SourceIneligible(RuntimeError):pass

def centre_octant_bounds(lo,hi,max_side,world_side):
 lower=np.asarray(lo,dtype=np.int64)-max_side//2+1;upper=np.asarray(hi,dtype=np.int64)+max_side//2;bits=[]
 for a,b in zip(lower,upper):
  if a>=world_side//2:bits.append(1)
  elif b<world_side//2:bits.append(0)
  else:return None
 return {'centre_lower':lower.tolist(),'centre_upper':upper.tolist(),'sole_octant':bits}

def certificate(index,max_side=256):
 leaves=index.leaves.astype(np.int64);volume=16**3
 for name,counts in (('lit_free',leaves[...,0]),('shadow_free',volume-leaves.sum(-1))):
  support=np.argwhere(counts>0);total=int(counts.sum())
  if not len(support):return {'version':'stored-coverage-ineligibility-v1','reason':'required_free_class_absent','class':name,'class_count':0,'max_side':max_side,'world_side':index.shape[0]}
  lo=support.min(0)[::-1]*16;hi=(support.max(0)[::-1]+1)*16-1;bounds=centre_octant_bounds(lo,hi,max_side,index.shape[0])
  if bounds is not None:return {'version':'stored-coverage-ineligibility-v1','reason':'all_informative_centres_in_one_octant','class':name,'class_count':total,'voxel_bbox_lower':lo.tolist(),'voxel_bbox_upper':hi.tolist(),'max_side':max_side,'world_side':index.shape[0],**bounds}
 return None

def recheck(y,m,proof):
 """Independent Python packed-brick check of every class voxel/bbox.

 Uniform absent-class bricks follow directly from their encoded constant;
 every potentially present nonuniform brick is decoded and CRC checked.
 """
 total=0;which=proof['class'];absent=proof['reason']=='required_free_class_absent'
 if y.shape!=m.shape or y.shape!=(proof['world_side'],)*3:raise ValueError('Eligibility field shape mismatch')
 if not absent:
  lo=np.asarray(proof['voxel_bbox_lower']);hi=np.asarray(proof['voxel_bbox_upper']);expected=centre_octant_bounds(lo,hi,proof['max_side'],proof['world_side'])
  if expected is None or any(proof[k]!=v for k,v in expected.items()):raise ValueError('Invalid centre-octant proof')
 gz,gy,gx=y.grid
 for z in range(gz):
  for v in range(gy):
   for x in range(gx):
    i=(z*gy+v)*gx+x;yc=y.index[i][2];mc=m.index[i][2]
    if mc==1 or (which=='lit_free' and yc==0) or (which=='shadow_free' and yc==1):continue
    if mc==0 and yc==(1 if which=='lit_free' else 0):
     count=64**3
     if absent or np.any(np.array([x,v,z])*64<lo) or np.any(np.array([x,v,z])*64+63>hi):raise ValueError('Uniform class brick outside claimed support')
    else:
     a=y.read_global_brick((x,v,z));b=m.read_global_brick((x,v,z));bits=(a if which=='lit_free' else ~a)&~b;count=int(bits.sum())
     if count:
      if absent:raise ValueError('Claimed absent class is present')
      lower=np.maximum(lo-np.array([x,v,z])*64,0);upper=np.minimum(hi-np.array([x,v,z])*64+1,64);inside=int(bits[lower[2]:upper[2],lower[1]:upper[1],lower[0]:upper[0]].sum()) if np.all(upper>lower) else 0
      if inside!=count:raise ValueError('Class voxel outside claimed support')
    total+=count
 if total!=proof['class_count']:raise ValueError('Eligibility class count differs')
 return True
