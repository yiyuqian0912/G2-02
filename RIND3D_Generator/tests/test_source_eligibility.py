import numpy as np
from types import SimpleNamespace
from rind3d.source_eligibility import certificate,centre_octant_bounds

def test_exact_centre_bounds_include_every_legal_origin():
 for side in (4,8,16):
  for x in range(32):
   a=x-side//2+1;b=x+side//2
   centres=[o+side//2 for o in range(33-side) if o<=x<o+side]
   assert all(a<=c<=b for c in centres)
 assert centre_octant_bounds([15,15,15],[16,16,16],2,32) is None
 assert centre_octant_bounds([17,17,17],[17,17,17],2,32)['sole_octant']==[1,1,1]

def test_solid_is_never_counted_as_shadow_free():
 leaves=np.zeros((8,8,8,2),np.int64);leaves[...,0]=16**3
 leaves[0,0,0]=[0,16**3]
 idx=SimpleNamespace(leaves=leaves,shape=(128,)*3)
 p=certificate(idx,32)
 assert p['class']=='shadow_free' and p['class_count']==0
 leaves[0,0,0]=[0,0]
 p=certificate(idx,32)
 assert p['reason']=='all_informative_centres_in_one_octant' and p['sole_octant']==[0,0,0]
 leaves[-1,-1,-1]=[0,0]
 assert certificate(idx,32) is None

def test_packed_recheck_independent_of_index_and_rejects_wrong_bbox(tmp_path):
 import pytest
 from tests.test_pack import write
 from rind3d.pack import RINDPack
 from rind3d.counts import CountIndex
 from rind3d.source_eligibility import recheck
 y=np.ones((128,)*3,dtype=bool);m=np.zeros_like(y);y[:8,:8,:8]=False;m[:4,:4,:4]=True
 write(tmp_path/'y',y);write(tmp_path/'m',m)
 with RINDPack(tmp_path/'y') as a,RINDPack(tmp_path/'m') as b:
  idx=CountIndex(a,b);proof=certificate(idx,32)
  assert proof['class']=='shadow_free' and recheck(a,b,proof)
  bad={**proof,'class_count':proof['class_count']+1}
  with pytest.raises(ValueError,match='count'):recheck(a,b,bad)
  bad={**proof,'voxel_bbox_lower':[16,0,0]}
  # Recompute all formal centre bounds, so actual packed bits expose the lie.
  bad.update(centre_octant_bounds(bad['voxel_bbox_lower'],bad['voxel_bbox_upper'],32,128))
  with pytest.raises(ValueError,match='support'):recheck(a,b,bad)
