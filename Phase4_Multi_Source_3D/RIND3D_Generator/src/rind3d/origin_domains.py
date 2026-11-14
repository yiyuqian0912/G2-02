"""Exact integer origin domains; these operations never infer response labels."""
from dataclasses import dataclass
from itertools import combinations
import math,numbers
import numpy as np


@dataclass(frozen=True)
class OriginBox:
    lower:tuple[int,int,int]
    upper:tuple[int,int,int]
    @property
    def count(self):return math.prod(max(0,b-a+1) for a,b in zip(self.lower,self.upper))


def subtract_box(box,forbidden):
    lo=np.maximum(box.lower,forbidden.lower);hi=np.minimum(box.upper,forbidden.upper)
    if np.any(lo>hi):return [box]
    lower=list(box.lower);upper=list(box.upper);result=[]
    for axis in range(3):
        if lower[axis]<lo[axis]:
            end=upper.copy();end[axis]=int(lo[axis])-1;result.append(OriginBox(tuple(lower),tuple(end)));lower[axis]=int(lo[axis])
        if upper[axis]>hi[axis]:
            start=lower.copy();start[axis]=int(hi[axis])+1;result.append(OriginBox(tuple(start),tuple(upper)));upper[axis]=int(hi[axis])
    return result


def voxel_pair_origins(positive,negative,source,side,resolution):
    """Generic integer-ROI math; the sampler separately enforces observation sides."""
    if isinstance(side,bool) or not isinstance(side,numbers.Integral) or not 1<=side<=resolution:raise ValueError('Invalid integer extent')
    coords=[np.asarray(p) for p in (positive,negative)]
    if any(p.shape!=(3,) or not np.issubdtype(p.dtype,np.integer) or np.any(p<0) or np.any(p>=resolution) for p in coords):raise ValueError('Invalid voxel witnesses')
    low=np.maximum(0,np.maximum(*coords)-side+1);high=np.minimum(resolution-side,np.minimum(*coords))
    box=OriginBox(tuple(map(int,low)),tuple(map(int,high)))
    if not box.count:return []
    s=np.asarray(source,dtype=float)*resolution
    if s.shape!=(3,) or not np.isfinite(s).all():raise ValueError('Invalid source')
    forbidden=OriginBox(tuple(int(np.floor(v-side))+1 for v in s),tuple(int(np.floor(v)) for v in s))
    return subtract_box(box,forbidden)


def sample_origins(boxes,count,rng):
    """Uniform subset of integer volume, including correct singleton-box weight."""
    boxes=sorted(boxes,key=lambda b:(b.lower,b.upper))
    for box in boxes:
        if len(box.lower)!=3 or len(box.upper)!=3 or not box.count:raise ValueError('Nonempty integer 3D origin boxes required')
        if any(isinstance(v,bool) or not isinstance(v,numbers.Integral) for v in (*box.lower,*box.upper)):raise ValueError('Integer origin boxes required')
    for a,b in combinations(boxes,2):
        if all(max(x,y)<=min(u,v) for x,y,u,v in zip(a.lower,b.lower,a.upper,b.upper)):raise ValueError('Overlapping origin boxes')
    if isinstance(count,bool) or not isinstance(count,numbers.Integral) or count<0:raise ValueError('Invalid origin count')
    volumes=[b.count for b in boxes];total=sum(volumes);size=min(int(count),total)
    if not size:return []
    selected=set();ranks=[]
    for end in range(total-size,total):
        value=int(rng.integers(0,end+1));value=end if value in selected else value;selected.add(value);ranks.append(value)
    rng.shuffle(ranks);ends=np.cumsum(volumes,dtype=np.int64);result=[]
    for rank in ranks:
        index=int(np.searchsorted(ends,rank,side='right'));box=boxes[index];offset=rank-(int(ends[index-1]) if index else 0)
        dy=box.upper[1]-box.lower[1]+1;dz=box.upper[2]-box.lower[2]+1
        result.append((box.lower[0]+offset//(dy*dz),box.lower[1]+offset//dz%dy,box.lower[2]+offset%dz))
    return result
