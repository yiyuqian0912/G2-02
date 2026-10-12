"""Draft student handoff aligned to the imported variable-count joint teacher."""
import numpy as np
from .search import canonical_sources

INTERFACE_VERSION = 'phase2-joint-draft-v1'


def validate_observation(sample):
    for key in ('scene_id','view_id','response','obstacle','window'):
        if key not in sample:raise ValueError(f'missing observation field: {key}')
    w=np.asarray(sample['window']);r=np.asarray(sample['response']);mask=np.asarray(sample['obstacle'])
    if w.shape!=(3,) or w.dtype.kind not in 'iu' or w[2]<=0 or (w[:2]<0).any():
        raise ValueError('window must be integer [x,y,size]')
    if r.shape!=(int(w[2]),int(w[2])) or not np.isfinite(r).all() or (r<0).any():
        raise ValueError('response must be finite nonnegative native total intensity')
    if mask.dtype!=bool or mask.shape!=r.shape or (r[mask]!=0).any():
        raise ValueError('obstacle mask must align with zero-response obstacle pixels')
    return sample


def validate_joint_candidates(sources,source_counts):
    """Validate canonical padded [N,Kmax,3] proposals; preserve source strengths."""
    values=np.asarray(sources,dtype=np.float64);counts=np.asarray(source_counts)
    if values.ndim!=3 or values.shape[0]<1 or values.shape[2]!=3 or counts.shape!=(len(values),) or counts.dtype.kind not in 'iu':
        raise ValueError('sources [N,Kmax,3] and integer source_counts [N] required')
    if not np.isfinite(values).all() or (counts<1).any() or (counts>values.shape[1]).any():
        raise ValueError('invalid active source count or nonfinite padded sources')
    seen=set()
    for row,k in zip(values,counts):
        if (row[int(k):]!=0).any():raise ValueError('inactive source rows must be zero padding')
        active=row[:int(k)]
        if (active[:,:2]<0).any() or (active[:,2]<0).any() or (active[:,2]>1).any():
            raise ValueError('positions must be nonnegative and strengths in [0,1]')
        canonical=canonical_sources(active)
        if not np.array_equal(active,canonical):raise ValueError('active source rows must use canonical source ordering')
        key=(int(k),canonical.tobytes())
        if key in seen:raise ValueError('duplicate complete source-set hypothesis; aggregate its base mass upstream')
        seen.add(key)
    return values,counts
