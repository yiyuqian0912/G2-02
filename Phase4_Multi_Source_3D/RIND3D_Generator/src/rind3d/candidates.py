"""Lossless source-pool references: packed offsets v2, legacy fixed rows v1."""
from pathlib import Path
import numpy as np

STORAGE='pool-offsets-v2'

def storage_version(root,meta):
    declared=meta.get('candidate_storage')
    if declared is not None:
        if declared not in (STORAGE,'fixed-128-v1'):raise ValueError('Unknown candidate storage')
        return 2 if declared==STORAGE else 1
    return 2 if (Path(root)/'candidate_offsets.bin').exists() else 1

def read_references(root,meta,window_count,counts):
    """One view per window. No padding in v2; v1 sentinels are never exposed."""
    root=Path(root);dtype=np.dtype(meta['candidate_id_dtype'])
    if dtype.str not in ('<u2','<u4'):raise ValueError('Unsupported candidate index endian/dtype')
    raw=(root/'candidate_ids.bin').read_bytes()
    if len(raw)%dtype.itemsize:raise ValueError('Truncated candidate IDs')
    ids=np.frombuffer(raw,dtype=dtype)
    if storage_version(root,meta)==2:
        data=(root/'candidate_offsets.bin').read_bytes()
        if len(data)!=(window_count+1)*8:raise ValueError('Candidate offsets length mismatch')
        offsets=np.frombuffer(data,dtype='<u8')
        if offsets[0]!=0 or offsets[-1]!=len(ids) or np.any(offsets[1:]<offsets[:-1]):raise ValueError('Invalid candidate offsets')
        if not np.array_equal(np.diff(offsets),counts):raise ValueError('Candidate offsets/counts disagree')
        return [ids[int(a):int(b)] for a,b in zip(offsets[:-1],offsets[1:])]
    if len(ids)!=window_count*128:raise ValueError('Legacy candidate/window table shape mismatch')
    table=ids.reshape(window_count,128)
    if np.any(counts>128):raise ValueError('Legacy count exceeds row width')
    for row,count in zip(table,counts):
        if np.any(row[int(count):]!=np.iinfo(dtype).max):raise ValueError('Invalid legacy sentinel')
    return [row[:int(n)] for row,n in zip(table,counts)]
