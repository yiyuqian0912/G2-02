"""Audit full release metadata and positive-source/subset-sum assumptions."""
from analyze import ROOT
from pathlib import Path
import numpy as np,json,argparse
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT.parent/'Phase2_Multi_Source/data/raw/phase2-multi-source');args=p.parse_args();root=args.data
c=np.load(root/'driver_counts.npy',mmap_mode='r');a=np.load(root/'driver_strengths.npy',mmap_mode='r');src=np.load(root/'drivers.npy',mmap_mode='r');v=np.load(root/'local_views.npy',mmap_mode='r');n=np.load(root/'view_counts.npy',mmap_mode='r')
min_gap=float('inf');dups=0
for i,k in enumerate(c):
    values=np.array([sum(a[i,j] for j in range(k) if mask>>j&1) for mask in range(1<<int(k))]);min_gap=min(min_gap,float(np.diff(np.sort(values)).min()))
    dups+=int(len(np.unique(src[i,:k],axis=0))<k)
active=np.arange(a.shape[1])[None,:]<c[:,None];vm=np.arange(v.shape[1])[None,:]<n[:,None]
out=dict(data_path=str(root.resolve()),num_scenes=len(c),num_configurations=len(c),total_views=int(n.sum()),counts_by_K={str(k):int((c==k).sum()) for k in np.unique(c)},counts_by_size={str(L):int((v[:,:,2][vm]==L).sum()) for L in np.unique(v[:,:,2][vm])},strength_min=float(a[active].min()),strength_max=float(a[active].max()),min_distinct_subset_sum_gap=min_gap,duplicate_source_position_scenes=dups,positions_all_integer=bool(np.all(src[active]==np.floor(src[active]))),manifest=json.loads((root/'manifest.json').read_text()))
assert out['strength_min']>0
(Path(__file__).resolve().parent/'data_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='manifest'},indent=2))
