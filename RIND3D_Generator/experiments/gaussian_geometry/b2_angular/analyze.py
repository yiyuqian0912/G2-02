"""Compare actual residual receivers and their recorded costs on common fields."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from ..study import atomic_json


def near_changes(a):
    out=np.zeros_like(a,dtype=bool);n=a.shape[0]
    for dz in (-1,0,1):
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if not (dx or dy or dz):continue
                shifts=(dz,dy,dx)
                dst=tuple(slice(max(0,-d),min(n,n-d)) for d in shifts)
                src=tuple(slice(max(0,d),min(n,n+d)) for d in shifts)
                out[dst]|=a[dst]!=a[src]
    return out


def records(roots):
    for root in roots:
        for path in sorted(root.glob('gaussian*.json')):
            row=json.loads(path.read_text())
            if 'method' in row and 'complete' in row:yield root,path,row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',choices=['A_fixed6','A_fixed7'],default='A_fixed6');args=p.parse_args()
    rows=list(records(args.inputs))
    def support(r):return (r['case'],r['resolution'],r['backend'],r.get('world_resolution',r['resolution']),tuple(r.get('origin_xyz',[0,0,0])))
    lookup={(str(root),*support(r),r['method']):(root,path,r) for root,path,r in rows}
    baselines={}
    for root,path,r in rows:
        if r['method']==args.baseline:baselines.setdefault(support(r),(root,path,r))
    result=[];pairwise=[]
    for root,path,row in rows:
        key=(str(root),*support(row),args.baseline)
        selected=lookup.get(key,baselines.get(support(row)))
        if selected is None or not row['complete']:continue
        base_root,base_path,base=selected
        if not base['complete']:continue
        a=np.load(base_path.with_suffix('.npz'));b=np.load(path.with_suffix('.npz'))
        assert a['valid'].all() and b['valid'].all()
        assert all(np.array_equal(a[k],b[k]) for k in ('response','solid'))
        old_scalar=np.isin(a['evaluation_path'],[6,7]);new_scalar=np.isin(b['evaluation_path'],[6,7]);avoided=old_scalar&~new_scalar
        solid_near=near_changes(a['solid']);shadow_near=near_changes(a['response'])&~solid_near&~a['solid']
        masks={'near_sampled_SOLID':solid_near,'near_sampled_visibility_transition':shadow_near,
               'other_FREE':~solid_near&~shadow_near&~a['solid']}
        costs={}
        for name,mask in masks.items():
            costs[name]={'original_scalar_receivers':int((old_scalar&mask).sum()),'original_ray_seconds':float(a['ray_nanoseconds'][old_scalar&mask].sum())/1e9,
                         'avoided_original_receivers':int((avoided&mask).sum()),'avoided_original_recorded_ray_seconds':float(a['ray_nanoseconds'][avoided&mask].sum())/1e9,
                         'new_actual_ray_seconds':float(b['ray_nanoseconds'][mask].sum())/1e9}
        by_label={}
        for label,value in [('MISS',6),('HIT',7)]:
            mask=a['evaluation_path']==value
            by_label[label]={'original_count':int(mask.sum()),'original_ray_seconds':float(a['ray_nanoseconds'][mask].sum())/1e9,
                             'avoided_count':int((mask&avoided).sum()),'avoided_original_recorded_ray_seconds':float(a['ray_nanoseconds'][mask&avoided].sum())/1e9}
        q95=np.quantile(a['ray_nanoseconds'][old_scalar],.95) if old_scalar.any() else 0
        evidence=b['visibility_evidence'] if 'visibility_evidence' in b else np.zeros_like(b['evaluation_path'])
        result.append({'run':root.name,'case':row['case'],'resolution':row['resolution'],'backend':row['backend'],'method':row['method'],
                       'baseline':args.baseline,'baseline_record_SHA256':hashlib.sha256(base_path.read_bytes()).hexdigest(),
                       'record_SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'cold_total_seconds':row['cold_total_seconds'],
                       'baseline_cold_total_seconds':base['cold_total_seconds'],'cold_speedup':base['cold_total_seconds']/row['cold_total_seconds'],
                       'original_scalar_count':int(old_scalar.sum()),'new_scalar_count':int(new_scalar.sum()),
                       'avoided_original_scalar_count':int(avoided.sum()),'new_scalar_not_in_original_set':int((new_scalar&~old_scalar).sum()),
                       'original_ray_seconds':float(a['ray_nanoseconds'].sum())/1e9,'new_actual_ray_seconds':float(b['ray_nanoseconds'].sum())/1e9,
                       'avoided_original_recorded_ray_seconds':float(a['ray_nanoseconds'][avoided].sum())/1e9,
                       'original_P95_expensive_count':int((old_scalar&(a['ray_nanoseconds']>=q95)).sum()),
                       'original_P95_expensive_avoided':int((avoided&(a['ray_nanoseconds']>=q95)).sum()),
                       'new_angular_evidence_on_original_residual':int((old_scalar&(evidence==1)).sum()),
                       'new_empty_prefix_evidence_on_original_residual':int((old_scalar&(evidence==2)).sum()),
                       'existing_point_cover_evidence_on_original_residual':int((old_scalar&(evidence==4)).sum()),
                       'by_sampled_neighborhood_class':costs,'by_original_ray_label':by_label,
                       'physical_bits_identical':True})
        if row['method']=='D_union':
            peer=lookup.get((str(root),*support(row),'C_angular'))
            if peer and peer[2]['complete']:
                c=np.load(peer[1].with_suffix('.npz'));c_scalar=np.isin(c['evaluation_path'],[6,7])
                pairwise.append({'run':root.name,'case':row['case'],'resolution':row['resolution'],'backend':row['backend'],
                                 'C_and_D_exact_same_scalar_receiver_set':bool(np.array_equal(c_scalar,new_scalar)),
                                 'C_scalar_D_not_scalar':int((c_scalar&~new_scalar).sum()),'D_scalar_C_not_scalar':int((new_scalar&~c_scalar).sum())})
    atomic_json(args.output,{'comparisons':result,'single_vs_union_pairwise':pairwise,'caution':'Neighborhood classes use 26 sampled neighbors, not exact continuous distance. Avoided ORIGINAL recorded time is a matched receiver-set diagnostic, not a guaranteed counterfactual runtime saving. Timings remain separate complete cold measurements. Baselines prefer the same run/ROI and are pinned by record hash.'})
    print(json.dumps({'comparisons':len(result),'output':str(args.output)}))


if __name__=='__main__':main()
