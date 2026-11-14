"""Matched selection geometry and overlap, distinct from hard validity."""
import json
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
from rind3d.pack import atomic_json
from .audit import BASE,OUT

def run():
    sources={s['scene_id']:np.array(s['source']) for s in json.loads((BASE/'report.json').read_text())['scenes']}
    files={'natural':OUT/'frozen-window-counts.parquet','stratified':OUT/'sampling/stratified-windows.parquet','mixed':OUT/'sampling-mixed-current/mixed-windows.parquet'};result={}
    grid=np.array(np.meshgrid(*([np.arange(32,1024,64)+.5]*3),indexing='ij')).reshape(3,-1).T
    for mode,path in files.items():
        rows=pq.read_table(path).to_pylist();scenes=[]
        for sid in sorted(sources):
            selected=[r for r in rows if r['scene_id']==sid];o=np.array([r['origin'] for r in selected]);l=np.array([r['side'] for r in selected]);e=o+l[:,None];centre=o+l[:,None]/2
            intersection=np.maximum(0,np.minimum(e[:,None],e[None])-np.maximum(o[:,None],o[None])).prod(axis=-1)
            iou=intersection/(l[:,None]**3+l[None]**3-intersection);np.fill_diagonal(iou,0)
            covered=np.any(np.all((grid[None]>=o[:,None])&(grid[None]<e[:,None]),axis=-1),axis=0)
            distance=np.linalg.norm(centre/1024-sources[sid],axis=1)
            scenes.append({'scene_id':sid,'family':selected[0]['family'],'occupied_octants':len({tuple(c) for c in (centre>=512).astype(int)}),
                           'occupied_centre_cells_128':len({tuple(c) for c in (centre//128).astype(int)}),
                           'mean_pair_iou':float(iou.sum()/(256*255)),'mean_max_other_iou':float(iou.max(axis=1).mean()),'max_iou':float(iou.max()),
                           'fixed_4096_control_point_union_coverage':float(covered.mean()),'source_distance_mean':float(distance.mean()),
                           'source_distance_p95':float(np.percentile(distance,95)),'bins':dict(Counter(r['ratio_bin'] for r in selected))})
        summary={k:float(np.mean([s[k] for s in scenes])) for k in scenes[0] if k not in ('scene_id','family','bins')}
        result[mode]={'scenes':scenes,'scene_mean_metrics':summary,'coverage_is_fixed_grid_estimate_not_exact_union_volume':True}
    atomic_json(OUT/'matched-selection-quality.json',result)

if __name__=='__main__':run()
