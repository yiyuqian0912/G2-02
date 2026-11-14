"""Audit all current pilot windows, geometry parameters, and nonlocal samples."""
import argparse,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
from rind3d.pack import RINDPack,atomic_json
from rind3d.counts import CountIndex
from rind3d.pipeline import parquet_write
from .nonlocal_audit import run as nonlocal_audit

def run(base,output):
    base=Path(base);output=Path(output);output.mkdir(parents=True,exist_ok=True);report=json.loads((base/'report.json').read_text());rows=[];scenes=[];start=time.perf_counter()
    if not report['complete']:raise ValueError('Audit requires the complete declared pilot, including failed-attempt accounting')
    for s in report['scenes']:
        path=base/'scenes'/s['scene_id'];selected=pq.read_table(path/'window_selection.parquet').to_pylist();acceptance=json.loads((path/'geometry_acceptance.private.json').read_text());recipe=json.loads((path/'recipe.private.json').read_text());components=acceptance['geometry_proof']['components']
        with RINDPack(path/'response.rindb',verify_file=True) as y,RINDPack(path/'solid.rindb',verify_file=True) as m:
            index=CountIndex.load(path/'counts.rindc',y,m);solid=sum(r[7] for r in m.index)/1024**3;lit=sum(r[7] for r in y.index)/1024**3
            for r in selected:
                c=index.query(tuple(r['origin']),r['side'])
                if not c.informative or (r['lit_free'],r['shadow_free'],r['solid'])!=(c.lit,c.shadow,c.solid):raise AssertionError('Stored current ratio metadata mismatch')
                if r['window_id']%16==0 and c!=index.direct(tuple(r['origin']),r['side']):raise AssertionError('Independent packed count mismatch')
                rows.append(dict(r,scene_id=s['scene_id'],slot=s['slot'],family=s['family'],geometry_faces=sum(c['actual_faces'] for c in components),
                                 obstacles=len(components),solid_occupancy=solid,shadow_occupancy=1-solid-lit,source_x=s['source'][0],source_y=s['source'][1],source_z=s['source'][2]))
        origins=np.array([r['origin'] for r in selected]);lengths=np.array([r['side'] for r in selected]);upper=origins+lengths[:,None]
        intersection=np.maximum(0,np.minimum(upper[:,None],upper[None])-np.maximum(origins[:,None],origins[None])).prod(axis=-1);iou=intersection/(lengths[:,None]**3+lengths[None]**3-intersection);np.fill_diagonal(iou,0)
        scenes.append({'scene_id':s['scene_id'],'family':s['family'],'components':len(components),'actual_faces':[c['actual_faces'] for c in components],
                       'declared_planes':[o['declared_planes'] for o in recipe['objects']],'half_parameters':[o['half_parameters'] for o in recipe['objects']],
                       'orientations':[o['orientation'] for o in recipe['objects']],'component_centres':[o['center'] for o in recipe['objects']],
                       'solid_occupancy':solid,'shadow_occupancy':1-solid-lit,'pairwise_iou_mean':float(iou.sum()/(256*255)),
                       'mean_max_other_iou':float(iou.max(axis=1).mean()),'max_other_iou':float(iou.max()),'selection':s['selection']})
    def histogram(a):return [sum(r['ratio_bin']==b for r in a) for b in range(10)]
    parquet_write(output/'window-counts.parquet',rows)
    summary={'complete':True,'scenes':len(scenes),'windows':len(rows),'bins':histogram(rows),'family_counts':dict(Counter(s['family'] for s in scenes)),
             'by_family':{f:histogram([r for r in rows if r['family']==f]) for f in sorted({r['family'] for r in rows})},
             'by_side':{str(side):histogram([r for r in rows if r['side']==side]) for side in (64,128,256)},'scene_records':scenes,'wall_seconds':time.perf_counter()-start}
    atomic_json(output/'distribution.json',summary)
    nonlocal_audit(base,output/'window-counts.parquet',output/'nonlocal')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.base,a.output)
