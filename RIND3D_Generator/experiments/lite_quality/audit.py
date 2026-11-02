"""Frozen-field distribution/count audit. Never writes into the baseline."""
import json,time
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
from rind3d.pack import RINDPack,atomic_json
from rind3d.counts import CountIndex
from rind3d.pipeline import parquet_write

BASE=Path('outputs/lite-v01-production/pilot-100')
OUT=Path('outputs/lite-v02-research')

def pct(v):
    a=np.asarray(v,dtype=float)
    return dict(zip(('p50','p95','p99','max'),map(float,[*np.percentile(a,[50,95,99]),a.max()]))) if len(a) else None

def audit():
    report=json.loads((BASE/'report.json').read_text());folder=OUT/'counts';folder.mkdir(parents=True,exist_ok=True)
    all_rows=[];scene_rows=[];start=time.perf_counter()
    for s in report['scenes']:
        path=BASE/'scenes'/s['scene_id'];records=np.fromfile(path/'window_records.bin',dtype='<u2').reshape(-1,4)
        with RINDPack(path/'response.rindb',verify_file=True,cache_bricks=0) as y,RINDPack(path/'solid.rindb',verify_file=True,cache_bricks=0) as m:
            output=folder/(s['scene_id']+'.rindc');t=time.perf_counter();cpu=time.process_time()
            index=CountIndex.load(output,y,m) if output.exists() else CountIndex.build(y,m)
            index_bytes=output.stat().st_size if output.exists() else index.save(output)
            build={'wall_seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-cpu,'ram_bytes':index.nbytes,'persistent_bytes':index_bytes,'native_cache_capacity_bytes':index.native.stats()['cache_capacity_bytes']}
            query=[];direct=[];checks=0;rows=[]
            for wid,r in enumerate(records):
                origin=tuple(map(int,r[:3]));side=int(r[3]);t=time.perf_counter();c=index.query(origin,side);query.append(time.perf_counter()-t)
                if not c.informative:raise AssertionError('Frozen READY observation is not informative')
                if wid%16==0:
                    t=time.perf_counter();other=index.direct(origin,side);direct.append(time.perf_counter()-t)
                    if c!=other:raise AssertionError('Count index/direct packed scan mismatch')
                    # An independent Python byte popcount on a complete ROI.
                    a=y.read_window(*origin,side,packed=True).data;b=m.read_window(*origin,side,packed=True).data
                    lut=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
                    if int(lut[a&~b].sum())!=c.lit or int(lut[b].sum())!=c.solid:raise AssertionError('Independent packed count mismatch')
                    checks+=1
                centre=(np.asarray(origin)+side/2)/1024
                row={'scene_id':s['scene_id'],'slot':s['slot'],'window_id':wid,'family':s['family'],'side':side,'origin':list(origin),
                     'lit_free':c.lit,'shadow_free':c.shadow,'solid':c.solid,'lit_ratio':c.lit_ratio,'ratio_bin':c.ratio_bin,
                     'geometry_faces':sum(s['diversity']['actual_faces']),'obstacles':s['diversity']['obstacles'],
                     'solid_occupancy':s['diversity']['occupancy']['solid'],'shadow_occupancy':s['diversity']['occupancy']['shadow_free'],
                     'source_x':s['source'][0],'source_y':s['source'][1],'source_z':s['source'][2],
                     'source_centre_distance':float(np.linalg.norm(centre-s['source']))}
                rows.append(row)
            # All pairwise overlap statistics are descriptive, no hard spacing.
            origins=records[:,:3].astype(float);lengths=records[:,3].astype(float);upper=origins+lengths[:,None]
            ext=np.maximum(0,np.minimum(upper[:,None],upper[None])-np.maximum(origins[:,None],origins[None]));intersection=ext.prod(axis=-1)
            volumes=lengths**3;iou=intersection/(volumes[:,None]+volumes[None]-intersection);np.fill_diagonal(iou,0)
            centres=origins+lengths[:,None]/2;distance=np.linalg.norm(centres[:,None]-centres[None],axis=-1);np.fill_diagonal(distance,np.inf)
            for i,row in enumerate(rows):row.update(max_other_window_iou=float(iou[i].max()),nearest_centre_distance=float(distance[i].min()/1024))
            summary={'scene_id':s['scene_id'],'slot':s['slot'],'family':s['family'],'bins':dict(Counter(row['ratio_bin'] for row in rows)),
                     'index_build':build,'query_seconds':pct(query),'direct_query_seconds':pct(direct),'independent_checks':checks,
                     'overlap_mean_offdiagonal':float(iou.sum()/(256*255)),'max_overlap':float(iou.max()),
                     'occupied_octants':len(set(map(tuple,(centres>=512).astype(int)))),
                     'query_total_seconds':sum(query),'direct_total_seconds':sum(direct),'native_work':index.native.stats()}
            all_rows.extend(rows);scene_rows.append(summary)
        atomic_json(OUT/'audit-progress.json',{'scenes':len(scene_rows),'elapsed_seconds':time.perf_counter()-start})
    parquet_write(OUT/'frozen-window-counts.parquet',all_rows)
    def histogram(rows):return [sum(r['ratio_bin']==b for r in rows) for b in range(10)]
    features=('geometry_faces','obstacles','solid_occupancy','shadow_occupancy','source_x','source_y','source_z','source_centre_distance','max_other_window_iou')
    correlations={}
    for key in features:
        a=np.array([r[key] for r in all_rows]);b=np.array([r['lit_ratio'] for r in all_rows])
        correlations[key]={'window_pearson':float(np.corrcoef(a,b)[0,1]),'window_spearman':float(spearmanr(a,b).statistic)}
        means=np.array([[np.mean([r[key] for r in all_rows if r['slot']==s['slot']]),np.mean([r['lit_ratio'] for r in all_rows if r['slot']==s['slot']])] for s in report['scenes']])
        correlations[key]['scene_mean_pearson']=float(np.corrcoef(means.T)[0,1])
    value={'complete':True,'scenes':100,'windows':len(all_rows),'bins':histogram(all_rows),
           'by_side':{str(side):histogram([r for r in all_rows if r['side']==side]) for side in (64,128,256)},
           'by_family':{f:histogram([r for r in all_rows if r['family']==f]) for f in sorted(set(r['family'] for r in all_rows))},
           'dark_dominated':sum(r['lit_ratio']<.5 for r in all_rows),'light_dominated':sum(r['lit_ratio']>.5 for r in all_rows),
           'equal_half':sum(r['lit_free']==r['shadow_free'] for r in all_rows),'invalid_exact_zero_or_one':0,
           'correlations':correlations,'correlation_caveat':'Windows within a scene are dependent; correlations are descriptive, no IID significance claim.',
           'index_ram_bytes':scene_rows[0]['index_build']['ram_bytes'],'scenes_detail':scene_rows,'wall_seconds':time.perf_counter()-start}
    atomic_json(OUT/'frozen-distribution-audit.json',value);return value

if __name__=='__main__':
    value=audit();print(json.dumps({k:v for k,v in value.items() if k not in ('scenes_detail','correlations','by_family')},indent=2))
