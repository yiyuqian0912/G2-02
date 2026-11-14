"""Matched frozen-field sampling ablation, count-index overhead included."""
import argparse,json,time
from pathlib import Path
from collections import Counter
import numpy as np
from rind3d.pack import RINDPack,atomic_json
from rind3d.counts import CountIndex
from rind3d.information import InformationIndex
from rind3d.stratified_windows import select,DatasetBalance
from rind3d.pipeline import parquet_write
from .audit import BASE,OUT

def run(limit=100,modes=('stratified','mixed'),label='sampling'):
    scenes=json.loads((BASE/'report.json').read_text())['scenes'][:limit];cfg=json.loads((BASE/'manifest.json').read_text())['config'];folder=OUT/label;folder.mkdir(parents=True,exist_ok=True)
    result={}
    for mode in modes:
        balance=DatasetBalance();rows=[];meta=[];start=time.perf_counter();cpu=time.process_time()
        for s in scenes:
            path=BASE/'scenes'/s['scene_id'];t=time.perf_counter()
            with RINDPack(path/'response.rindb',verify_file=True) as y,RINDPack(path/'solid.rindb',verify_file=True) as m:
                info=InformationIndex.load(path/'information.rindi',y,m);index=CountIndex.load(OUT/'counts'/(s['scene_id']+'.rindc'),y,m)
                loaded=time.perf_counter()-t;selected=select(info,index,s['source'],seed=cfg['seed'],scene=s['slot'],balance=balance,observation={'mode':mode},natural_settings=cfg['window_selection'])
                if not selected.report['complete']:raise RuntimeError('Quota failed '+s['scene_id'])
                if mode=='natural':
                    frozen=np.fromfile(path/'window_records.bin',dtype='<u2').reshape(-1,4)
                    if not np.array_equal(frozen,np.array([w.record() for w in selected.windows])):raise AssertionError('Natural sampler behavior changed')
                scene_rows=[dict(r,scene_id=s['scene_id'],slot=s['slot'],family=s['family']) for r in selected.provenance];rows.extend(scene_rows)
                meta.append({'scene_id':s['scene_id'],'load_seconds':loaded,'selection':selected.report})
            atomic_json(folder/(mode+'-progress.json'),{'complete':False,'scenes':len(meta),'elapsed_seconds':time.perf_counter()-start,'bins':balance.bins.tolist()})
        parquet_write(folder/(mode+'-windows.parquet'),rows)
        def hist(group):return [sum(r['ratio_bin']==b for r in group) for b in range(10)]
        result[mode]={'complete':True,'scenes':len(meta),'bins':hist(rows),'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
                      'by_side':{str(side):hist([r for r in rows if r['side']==side]) for side in (64,128,256)},
                      'by_family':{family:hist([r for r in rows if r['family']==family]) for family in sorted({r['family'] for r in rows})},
                      'scene_records':meta,'actual_components':dict(Counter(r['mixture_component'] for r in rows)),
                      'shortages_are_pool_availability_not_proofs_of_absence':True}
        atomic_json(folder/'summary.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=100);p.add_argument('--modes',nargs='+',choices=('natural','stratified','mixed'),default=['stratified','mixed']);p.add_argument('--label',default='sampling');a=p.parse_args();run(a.limit,a.modes,a.label)
