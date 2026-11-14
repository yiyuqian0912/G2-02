"""Current count-index versus full packed ROI scans, all costs measured."""
import json,time,resource
from pathlib import Path
from collections import defaultdict
import numpy as np
from rind3d.pack import RINDPack,atomic_json
from rind3d.counts import CountIndex
from rind3d.windows import sobol_points
from .audit import BASE,OUT,pct

def run():
    old=json.loads((BASE/'report.json').read_text())['scenes'];rows=[];folder=OUT/'current-count-benchmark';folder.mkdir(exist_ok=True)
    for s in old[:18]:
        path=BASE/'scenes'/s['scene_id'];latency=defaultdict(list);direct=defaultdict(list)
        with RINDPack(path/'response.rindb',verify_file=True,cache_bricks=0) as y,RINDPack(path/'solid.rindb',verify_file=True,cache_bricks=0) as m:
            index=CountIndex.build(y,m);build=dict(index.build_report);index_bytes=index.save(folder/(s['scene_id']+'.rindc'))
            for side in (64,128,256):
                origins=np.floor(sobol_points(83,s['slot'],side,100)*(1024-side+1)).astype(int)
                for i,o in enumerate(origins):
                    # Alternate method order to avoid a systematic cache advantage.
                    methods=(index.query,index.direct) if i%2 else (index.direct,index.query);values=[]
                    for method in methods:
                        t=time.perf_counter();values.append(method(tuple(o),side));elapsed=time.perf_counter()-t
                        (latency if method.__name__=='query' else direct)[side].append(elapsed)
                    if values[0]!=values[1]:raise AssertionError('Exact index versus packed scan mismatch')
            rows.append({'slot':s['slot'],'family':s['family'],'build':build,'persistent_bytes':index_bytes,
                         'queries_per_side':100,'query_seconds':{str(k):pct(v) for k,v in latency.items()},'direct_seconds':{str(k):pct(v) for k,v in direct.items()},
                         'query_total_seconds':sum(map(sum,latency.values())),'direct_total_seconds':sum(map(sum,direct.values())),
                         'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'native_work':index.native.stats()})
    atomic_json(folder/'summary.json',{'complete':True,'rows':rows,'queries':len(rows)*300,'all_counts_exactly_equal':True,
                                      'cache_policy':'128 compressed-field bricks per channel; alternate scan/index query order; build and file hashing included separately'})

if __name__=='__main__':run()
