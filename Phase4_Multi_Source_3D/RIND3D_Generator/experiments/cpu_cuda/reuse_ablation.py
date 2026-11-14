"""Native per-window replay without evidence reuse: batching/reuse ablation."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import time
import numpy as np
from rind3d.geometry import load_geometry
from rind3d.windows import Window
from rind3d.pack import RINDPack,atomic_json
from experiments.shadow_volumes.cases import SOURCES
from .benchmark import ROOT,OLD,Control
from .cpu import GeometryGroup


def main():
    path=OLD/'lite-compatibility/scenes/scene_box_source0'
    records=np.fromfile(path/'window_records.bin',dtype='<u2').reshape(-1,4)
    windows=[Window(tuple(map(int,r[:3])),int(r[3])) for r in records]
    g=GeometryGroup(load_geometry(path/'geometry.npz'));a=g.bind(SOURCES[0]);b=Control(g,SOURCES[0],'B_native')
    rows=[];t=time.perf_counter();cpu=time.process_time();unique=set();refs=0
    with RINDPack(path/'response.rindb') as response,RINDPack(path/'solid.rindb') as solid,ThreadPoolExecutor(max_workers=2) as executor:
        for index,w in enumerate(windows):
            tick=time.perf_counter();observed=response.read_window(*w.origin,w.size);mask=solid.read_window(*w.origin,w.size)
            row={'index':index,'side':w.size,'read_seconds':time.perf_counter()-tick}
            tiles=[(x,y,z) for z in range(0,w.size,64) for y in range(0,w.size,64) for x in range(0,w.size,64)]
            refs+=len(tiles)
            for p in tiles:unique.add(tuple(np.array(w.origin)+p))
            for name,active in [('primary',a),('independent',b)]:
                tick=time.perf_counter()
                def job(p):return p,active.render(SOURCES[0],tuple(np.array(w.origin)+p),64,1024)
                for p,(y,m,_) in executor.map(job,tiles):
                    x,v,z=p
                    assert np.array_equal(y,observed[z:z+64,v:v+64,x:x+64])
                    assert np.array_equal(m,mask[z:z+64,v:v+64,x:x+64])
                row[name+'_seconds']=time.perf_counter()-tick
            rows.append(row)
    atomic_json(ROOT/'native-window-replay.json',{'windows':len(windows),'full_window_verification':True,
                'rows':rows,'wall_seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-cpu,
                'physical_tiles_per_pass':refs,'distinct_unaligned_tile_origins':len(unique),
                'physical_work_primary':dict(a.work),'physical_work_independent':dict(b.work),
                'note':'Same native physical implementations as evidence pilot; no cross-window evidence reuse; solid checked too',
                'totals':{k:sum(r[k] for r in rows) for k in ('read_seconds','primary_seconds','independent_seconds')}})


if __name__=='__main__':main()
