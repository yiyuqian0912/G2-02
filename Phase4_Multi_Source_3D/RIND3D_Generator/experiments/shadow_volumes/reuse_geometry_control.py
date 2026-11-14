"""Separate geometry-only reuse from geometry plus stored-mask reuse."""
import time
from rind3d.geometry import Oracle
from rind3d.pack import atomic_json
from .benchmark import ROOT,freeze,run_field,compare
from .cases import SOURCES
from .renderer import SharedGeometry


def main():
    name='rotated_box';root=ROOT/'reuse-1024'/name
    _,bundle=freeze(name);start=time.perf_counter();cpu=time.process_time();base=Oracle(bundle);shared=SharedGeometry(base);setup=time.perf_counter()-start;rows=[]
    for rank,source in enumerate(SOURCES):
        a=shared.bind(source);path=root/f'geometry-only-timing-source{rank}';row=run_field(base,a,path,1024,'D',60);rows.append(row)
    total=time.perf_counter()-start;total_cpu=time.process_time()-cpu;t=time.perf_counter()
    for rank,row in enumerate(rows):
        if row['complete']:compare(root/f'cold-source{rank}',root/f'geometry-only-timing-source{rank}')
    atomic_json(root/'geometry-only-control-timing.json',{'complete':all(r['complete'] for r in rows),'geometry_setup_once_seconds':setup,'wall_seconds':total,'cpu_seconds':total_cpu,'sources':rows,'mask_reused':False,'production_ready':False,'cross_method_verification_wall':time.perf_counter()-t,'verification_excluded_from_generation_totals':True})


if __name__=='__main__':main()
