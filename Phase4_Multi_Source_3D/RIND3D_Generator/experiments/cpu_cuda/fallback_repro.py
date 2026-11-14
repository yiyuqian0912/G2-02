"""Minimal reproducible UNKNOWN/CGAL cost case, default only eight rays."""
import argparse
import json
import time
import numpy as np
from rind3d.geometry import centres
from .canonical_audit import difficult_cases
from .cpu import GeometryGroup
from .cuda_runtime import Runtime
from .gpu import GPUGroup
from .benchmark import Control


def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=8);args=p.parse_args()
    if not 1<=args.count<=534:raise ValueError('count must be 1..534')
    _,bundle,source,_=difficult_cases()[1];source=np.asarray(source);g=GeometryGroup(bundle)
    points=centres((0,0,0),64,64);reference=Control(g,source,'C_shadow')
    with Runtime() as r:
        gg=GPUGroup(g,r);canonical=gg.bind(source);native=gg.bind(source,fallback_backend='certified_native')
        native.attach_independent_reference(reference)
        try:
            y,_=canonical.point_labels(points,resolve=False);unknown=np.flatnonzero(y<0)
            selected=points[unknown[:args.count]]
            t=time.perf_counter();g.canonical;setup=time.perf_counter()-t
            t=time.perf_counter();original=canonical.fallback(selected,0);original_time=time.perf_counter()-t
            t=time.perf_counter();exact=native.fallback(selected,0);native_time=time.perf_counter()-t
            assert np.array_equal(original,exact)
            print(json.dumps({'geometry_id':g.geometry_id,'source':source.tolist(),'resolution':64,
                'unknown_rays':len(unknown),'queried_unknown_rays':len(selected),'canonical_setup_seconds':setup,
                'CGAL_seconds':original_time,'certified_native_plus_independent_check_seconds':native_time,
                'all_equal':True,'receivers':selected.tolist(),'native_work':dict(native.work)},indent=2))
        finally:canonical.close();native.close();gg.close()


if __name__=='__main__':main()
