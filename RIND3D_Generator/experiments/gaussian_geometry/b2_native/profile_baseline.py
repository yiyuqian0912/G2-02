"""Profile immutable B2 v1 inputs before choosing native optimizations."""
import argparse
import cProfile
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import pstats
import time
import tracemalloc

import numpy as np
from scipy.stats import qmc

from ..models import load
from ..queries import FieldOracle
from ..reference import Reference
from ..study import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for ordinal in (0,7,14,21):
        case=f'gaussian_sum-{ordinal:03d}';path=args.input/case;scene=load(path/'geometry.json')
        source=np.array(json.loads((path/'result.json').read_text())['source_xyz_synthetic'])
        oracle=FieldOracle(scene);points=qmc.Sobol(3,scramble=True,seed=902+ordinal).random_base2(6)
        points=np.array([x for x in points if oracle.membership(x).status=='FREE'])[:32]
        costs=[];decisions=[]
        for x in points:
            t=time.perf_counter();d=oracle.segment(source,x);costs.append(time.perf_counter()-t);decisions.append(d.status)
        profiler=cProfile.Profile();profiler.enable()
        for x in points:oracle.segment(source,x)
        profiler.disable();profiler.dump_stats(str(args.output/f'{case}.prof'))
        stream=io.StringIO();pstats.Stats(profiler,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(30)
        (args.output/f'{case}-profile.txt').write_text(stream.getvalue())
        stats=pstats.Stats(profiler);functions=[]
        for (file,line,name),(primitive,total,selftime,cumulative,_) in stats.stats.items():
            functions.append({'file':Path(file).name,'line':line,'function':name,'primitive_calls':primitive,
                              'total_calls':total,'self_seconds':selftime,'cumulative_seconds':cumulative})
        tracemalloc.start();before=tracemalloc.take_snapshot()
        for x in points[:8]:oracle.segment(source,x)
        after=tracemalloc.take_snapshot();current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        reference=Reference(scene,seconds=8.);t=time.perf_counter();checks=[reference.segment(source,x).status for x in points[:8]]
        rows.append({'case':case,'kernel_count':len(scene.kernels),'geometry_sha256':hashlib.sha256((path/'geometry.json').read_bytes()).hexdigest(),
                     'unprofiled_ray_seconds':costs,'decisions':dict(Counter(decisions)),
                     'profile_functions':sorted(functions,key=lambda x:x['cumulative_seconds'],reverse=True),
                     'tracemalloc_8_rays':{'peak_traced_bytes':peak,'live_traced_bytes':current,
                       'positive_live_allocation_delta':sum(max(0,x.count_diff) for x in after.compare_to(before,'lineno')),
                       'scope':'Python-traced live allocations; excludes internal GMP/Arb allocations, not an allocation count for all temporaries'},
                     'independent_8_rays_seconds':time.perf_counter()-t,'independent_statuses':dict(Counter(checks)),
                     'work':dict(oracle.work)})
    atomic_json(args.output/'summary.json',{'cases':rows,'timing_note':'cProfile and tracemalloc runs are separate from unprofiled timings; categories overlap in cumulative time'})


if __name__=='__main__':main()
