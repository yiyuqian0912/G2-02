"""One-factor field ablations on frozen B2 case 7, no geometry changes."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..models import load
from ..study import atomic_json
from .api import NativeOracle
from .build import sources


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();path=args.input/'gaussian_sum-007'
    scene=load(path/'geometry.json');source=json.loads((path/'result.json').read_text())['source_xyz_synthetic']
    modes=[('primary_scalar',False,True,dict()),('primary_regions',False,False,dict(inherited_regions=False)),
           ('primary_inherited',False,False,dict(inherited_regions=True)),
           ('reference_base_regions',True,False,dict(inherited_regions=False,tail_bounds=False,taylor=False)),
           ('reference_inherited',True,False,dict(inherited_regions=True,tail_bounds=False,taylor=False)),
           ('reference_positive_tails',True,False,dict(inherited_regions=True,tail_bounds=True,taylor=False)),
           ('reference_taylor',True,False,dict(inherited_regions=True,tail_bounds=True,taylor=True)),
           ('reference_scalar',True,True,dict())]
    rows=[];baseline=None
    for name,independent,scalar,opts in modes:
        engine=NativeOracle(scene,independent=independent,**opts);t=time.perf_counter();cpu=time.process_time();field=engine.field(source,32,120.,scalar=scalar)
        row={k:v for k,v in field.items() if k not in ('response','solid','valid')}
        row.update(mode=name,whole_call_seconds=time.perf_counter()-t,cpu_seconds=time.process_time()-cpu,valid_voxels=int(field['valid'].sum()))
        if field['complete']:
            payload=np.packbits(field['response'].ravel(),bitorder='little').tobytes()+np.packbits(field['solid'].ravel(),bitorder='little').tobytes()
            row['physical_bits_sha256']=hashlib.sha256(payload).hexdigest()
            if baseline is None:baseline=row['physical_bits_sha256']
            else:assert row['physical_bits_sha256']==baseline
        rows.append(row);atomic_json(args.output,{'case':'gaussian_sum-007','resolution':32,'native_source':sources(),'cases':rows,
                                                'note':'ordered ablations; conditional benefits, not independent additive percentages'})
        print(json.dumps({'mode':name,'complete':row['complete'],'seconds':row['wall_seconds']}),flush=True)


if __name__=='__main__':main()
