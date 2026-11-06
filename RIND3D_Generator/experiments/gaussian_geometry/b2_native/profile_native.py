"""Optional Callgrind instruction profile, outside the timing runs and SDK."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import numpy as np

from ..models import load
from ..study import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--queries',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).parent;exe=args.output/'native-profile'
    subprocess.run(['c++','-std=c++20','-O3','-g','-fno-fast-math','-ffp-contract=off','-I'+str(root),
                    str(root/'benchmarks/native_profile.cpp'),str(root/'primary.cpp'),str(root/'reference.cpp'),
                    '-lmpfr','-lgmpxx','-lgmp','-o',str(exe)],check=True)
    case='gaussian_sum-014';scene=load(args.input/case/'geometry.json');queries=np.load(args.queries/case/'queries.npz');points=queries['receivers'][:128]
    raw=struct.pack('<II',len(scene.kernels),len(points))
    for k in scene.kernels:raw+=np.r_[k.center,k.matrix.ravel(),k.weight].astype('<f8').tobytes()
    raw+=np.array([scene.threshold.value,float(scene.threshold.kind=='exp')],dtype='<f8').tobytes()+queries['source'].astype('<f8').tobytes()+points.astype('<f8').tobytes()
    data=args.output/'input.bin';data.write_bytes(raw)
    for mode in ('primary','reference'):
        with (args.output/f'{mode}-log.txt').open('w') as log:
            subprocess.run(['valgrind','--tool=callgrind','--instr-atstart=no',
                            '--callgrind-out-file='+str(args.output/f'{mode}.callgrind'),str(exe),str(data),mode],stdout=log,stderr=log,check=True)
        dumps=sorted(args.output.glob(f'{mode}.callgrind*'))
        dump=max(dumps,key=lambda x:x.stat().st_size)
        with (args.output/f'{mode}-annotated.txt').open('w') as log:
            subprocess.run(['callgrind_annotate','--inclusive=yes','--threshold=95',str(dump)],stdout=log,check=True)
    atomic_json(args.output/'identity.json',{'case':case,'rays':len(points),'input_sha256':hashlib.sha256(raw).hexdigest(),
          'scope':'standalone O3/debug-symbol harness of the same numerical sources; instruction counts, not wall-time percentages',
          'profile_time_budget_seconds':1000,'production':False})


if __name__=='__main__':main()
