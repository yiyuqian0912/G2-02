"""Fresh deterministic random-subset + Tau-pencil baseline on the same public inputs."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter, process_time
import numpy as np
import zstandard
from rind3d_posterior.localfirst.pipeline import label_window_localfirst


def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('manifest',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--all-sizes', action='store_true', help='include 128 and 256 crops')
    p.add_argument('--sizes', type=int, nargs='+', choices=(64,128,256))
    args=p.parse_args()
    dz=zstandard.ZstdDecompressor()
    with args.output.open('x') as fp:
        for t in json.loads(args.manifest.read_bytes())['holdout']:
            sizes=args.sizes or ((64,128,256) if args.all_sizes else (64,))
            if t['L'] not in sizes: continue
            start,cpu=perf_counter(),process_time()
            slot,w,s=t['slot'],t['w'],t['s']
            with np.load(args.root/'scenes'/f'{slot//1000:04d}'/f'{slot:06d}.npz',allow_pickle=False) as z:
                indices={(int(i),int(j)):(int(a),int(n)) for i,j,a,n in z['obs_index']}
                blob=z['obs'].tobytes()
                def bits(ch):
                    a,n=indices[w,ch]
                    return np.frombuffer(dz.decompress(blob[a:a+n]),np.uint8)
                win=z['windows'][w]
                task=SimpleNamespace(lit_bits=bits(s),solid_bits=bits(-1),L=int(win['L']),
                        origin=tuple(int(win[k]) for k in ('x','y','z')),P=z['P'],K=z['K'],
                        N=int(json.loads(z['meta'].tobytes())['resolution']),key=(slot,w,s))
            r=label_window_localfirst(task,select='random')
            cells,inv=np.unique(r['cell'],return_inverse=True)
            m=np.bincount(inv,weights=r['lo'])
            if m.sum()==0: m=np.bincount(inv,weights=r['hi'])
            if m.sum()==0: raise ArithmeticError('empty reference')
            np.savez_compressed(args.output.parent/(t['key']+'_cpu.npz'),cell=cells,q=m/m.sum())
            rec=dict(task=t,status=r['status'],tv_bound=r['tv_bound'],wall_s=perf_counter()-start,
                     cpu_s=process_time()-cpu,path=r['meta']['path'])
            fp.write(json.dumps(rec)+'\n')
            fp.flush()
            print(t['key'],r['status'],rec['wall_s'],flush=True)


if __name__=='__main__': main()
