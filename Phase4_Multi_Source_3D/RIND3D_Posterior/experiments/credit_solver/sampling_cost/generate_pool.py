"""Replay the same natural proposals, including rejects, until two exact valid windows. CPU only."""
import argparse
import io
import json
from pathlib import Path
from time import perf_counter,process_time

import numpy as np
from rind3d.geometry import load_geometry
from rind3d.rendering.backends import BackendFactory
from rind3d_100k.local_eval import LocalScene, evaluate_proposal, render_accepted
from rind3d_100k.physical import POLICY, _nprobe


def groups():
    return [(1,64)]+[(s,(64,128,256)[s%3]) for s in range(18,27)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    cfg=json.loads((Path(__file__).resolve().parents[4]/"RIND3D_Dataset100K/contract/RIND3D_100K_CONTRACT_v3.json").read_bytes())
    with args.out.open("x") as out:
        for slot,L in groups():
            start=perf_counter()
            with np.load(args.work/f"sampling_impact/v3/scenes/0000/{slot:06d}.npz") as z:
                bundle=load_geometry(io.BytesIO(z["geometry"].tobytes()))
                sources=np.array([[float.fromhex(v) for v in row] for row in z["sources_hex"]])
                log=z[f"log_L{L}"]
                meta=json.loads(z["meta"].tobytes())
            f=BackendFactory(cuda_allowed=False)
            scene=LocalScene(f,bundle,sources,POLICY,backend="cpu_native_certified")
            try:
                scene.enable_hull(f.context(bundle,POLICY))
                setup=perf_counter()-start
                rng=np.random.default_rng([12119,slot,L])
                t,cpu=perf_counter(),process_time()
                accepted=0
                for n,row in enumerate(log,1):
                    origin=tuple(map(int,row[:3]))
                    if int(row[9])!=0:
                        raise ValueError("unexpected non-natural stream")
                    d=evaluate_proposal(scene,sources,origin,L,float(row[7]),[1.]*4,rng,
                                        n_probe=_nprobe(cfg,L),rule="direct_interface")
                    if d.accept != bool(row[8]):
                        raise ValueError("replay decision mismatch")
                    if d.accept:
                        result=render_accepted(scene,origin,L,rule="direct_interface")
                        w=(64,128,256).index(L)*4+accepted
                        old=meta["windows_full"][w]
                        if (tuple(old["origin"])!=origin or old["mask"]!=result[4] or
                            old["counts"]!=result[2].tolist() or old["interface_edges"]!=[list(e) for e in result[5]["interface_edges"]]):
                            raise ValueError("archive mismatch")
                        del result
                        accepted+=1
                        print(json.dumps(dict(slot=slot,L=L,w=w,k=len(sources),prefix=accepted,
                            proposals=n,wall=perf_counter()-t,cpu=process_time()-cpu,setup=setup,
                            generation_stats=dict(scene.stats))),file=out,flush=True)
                        if accepted==2:
                            break
                if accepted!=2:
                    raise ValueError("incomplete prefix")
            finally:
                scene.close()


if __name__=="__main__":
    main()

