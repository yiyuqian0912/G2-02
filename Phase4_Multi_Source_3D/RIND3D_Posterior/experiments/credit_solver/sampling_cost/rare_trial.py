"""Conditional quota-preserving trial. No posterior or timing data used to trigger or choose replacements."""
import argparse
from collections import defaultdict
from fractions import Fraction
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import numpy as np
import zstandard

from rind3d.geometry import load_geometry
from rind3d.rendering.backends import BackendFactory
from rind3d_100k.local_eval import LocalScene,evaluate_proposal,render_accepted
from rind3d_100k.physical import POLICY,_packz,_nprobe
import inputs
import evidence
from cbc_search import solve
from cbc_run import warm_jit
from run import write_result
from rare_summary import reserve_slot


def trial_rows(rows):
    cutoff=float(np.quantile([r["fraction"] for r in rows if r["slot"]<18],.01))
    flags={(r["slot"],r["w"]) for r in rows if (18<=r["slot"]<36 or (r["slot"],r["w"])==(1,2))
           and r["stream"]=="natural" and r["fraction"]<=cutoff}
    return cutoff,sorted(flags)


def pack_candidate(path,geometry,meta,w,origin,physical):
    lits,solid,counts,m,mask,extra=physical
    side=solid.shape[0]
    win=np.zeros(1,dtype=[(k,"<i4") for k in ("x","y","z","L","mask")])
    win[0]=(*origin,side,mask)
    blob=io.BytesIO()
    index=[]
    compressor=zstandard.ZstdCompressor(level=3)
    for j,arr in [(-1,solid)]+list(enumerate(lits)):
        data=_packz(arr,compressor)
        index.append((0,j,blob.tell(),len(data)))
        blob.write(data)
    metadata={k:meta[k] for k in ("resolution","family","slot")}
    np.savez(path,geometry=geometry,windows=win,obs_index=np.asarray(index,dtype=np.int64),
             obs=np.frombuffer(blob.getvalue(),np.uint8),meta=np.frombuffer(json.dumps(metadata).encode(),np.uint8))


def channel_cost(path,idx,slot,w,j,tmp):
    t=perf_counter()
    obs,ident=inputs.load(path,idx,j)
    tol=Fraction(1,20)
    key=f"{slot:06d}_w{w:02d}_s{j}"
    raw,diag=solve(obs,key,tol,10)
    write_result(tmp,ident,tol,raw,diag)
    artifact=json.loads(tmp.read_bytes())
    if raw is not None:
        evidence.check_artifact(artifact,ident)
    return dict(key=key,seconds=perf_counter()-t,crop=ident["observation_xyz_sha256"],
                cells=artifact.get("cells",0),wire=artifact.get("wire_sha256"),
                status=artifact["status"],tv=diag["tv"],reason=diag["search_reason"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("rows",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--focal-from",type=Path,help="Reuse measured baseline arms; preserve the rarest original source channel and label all newly eligible channels.")
    args=p.parse_args()
    cutoff,groups=trial_rows(list(map(json.loads,args.rows.read_text().splitlines())))
    cfg=json.loads((Path(__file__).resolve().parents[4]/"RIND3D_Dataset100K/contract/RIND3D_100K_CONTRACT_v3.json").read_bytes())
    comparison={(v["slot"],v["w"]):v for v in map(json.loads,args.focal_from.read_text().splitlines())} if args.focal_from else {}
    warm_jit()
    with args.out.open("x") as output, TemporaryDirectory() as tmp:
        altfile=Path(tmp)/"alternative.npz"
        label=Path(tmp)/"label.json"
        for slot,w in groups:
            loadtime=perf_counter()
            path=args.work/f"sampling_impact/v3/scenes/0000/{slot:06d}.npz"
            with np.load(path) as z:
                geometry=z["geometry"]
                meta=json.loads(z["meta"].tobytes())
                # Used only by the original exact forward evaluator and its source-exclusion rule.
                sources=np.array([[float.fromhex(s) for s in v] for v in z["sources_hex"]])
            win=meta["windows_full"][w]
            side=win["L"]
            seen={tuple(v["origin"]) for v in meta["windows_full"] if v["L"]==side}
            rng=np.random.default_rng([12119,slot,w,739])
            probes=np.random.default_rng([12119,slot,w,740])
            factory=BackendFactory(cuda_allowed=False)
            bundle=load_geometry(io.BytesIO(geometry.tobytes()))
            scene=LocalScene(factory,bundle,sources,POLICY,backend="cpu_native_certified")
            scene.enable_hull(factory.context(bundle,POLICY))
            setup=perf_counter()-loadtime
            start=perf_counter()
            reasons=defaultdict(int)
            fractions=None
            eligible=[]
            match=False
            origin=None
            try:
                for attempt in range(1,17):
                    origin=tuple(map(int,rng.integers(0,1024-side+1,3)))
                    if origin in seen:
                        reasons["duplicate"]+=1
                        continue
                    seen.add(origin)
                    decision=evaluate_proposal(scene,sources,origin,side,0.,[1.]*4,probes,
                                               n_probe=_nprobe(cfg,side),rule="direct_interface")
                    reasons[decision.why]+=1
                    if decision.accept:
                        phys=render_accepted(scene,origin,side,"direct_interface")
                        eligible=[j for j in range(len(sources)) if phys[4]&(1<<j)]
                        fractions=[min(phys[2][j,:2])/sum(phys[2][j,:2]) for j in eligible]
                        match=phys[4]==win["mask"]
                        pack_candidate(altfile,geometry,meta,w,origin,phys)
                        del phys
                        break
            finally:
                scene.close()
            screen=perf_counter()-start
            base_fraction=min(min(c[:2])/sum(c[:2]) for j,c in enumerate(win["counts"]) if win["mask"]&(1<<j))
            focal=min((min(c[:2])/sum(c[:2]),j) for j,c in enumerate(win["counts"]) if win["mask"]&(1<<j))[1]
            take=(focal in eligible and fractions[eligible.index(focal)]>base_fraction
                  if args.focal_from else bool(match and min(fractions)>base_fraction))
            # This holdback is chosen before either posterior is computed.
            reserve=reserve_slot(slot,w)
            arms={}
            for arm in (("base","swap") if (slot+w)%2 else ("swap","base")):
                if arm=="swap" and not take:
                    continue
                if arm=="base" and args.focal_from:
                    prev=comparison[slot,w]
                    if (prev["scene"],prev["alternative_origin"],prev["alternative_fractions"])!=(meta["scene_id"],list(origin),fractions):
                        raise ValueError("alternative replay mismatch")
                    arms["base"]=prev["arms"]["base"]
                    continue
                channels=eligible if arm=="swap" else [j for j in range(len(sources)) if win["mask"]&(1<<j)]
                arms[arm]=[channel_cost(path if arm=="base" else altfile,w if arm=="base" else 0,slot,w,j,label)
                           for j in channels]
            print(json.dumps(dict(slot=slot,w=w,L=side,family=meta["family"],scene=meta["scene_id"],cutoff=cutoff,
                focal=focal,original_mask=win["mask"],alternative_channels=eligible,original_fraction=base_fraction,alternative_fractions=fractions,alternative_origin=origin,
                eligible=fractions is not None,same_mask=match,take=bool(take),reserve=reserve,setup=setup,screen=screen,
                attempts=attempt,reasons=dict(reasons),arms=arms)),file=output,flush=True)


if __name__=="__main__":
    main()
