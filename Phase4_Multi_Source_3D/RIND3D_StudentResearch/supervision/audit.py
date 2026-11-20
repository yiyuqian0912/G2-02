"""Time real reads and verify the exact training tensors on every issued channel."""
import argparse
import base64
from fractions import Fraction
import gzip
import json
from pathlib import Path
import resource
import struct
from time import perf_counter, process_time

import numpy as np
import torch
from decoder import IssuedExamples, TICKS, torch_batch

HERE = Path(__file__).resolve().parent
PROOFS = HERE.parents[1]/"RIND3D_Posterior/experiments/credit_solver/assurance"
PLAN_SHA = "0f1fa5c3df22f32085cd25f44a73db66dc475d9af8490830c0bea657be833f3d"
INDEX_SHA = "13a0dacfe3141dc551c355ac51b9b1c743f292fac4fd98510d44d193ea1baaef"


def audit(campaign, scenes, output):
    torch.set_num_threads(1)
    readers = {mode:IssuedExamples(campaign, scenes, PROOFS/"results/issued-sha256.json",
                manifest_sha256=PLAN_SHA,index_sha256=INDEX_SHA,representation=mode)
               for mode in ("binary64","float32_terms")}
    references = {}
    receipts = json.loads((PROOFS.parents[1]/"global_inverse/records/inputs.json").read_bytes())
    for archive in ("heldout_raster","fresh_raster"):
        with gzip.open(PROOFS.parents[1]/"global_inverse/records"/(archive+".jsonl.gz"),"rt") as f:
            for r in map(json.loads,f):
                if r.get("grid_status")=="exact_normalized" and r["unknown_pieces"]==0:
                    references[r["task"]["key"]]=r
    results=[]
    start,cpu=perf_counter(),process_time()
    for n,key in enumerate(readers["binary64"].keys):
        entry = dict(key=key)
        path=sorted((campaign/"tasks"/key).glob("*/receipt.json"))[-1].parent/"label.json"
        data=json.loads(path.read_bytes())
        wire=base64.b64decode(data["wire"],validate=True)
        expected=np.fromiter((int(v*TICKS) for _,v in struct.iter_unpack("<Id",wire)),dtype=np.int64)
        address=np.fromiter((i for i,_ in struct.iter_unpack("<Id",wire)),dtype=np.int64)
        entry.update(cells=len(expected),side=data["problem"]["side"],bound=data["tv"])
        for mode in list(readers)[::(-1 if n%2 else 1)]:
            t,c=perf_counter(),process_time()
            x,target=readers[mode][key]
            batch,ids,weights=torch_batch([target])
            entry[mode+"_wall_s"]=perf_counter()-t
            entry[mode+"_cpu_s"]=process_time()-c
            del x
            if set(batch.unique().tolist())!={0}:
                raise ArithmeticError("wrong batch")
            addresses, counts = np.unique(ids.numpy(),return_counts=True)
            if counts.max()>3 or not np.array_equal(addresses,address) or not torch.all(ids[1:]>=ids[:-1]):
                raise ArithmeticError("address mismatch")
            # Independent additive COO interpretation on the actual CPU tensors.
            mass=weights.double().mul(TICKS)
            iticks=mass.long()
            if not torch.equal(mass,iticks.double()) or not torch.all(iticks>0):
                raise ArithmeticError("noncanonical training mass")
            sums=torch.zeros(len(expected),dtype=torch.int64)
            positions=torch.repeat_interleave(torch.arange(len(expected)),torch.from_numpy(counts))
            sums.index_add_(0,positions,iticks)
            if not np.array_equal(sums.numpy(),expected) or int(sums.sum())!=TICKS:
                raise ArithmeticError("actual training target changed")
            if mode=="float32_terms":
                entry['terms']=weights.numel()
                entry['max_terms_per_cell']=int(counts.max())
        if key in references:
            ref=references[key]
            problem = data['problem']
            receipt = receipts[key]
            if (not problem['geometry_sha256'] == ref['geometry_sha256'] == receipt['geometry']
                    or problem['observation_xyz_sha256'] != receipt['classes_xyz']
                    or any(problem[k] != receipt[k] for k in ('side','resolution','origin'))):
                raise ValueError("unmatched reference")
            q={int(k):Fraction(v) for k,v in ref['probabilities'].items()}
            p={int(k):Fraction(int(v),TICKS) for k,v in zip(address,sums.numpy())}
            actual=sum(abs(p.get(i,0)-q.get(i,0)) for i in p.keys()|q.keys())/2
            if actual>target.tv_bound or sum(q.values())!=1:
                raise ArithmeticError("violated independent posterior bound")
            entry['reference_tv']=str(actual)
        results.append(entry)
    output.mkdir(parents=True,exist_ok=False)
    (output/'channels.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in results))
    summary=dict(channels=len(results),exact_tensor_matches_each=len(results),terms=sum(r['terms'] for r in results),
                 cells=sum(r['cells'] for r in results),reference_count=sum('reference_tv' in r for r in results),
                 max_reference_tv=max(float(Fraction(r['reference_tv'])) for r in results if 'reference_tv' in r),
                 peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 wall_including_comparison_s=perf_counter()-start,cpu_including_comparison_s=process_time()-cpu,
                 torch=torch.__version__,numpy=np.__version__)
    for mode in readers:
        times=[r[mode+'_wall_s'] for r in results]
        summary[mode]=dict(wall_sum_s=sum(times),cpu_sum_s=sum(r[mode+'_cpu_s'] for r in results),
                           wall_p50_p95_max_s=np.quantile(times,[.5,.95,1]).tolist())
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("campaign","scenes","output"):
        ap.add_argument(k,type=Path)
    args=ap.parse_args()
    audit(args.campaign,args.scenes,args.output)
