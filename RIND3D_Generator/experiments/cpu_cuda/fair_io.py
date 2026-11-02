"""Conservative GPU control with checkpoint durability at least as often as CPU.

The main GPU run checkpoints every batch (32 bricks); the production CPU
stream checkpoints every eight. This isolates the small durability advantage
instead of attributing it to geometric predicates or GPU compute.
"""
import json
import time
from .benchmark import ROOT,frozen,field,compare
from .cpu import GeometryGroup
from .cuda_runtime import Runtime
from .gpu import GPUGroup
from rind3d.pack import RINDPackWriter
from experiments.shadow_volumes.cases import SOURCES


def run():
    init=RINDPackWriter.__init__;write=RINDPackWriter.write;checkpoint=RINDPackWriter.checkpoint
    last={};timing=[0.];calls=[0]
    def checkpoint_once(obj):
        count=len(obj.committed)
        if last.get(id(obj))==count:return
        tick=time.perf_counter();checkpoint(obj);timing[0]+=time.perf_counter()-tick;calls[0]+=1;last[id(obj)]=count
    def write_eight(obj,index,raw):
        value=write(obj,index,raw)
        if (index+1)%8==0:checkpoint_once(obj)
        return value
    RINDPackWriter.write=write_eight;RINDPackWriter.checkpoint=checkpoint_once
    try:
        with Runtime() as r:
            for name in ('box','poly16','multiscale24'):
                out=ROOT/'fair-io'/name
                if (out/'result.json').exists():continue
                g=GeometryGroup(frozen(name));gg=GPUGroup(g,r);a=gg.bind(SOURCES[0]);before=timing[0];count=calls[0]
                try:row=field(a,out,1024,'GPU',120)
                finally:a.close();gg.close()
                assert row['complete'];compare(out,ROOT/'full'/name/'GPU')
                # Keep the raw timing record and add a separate audit record.
                from rind3d.pack import atomic_json
                atomic_json(out/'checkpoint-control.json',{'all_bits_equal_main_gpu':True,'checkpoint_bricks':8,
                            'main_gpu_checkpoint_bricks':32,'checkpoint_calls':calls[0]-count,
                            'checkpoint_seconds':timing[0]-before,
                            'progress_json_note':'Original batch-level progress JSON retained; timings include this overhead too'})
    finally:RINDPackWriter.__init__=init;RINDPackWriter.write=write;RINDPackWriter.checkpoint=checkpoint


if __name__=='__main__':run()
