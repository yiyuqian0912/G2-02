"""Global count/layer normalization over one complete optimizer step.

Only tiny label/weight reductions are transferred to GPU ahead of the forwards;
images stay in the existing CPU prefetch pipeline until their microbatch runs.
There are no collectives inside the model/loss or unequal validation loops.
"""
from itertools import islice
import torch
import torch.distributed as dist
from sequential_objective import class_weight


def optimizer_batches(iterator,accumulation,loss_config,device,world_size=1):
    if accumulation<1:raise ValueError('accumulation must be positive')
    iterator=iter(iterator);number=0
    while True:
        group=list(islice(iterator,accumulation))
        if not group:return
        denominators=torch.zeros(2,device=device,dtype=torch.float64)
        with torch.no_grad():
            for batch in group:
                r=batch['layer_mask'].to(device,non_blocking=True).sum(1).long()
                cw=class_weight(loss_config.get('count_class_weights'),r).double()
                sw=class_weight(loss_config.get('shape_class_weights'),r).double()
                denominators+=torch.stack((cw.sum(),(sw*r).sum()))
            if world_size>1:
                if not dist.is_initialized() or dist.get_world_size()!=world_size:
                    raise ValueError('optimizer normalization requires the matching process group')
                dist.all_reduce(denominators)
        normalizers={'count':denominators[0],'shape':denominators[1],'ddp_scale':world_size}
        for batch in group:
            number+=1
            yield number,batch,normalizers
