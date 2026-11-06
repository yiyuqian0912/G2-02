"""Exactly two equally weighted supervised tasks; no response/strength loss."""
import math
import torch
from pairwise_ops import masked_pairwise_dot
from torch.nn import functional as F
from decomposition_model import exact_batched_assignment

_WEIGHTS={}
_VERIFIED_MATCHING={}


@torch.no_grad()
def effective_assignment(cost,real,on):
    """Exact set assignment after removing only surplus empty/empty padding.

    Both callers give every dummy/dummy pair cost zero, every dummy/active
    pair a cardinality penalty, and every real/inactive pair a column-independent
    missing-layer cost. Permute active rows/columns first and keep
    L=max(max true count, max predicted count). All removed rows and columns
    are empty. Any optimal matching can be rearranged inside those empty rows
    and columns without changing its cost, so the L-square optimum extends
    to a full optimum. No real or enabled layer is dropped. Teacher answers
    are used here only for loss/metrics, never to choose model rollout steps.
    """
    slots=cost.shape[1]
    width=int(torch.maximum(real.sum(1),on.sum(1)).max())
    if width>=slots:return exact_batched_assignment(cost)
    row_order=(~real).long().argsort(dim=1,stable=True)
    col_order=(~on).long().argsort(dim=1,stable=True)
    if width:
        compact=cost.gather(1,row_order[:,:width,None].expand(-1,-1,slots))
        compact=compact.gather(2,col_order[:,None,:width].expand(-1,width,-1))
        matched=exact_batched_assignment(compact)
        leading=col_order[:,:width].gather(1,matched)
        ordered=torch.cat((leading,col_order[:,width:]),1)
    else:ordered=col_order
    return torch.empty_like(row_order).scatter_(1,row_order,ordered)


@torch.no_grad()
def verify_effective_assignment(device):
    """Once per process/device, compare native kernels before optimization.

    Integer-valued diagnostic costs avoid near-tie roundoff. This is a kernel
    parity check, not an artificial-scene experiment or model-quality score.
    """
    key=str(device)
    if key in _VERIFIED_MATCHING:return _VERIFIED_MATCHING[key]
    b,s=4,32;axis=torch.arange(s,device=device)
    real=axis[None]<torch.tensor([1,4,3,0],device=device)[:,None]
    real=real.roll(3,1)
    on=((axis-5)%s)[None]<torch.tensor([3,2,4,0],device=device)[:,None]
    shape=(torch.arange(b*s*s,device=device).reshape(b,s,s)*37%997).float()
    shape=torch.where(on[:,None],shape,997.)
    shape=torch.where(real[:,:,None],shape,0.)
    cost=shape+32001*(real[:,:,None]!=on[:,None,:])
    full=exact_batched_assignment(cost);small=effective_assignment(cost,real,on)
    objective=lambda assignment:cost.gather(2,assignment[:,:,None]).sum((1,2))
    if not torch.equal(objective(full),objective(small)):
        raise RuntimeError('effective-layer GPU matching parity failed; training not started')
    report={'device':key,'status':'passed','capacity':s,'effective_width':4,
            'cases':'missing/extra/zero layers, non-prefix active slots, permuted teacher rows',
            'test_kind':'kernel parity only; no model weights or dataset labels changed'}
    _VERIFIED_MATCHING[key]=report
    return report


def class_weight(values,classes):
    if values is None:return torch.ones_like(classes,dtype=torch.float32)
    key=(tuple(values),str(classes.device))
    if key not in _WEIGHTS:
        if any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('invalid training-only class weights')
        _WEIGHTS[key]=torch.tensor(values,device=classes.device,dtype=torch.float32)
    return _WEIGHTS[key][classes]


def learning_rate_scale(epoch,epochs,warmup=2,minimum=.1):
    if not 0<=warmup<epochs or not 0<minimum<=1:raise ValueError('invalid learning-rate schedule')
    if warmup and epoch<=warmup:return epoch/warmup
    fraction=(epoch-warmup)/max(1,epochs-warmup)
    return minimum+(1-minimum)*.5*(1+math.cos(math.pi*fraction))


@torch.no_grad()
def activation_assignment(output,batch):
    """Match actual enabled layers, using shape only, never teacher strength."""
    # Older reference networks emit masks for disabled slots as well. Their
    # shapes must not influence which *enabled* output matches a real target.
    # Missing predictions have the same zero mask, independently of slot ID.
    on=output['active_slots'].bool()
    m=((output['mask_probability']>=.5)&on[:,:,None,None]).flatten(2)
    t=batch['teacher_mask'].bool().flatten(2)
    free=(~batch['obstacle']).flatten(1)
    inter=masked_pairwise_dot(t.float(), m.float(), free.float())
    union=(t&free[:,None]).sum(2)[:,:,None]+(m&free[:,None]).sum(2)[:,None]-inter
    iou=inter/union.clamp_min(1)
    real=batch['layer_mask'].bool()
    mismatch=real[:,:,None]!=on[:,None,:]
    s=m.shape[1]
    # >s dominates the total shape-cost range, including dummy rows.
    cost=torch.where(real[:,:,None],1-iou,torch.zeros_like(iou))+(s+1)*mismatch
    return effective_assignment(cost,real,on)


def _shape(logits,probability,target,free):
    pixels=free.sum((-1,-2)).clamp_min(1)
    bce=(F.binary_cross_entropy_with_logits(logits.float(),target.float(),reduction='none')*free).sum((-1,-2))/pixels
    inter=(probability*target*free).sum((-1,-2))
    dice=(2*inter+1e-6)/((probability*free).sum((-1,-2))+(target*free).sum((-1,-2))+1e-6)
    # BCE/log(2) and Dice error have explicit reference scales. Equal task
    # coefficients do not assert identical gradient magnitudes.
    return .5*bce/math.log(2)+.5*(1-dice),dice


@torch.no_grad()
def training_assignment(output,batch):
    """Minimize the SAME shape objective as training, preserving set symmetry.

    Hard-IoU ties must not arbitrarily assign soft logits and make the loss
    depend on the order in which teacher masks happened to be stored.
    """
    z=output['mask_logits'].float().flatten(2)
    m=output['soft_mask_probability'].float().flatten(2)
    t=batch['teacher_mask'].float().flatten(2)
    free=(~batch['obstacle']).float().flatten(1)
    pixels=free.sum(1).clamp_min(1)
    bce=((F.softplus(z)*free[:,None]).sum(2)[:,None]-masked_pairwise_dot(t, z, free))/pixels[:,None,None]
    inter=masked_pairwise_dot(t, m, free)
    dice=(2*inter+1e-6)/((t*free[:,None]).sum(2)[:,:,None]+(m*free[:,None]).sum(2)[:,None]+1e-6)
    real=batch['layer_mask'].bool();on=output['active_slots'].bool()
    shape=.5*bce/math.log(2)+.5*(1-dice)
    shape=torch.where(on[:,None],shape,torch.ones_like(shape))
    shape=torch.where(real[:,:,None],shape,torch.zeros_like(shape))
    penalty=(shape.amax((1,2))-shape.amin((1,2)))*m.shape[1]+1
    return effective_assignment(shape+penalty[:,None,None]*(real[:,:,None]!=on[:,None,:]),real,on)


def loss(output,batch,*,count_weight=.5,mask_weight=.5,global_count_weight=.5,
         slot_count_weight=.25,location_weight=0.,match_location_weight=0.,
         count_class_weights=None,shape_class_weights=None,optimizer_normalizers=None):
    if count_weight!=.5 or mask_weight!=.5 or location_weight or match_location_weight:
        raise ValueError('v6 requires equal .5/.5 tasks and no location/reconstruction/strength loss')
    if global_count_weight<0 or slot_count_weight<0:raise ValueError('negative count subweight')
    real=batch['layer_mask'].bool();r=real.sum(1).long()
    cw=class_weight(count_class_weights,r)
    cross_sum=lambda logits:(F.cross_entropy(logits.float(),r,reduction='none')*cw).sum()
    sums=[cross_sum(output[k]) for k in ('count_logits','global_count_logits','slot_count_logits')]
    count_numerator=(sums[0]+global_count_weight*sums[1]+slot_count_weight*sums[2])
    count_numerator=count_numerator/((1+global_count_weight+slot_count_weight)*math.log(output['count_logits'].shape[1]))
    count_denominator=cw.double().sum()
    ce=sums[0]/count_denominator
    free=(~batch['obstacle']).float()
    assignment=training_assignment(output,batch)
    z=output['mask_logits'].gather(1,assignment[...,None,None].expand_as(output['mask_logits']))
    source=output['soft_mask_probability']
    m=source.gather(1,assignment[...,None,None].expand_as(source))
    enabled=output['active_slots'].gather(1,assignment)
    m=m*enabled[...,None,None]
    per,dice=_shape(z,m,batch['teacher_mask'].float(),free[:,None])
    # Missing output layers count as a shape failure, without a huge constant
    # BCE from padded logits. Cardinality CE trains the missing-layer decision.
    per=torch.where(enabled,per,torch.ones_like(per))
    sw=class_weight(shape_class_weights,r)[:,None]*real
    shape_numerator=(per*sw).sum();shape_denominator=sw.double().sum().clamp_min(1e-12)
    shape_task=shape_numerator/shape_denominator
    count_task=count_numerator/count_denominator
    mean_dice=(dice*real).sum()/real.sum().clamp_min(1)
    result={'loss':.5*count_task+.5*shape_task,'count_task':count_task,'shape_task':shape_task,
            'count':ce,'mask':shape_task,'mask_dice':mean_dice,
            'count_accuracy':(output['predicted_count']==r).float().mean(),
            'count_mae':(output['predicted_count']-r).abs().float().mean(),
            'assignment':assignment,'_count_log_probability':output['count_logits'].detach(),
            '_count_numerator':count_numerator.detach(),'_count_denominator':count_denominator.detach(),
            '_shape_numerator':shape_numerator.detach(),'_shape_denominator':shape_denominator.detach()}
    if optimizer_normalizers is not None:
        # DDP averages gradients. Scale each rank by world_size so its mean
        # equals the global weighted sum; accumulation must NOT divide again.
        n=optimizer_normalizers
        result['optimization_loss']=.5*n['ddp_scale']*(
            count_numerator/n['count'].clamp_min(1e-12)+
            shape_numerator/n['shape'].clamp_min(1e-12))
    return result


@torch.no_grad()
def primary_counts(output,batch,assignment=None):
    """33x4 device counts: windows/count-correct/teacher-layers/shape-pass."""
    real=batch['layer_mask'].bool();r=real.sum(1).long();b=len(r)
    free=(~batch['obstacle']).bool()
    if assignment is None:assignment=activation_assignment(output,batch)
    pred=output['mask_probability'].gather(1,assignment[...,None,None].expand_as(output['mask_probability']))>=.5
    pred=pred&output['active_slots'].gather(1,assignment)[...,None,None]
    target=batch['teacher_mask'].bool()
    inter=(target&pred&free[:,None]).sum((-1,-2));union=((target|pred)&free[:,None]).sum((-1,-2)).clamp_min(1)
    shape_pass=(((inter/union)>=.9)&real).sum(1)
    counts=torch.zeros(33,4,device=r.device,dtype=torch.long)
    counts.index_add_(0,r,torch.stack((torch.ones_like(r),(output['predicted_count']==r).long(),r,shape_pass),1))
    return counts


def summarize_primary(counts):
    from count_objective import GROUPS
    a=counts.tolist() if hasattr(counts,'tolist') else counts
    def result(rows):
        total=[sum(row[k] for row in rows) for k in range(4)]
        return {'windows':total[0],'count_correct':total[1],'layers':total[2],'shape_correct':total[3],
                'count_accuracy':total[1]/total[0] if total[0] else None,
                'layer_accuracy':total[3]/total[2] if total[2] else None}
    overall=result(a)
    by_group={k:result(a[lo:hi+1]) for k,(lo,hi) in GROUPS.items()}
    present=[v for v in by_group.values() if v['windows']]
    return {'layer_accuracy':overall['layer_accuracy'],
            'shape_group_macro':sum(v['layer_accuracy'] for v in present)/len(present),
            'primary_by_group':by_group,'primary_by_R':{str(r):result([row]) for r,row in enumerate(a) if row[0]},
            'primary_counts':a}
