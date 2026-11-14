"""Separate count/existence, shape, strength and response objectives.

Teacher matching occurs ONLY in the loss. Model forward and its refinement
feedback always use the model's own predicted count, masks and strengths.
"""
import torch
from pairwise_ops import masked_pairwise_dot
from torch.nn import functional as F

from decomposition_model import exact_batched_assignment, slot_activation


def _single(output, batch, *, mask_weight=1., dice_weight=1., intensity_weight=1.,
            count_weight=1., global_count_weight=.5, slot_count_weight=.25,
            presence_weight=.5, empty_presence_weight=.25, reconstruction_weight=1.,
            deployment_reconstruction_weight=.25, residual_weight=0.,
            strength_log_beta=.1, matching_strength_weight=.1,
            location_weight=0., match_location_weight=0., _assignment=None):
    if not output.get('_disentangled',False):raise ValueError('disentangled objective requires its model')
    if location_weight or match_location_weight:raise ValueError('v5 objective is decomposition-only')
    weights=(mask_weight,dice_weight,intensity_weight,count_weight,global_count_weight,slot_count_weight,
             presence_weight,empty_presence_weight,reconstruction_weight,deployment_reconstruction_weight,
             residual_weight,matching_strength_weight)
    if any(not isinstance(w,(int,float)) or not 0<=w<float('inf') for w in weights):
        raise ValueError('objective weights must be finite and nonnegative')
    if not 0<strength_log_beta<float('inf'):raise ValueError('strength_log_beta must be positive')
    masks=output['mask_probability'].float();logits=output['mask_logits'].float()
    target=batch['teacher_mask'].float();real=batch['layer_mask'].bool()
    if masks.shape!=target.shape or masks.shape!=logits.shape or masks.shape[:2]!=real.shape:
        raise ValueError('misaligned complete layer targets')
    free=(~batch['obstacle']).float().flatten(1);pixels=free.sum(1).clamp_min(1)
    count=real.sum(1);number=count.clamp_min(1);empty=(~real).sum(1).clamp_min(1)
    m,t,z=masks.flatten(2),target.flatten(2),logits.flatten(2)
    bce=((F.softplus(z)*free[:,None]).sum(2)[:,None]-masked_pairwise_dot(t, z, free))/pixels[:,None,None]
    intersection=masked_pairwise_dot(t, m, free)
    dice=1-(2*intersection+1e-6)/((t*free[:,None]).sum(2)[:,:,None]+(m*free[:,None]).sum(2)[:,None]+1e-6)
    shape=bce+dice_weight*dice
    scale=batch['response'].float().amax((-1,-2)).clamp_min(1e-12)
    a=output['intensity'].float();truth=batch['teacher_intensity'].float()
    log_a=torch.log((a/scale[:,None]).clamp_min(1e-12))
    log_truth=torch.log((truth/scale[:,None]).clamp_min(1e-12))
    strength=F.smooth_l1_loss(log_a[:,None].expand_as(shape),log_truth[:,:,None].expand_as(shape),
                            beta=strength_log_beta,reduction='none')
    # One permutation for all tasks. Existence cannot hide a badly shaped layer.
    cost=shape.detach()+matching_strength_weight*strength.detach()
    cost=torch.where(real[:,:,None],cost,torch.zeros_like(cost))
    assignment=exact_batched_assignment(cost) if _assignment is None else _assignment
    pair=assignment[:,:,None]
    mean_layers=lambda value:((value*real).sum(1)/number).mean()
    shape_loss=mean_layers(shape.gather(2,pair).squeeze(2))
    intensity_loss=mean_layers(strength.gather(2,pair).squeeze(2))
    aligned_logits=output['presence_logits'].float().gather(1,assignment)
    existence=((F.softplus(-aligned_logits)*real).sum(1)/number+
               empty_presence_weight*(F.softplus(aligned_logits)*~real).sum(1)/empty).mean()
    fused_count=F.cross_entropy(output['count_logits'].float(),count)
    global_count=F.cross_entropy(output['global_count_logits'].float(),count)
    slot_count=F.cross_entropy(output['slot_count_logits'].float(),count)
    count_task=fused_count+global_count_weight*global_count+slot_count_weight*slot_count
    aligned_masks=masks.gather(1,assignment[...,None,None].expand_as(masks))
    aligned_a=a.gather(1,assignment)
    # Teacher-conditioned set reconstruction is an AUXILIARY TRAINING LOSS;
    # no teacher cardinality or mask is fed into forward / refinement.
    supervised_contribution=aligned_masks*(aligned_a*real)[...,None,None]
    supervised=supervised_contribution.sum(1)
    y=batch['response'].float();free_image=free.reshape_as(y)
    error=lambda value:((value-y).abs()*free_image).sum((-1,-2))/pixels
    supervised_error=error(supervised);deployment_error=error(output['reconstruction'].float())
    reconstruction=(supervised_error/scale).mean()
    deployment=(deployment_error/scale).mean()
    predicted_residual=y[:,None]-supervised_contribution.cumsum(1)
    true_residual=y[:,None]-(target*(truth*real)[...,None,None]).cumsum(1)
    residual_per=(((predicted_residual-true_residual).abs()*free_image[:,None]).sum((-1,-2))/pixels[:,None])
    residual=mean_layers(residual_per/scale[:,None])
    loss=(mask_weight*shape_loss+intensity_weight*intensity_loss+count_weight*count_task+
          presence_weight*existence+reconstruction_weight*reconstruction+
          deployment_reconstruction_weight*deployment+residual_weight*residual)
    selected=slot_activation(output);pred_count=selected.sum(1)
    relative_a=(aligned_a-truth).abs()/truth.clamp_min(1e-12)
    with torch.no_grad():
        hard=aligned_masks>=.5;visible=free_image[:,None].bool();teacher=target.bool()
        intersection_h=(hard & teacher & visible).sum((-1,-2))
        union=((hard | teacher) & visible).sum((-1,-2)).clamp_min(1)
        all_iou=(((intersection_h/union)>=.9)|~real).all(1)
        all_strength=((relative_a<=.1)|~real).all(1)
        exact_presence=(selected.gather(1,assignment)==real).all(1)
        relative_error=deployment_error/((y*free_image).sum((-1,-2))/pixels).clamp_min(1e-12)
        strict=(exact_presence & all_iou & all_strength & (relative_error<=.05)).float().mean()
    return {'strict_decomposition_success':strict,'loss':loss,'mask':shape_loss,'intensity':intensity_loss,'presence':existence,
            'count':fused_count,'count_global':global_count,'count_slot':slot_count,'count_task':count_task,
            'reconstruction':deployment_error.mean(),'reconstruction_supervised_normalized':reconstruction,
            'reconstruction_deployment_normalized':deployment,'residual':mean_layers(residual_per),
            'mask_dice':mean_layers(1-dice.gather(2,pair).squeeze(2)),
            'strength_relative_mae':mean_layers(relative_a),
            'strength_within_10pct':mean_layers((relative_a<=.1).float()),
            'count_accuracy':(pred_count==count).float().mean(),
            'count_global_accuracy':(output['global_count_logits'].argmax(1)==count).float().mean(),
            'slot_threshold_count_accuracy':((output['presence_probability']>=.5).sum(1)==count).float().mean(),
            'count_mae':(pred_count-count).abs().float().mean(),
            'reconstruction_relative_l1':(deployment_error/((y*free_image).sum((-1,-2))/pixels).clamp_min(1e-12)).mean(),
            'assignment':assignment,'_count_log_probability':output['count_logits'].float().log_softmax(1).detach()}


def disentangled_training_loss(output,batch,*,auxiliary_weight=.25,**kwargs):
    if not 0<=auxiliary_weight<float('inf'):raise ValueError('invalid auxiliary weight')
    result=_single(output,batch,**kwargs)
    rounds=output.get('refinement_outputs',[])
    if rounds and auxiliary_weight:
        auxiliary=torch.stack([_single(r,batch,_assignment=result['assignment'],**kwargs)['loss'] for r in rounds]).mean()
        result['loss']=result['loss']+auxiliary_weight*auxiliary
        result['auxiliary_loss']=auxiliary.detach()
    return result


def evaluation_assignment(output,batch):
    """Fixed shape/strength matching for BOTH v3 reference and v5 predictions."""
    z=output['mask_logits'].float().flatten(2);m=output['mask_probability'].float().flatten(2)
    t=batch['teacher_mask'].float().flatten(2);free=(~batch['obstacle']).float().flatten(1)
    pixels=free.sum(1).clamp_min(1)
    bce=((F.softplus(z)*free[:,None]).sum(2)[:,None]-masked_pairwise_dot(t, z, free))/pixels[:,None,None]
    inter=masked_pairwise_dot(t, m, free)
    dice=1-(2*inter+1e-6)/((t*free[:,None]).sum(2)[:,:,None]+(m*free[:,None]).sum(2)[:,None]+1e-6)
    scale=batch['response'].float().amax((-1,-2)).clamp_min(1e-12)
    pa=torch.log((output['intensity'].float()/scale[:,None]).clamp_min(1e-12))
    ta=torch.log((batch['teacher_intensity'].float()/scale[:,None]).clamp_min(1e-12))
    strength=F.smooth_l1_loss(pa[:,None].expand_as(bce),ta[:,:,None].expand_as(bce),beta=.1,reduction='none')
    cost=torch.where(batch['layer_mask'][:,:,None],bce+dice+.1*strength,torch.zeros_like(bce))
    return exact_batched_assignment(cost)
