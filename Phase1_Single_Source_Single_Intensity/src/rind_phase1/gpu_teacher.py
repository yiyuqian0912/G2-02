"""CUDA evaluation of one complete Phase-I uniform candidate grid."""
from __future__ import annotations

import time
import numpy as np

from .gpu_visibility import TorchVisibilityRenderer


def _boundary_weights_cuda(response, boundary_lambda, boundary_sigma, torch, device,
                           pair_budget=16777216):
    """Exact boundary-distance weights using only Torch operations on ``device``."""
    observed=torch.as_tensor(response,dtype=torch.float64,device=device)
    if boundary_lambda==0:
        return observed,torch.ones_like(observed)
    boundary=torch.zeros_like(observed,dtype=torch.bool)
    horizontal=observed[:,1:]!=observed[:,:-1]
    boundary[:,1:]|=horizontal;boundary[:,:-1]|=horizontal
    vertical=observed[1:,:]!=observed[:-1,:]
    boundary[1:,:]|=vertical;boundary[:-1,:]|=vertical
    boundary_xy=torch.nonzero(boundary,as_tuple=False).to(torch.float64)
    if not len(boundary_xy):
        return observed,torch.ones_like(observed)
    height,width=observed.shape
    yy,xx=torch.meshgrid(torch.arange(height,device=device,dtype=torch.float64),
                         torch.arange(width,device=device,dtype=torch.float64),indexing='ij')
    pixels=torch.stack((yy.reshape(-1),xx.reshape(-1)),dim=1)
    boundary_chunk=min(len(boundary_xy),4096)
    pixel_chunk=max(1,min(len(pixels),pair_budget//max(1,boundary_chunk)))
    minimum=torch.empty(len(pixels),dtype=torch.float64,device=device)
    for pixel_start in range(0,len(pixels),pixel_chunk):
        current=pixels[pixel_start:pixel_start+pixel_chunk]
        current_min=torch.full((len(current),),float('inf'),dtype=torch.float64,device=device)
        for boundary_start in range(0,len(boundary_xy),boundary_chunk):
            target=boundary_xy[boundary_start:boundary_start+boundary_chunk]
            dy=current[:,None,0]-target[None,:,0]
            dx=current[:,None,1]-target[None,:,1]
            current_min=torch.minimum(current_min,(dy.square()+dx.square()).amin(dim=1))
        minimum[pixel_start:pixel_start+len(current)]=current_min
    weights=1.0+float(boundary_lambda)*torch.exp(
        -minimum.reshape(height,width)/(2.0*float(boundary_sigma)**2))
    return observed,weights


def evaluate_uniform_grid_cuda(dataset,scene_id,observation,candidate_xy,*,
                               alpha=1.0,beta=0.0,boundary_lambda=0.0,
                               boundary_sigma=1.0,temperature=0.05,device='cuda:0',
                               max_candidate_batch=16384,pixel_chunk=16384,
                               pair_budget=134217728):
    """Return validity and response cost; all candidate/pixel work stays on CUDA."""
    import torch
    if beta!=0:
        raise ValueError('CUDA Phase-I teacher currently requires beta=0')
    if max_candidate_batch<1 or pixel_chunk<1 or pair_budget<1:
        raise ValueError('CUDA batch, pixel chunk and pair budget must be positive')
    candidates=np.asarray(candidate_xy,dtype=np.float64)
    scene=dataset.get_scene(int(scene_id))
    count=int(scene.get('obstacle_count',len(scene['obstacle_types'])))
    renderer=TorchVisibilityRenderer(
        observation['window'],observation['obstacle'],
        scene['obstacle_types'][:count],scene['obstacle_params'][:count],
        device=device,pixel_chunk=pixel_chunk)
    free_pixels=max(1,len(renderer.pixels))
    configured_batch=int(max_candidate_batch)
    effective_batch=max(1,min(configured_batch,pair_budget//free_pixels))
    # Report the largest batch that can actually be submitted.  At the Phase-I
    # spacing of 16 there are only 4096 world candidates, so reporting the
    # configured ceiling (for example 8192) as an effective batch is misleading.
    submitted_batch=max(1,min(effective_batch,len(candidates)))
    torch_device=renderer.device
    observed,weights=_boundary_weights_cuda(
        observation['response'],boundary_lambda,boundary_sigma,torch,torch_device)
    denominator=weights.sum()
    candidate_tensor=torch.as_tensor(candidates,dtype=torch.float64,device=torch_device)
    valid_gpu=torch.zeros(len(candidates),dtype=torch.bool,device=torch_device)
    costs_gpu=torch.full((len(candidates),),float('inf'),dtype=torch.float64,device=torch_device)
    batches=0;started=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(candidates),effective_batch):
            stop=min(start+effective_batch,len(candidates));points=candidate_tensor[start:stop]
            current_valid=renderer.positions_valid_tensor(points)
            valid_gpu[start:stop]=current_valid
            indices=torch.nonzero(current_valid,as_tuple=False).flatten()
            if not len(indices):
                continue
            selected=points[indices]
            rendered=renderer.render_tensor(selected)
            current_cost=(weights[None]*torch.abs(rendered.to(torch.float64)-observed[None])).sum(
                dim=(1,2))/denominator
            local_costs=costs_gpu[start:stop]
            local_costs[current_valid]=float(alpha)*current_cost
            batches+=1
        probability_gpu=torch.zeros_like(costs_gpu)
        selected_costs=costs_gpu[valid_gpu]
        if len(selected_costs):
            weights_gpu=torch.exp(-(selected_costs-selected_costs.min())/float(temperature))
            probability_gpu[valid_gpu]=weights_gpu/weights_gpu.sum()
    if torch_device.type=='cuda':torch.cuda.current_stream(torch_device).synchronize()
    valid=valid_gpu.cpu().numpy();costs=costs_gpu.cpu().numpy()
    probabilities=probability_gpu.cpu().numpy()
    return valid,costs,probabilities,{
        'gpu_device':str(torch_device),'gpu_configured_batch_size':configured_batch,
        'gpu_effective_batch_size':submitted_batch,
        'gpu_candidate_count':int(len(candidates)),
        'gpu_render_batches':batches,'gpu_free_pixels':free_pixels,
        'gpu_boundary_weights':True,'gpu_probability_normalization':True,
        'elapsed_seconds':time.perf_counter()-started,
    }
