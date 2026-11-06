"""One shared layer extractor, explicit subtraction, and a conservation gate.

The ONLY model observations are the original response and window. Teacher
answers are consumed only by the separate final-set loss, never this model.
Binary masks and one scalar per layer preserve the RIND observation model.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from boundary_refined_model import edge_channels, pooled_tokens, _ConvBlock
from disentangled_model import DisentangledDecompositionField


def error_tokens(feature,sides):
    return torch.cat([F.interpolate(feature,size=(side,side),mode='bilinear',align_corners=False)
                      .flatten(2).transpose(1,2) for side in sides],1)


def single_query_block(block,query,memories):
    """Reuse decoder weights with spatial attention, not 1-slot competition.

    A softmax over one slot is constant and cannot focus on a contribution. Boundary,
    region and global roles instead attend spatially to their own memories.
    """
    q=block.self_norm(query)
    query=query+block.self_attention(q,q,q,need_weights=False)[0]
    q=block.cross_norm(query);values=[]
    for role,norm,memory in zip(block.roles,block.memory_norms,memories):
        b,n,_=q.shape;memory=norm(memory)
        qq=role.query(q).reshape(b,n,role.heads,role.width).transpose(1,2)
        kk=role.key(memory).reshape(b,-1,role.heads,role.width).transpose(1,2)
        vv=role.value(memory).reshape(b,-1,role.heads,role.width).transpose(1,2)
        value=F.scaled_dot_product_attention(qq,kk,vv)
        values.append(value.transpose(1,2).reshape(b,n,-1))
    query=query+block.cross_output(torch.cat(values,-1))
    return query+block.ff(block.ff_norm(query))


class _HardMaskSurrogate(torch.autograd.Function):
    """Exactly binary forward values; identity proxy derivative to probabilities."""
    @staticmethod
    def forward(ctx,probability,hard):return hard.to(probability.dtype)

    @staticmethod
    def backward(ctx,gradient):return gradient,None


def peel(residual,probability,enabled,*,straight_through=False):
    """Provisional debit of a learned mask, capped by the known pixel budget.

    No minimum-region seed or teacher strategy determines this mask. The largest
    feasible scalar for its CURRENT support is a constraint calculation only.
    """
    mask=(probability>=.5)&enabled[:,None,None]
    exists=mask.flatten(1).any(1)
    cap=torch.where(mask,residual,torch.inf).flatten(1).amin(1)
    a=torch.where(exists,cap,torch.zeros_like(cap))
    debit_mask=(_HardMaskSurrogate.apply(probability,mask) if straight_through else mask)
    # Budget feasibility is not another learned objective. Stop its min/switch
    # derivative; the proxy path is residual -> future features and mask -> debit.
    debit_strength=a.detach() if straight_through else a
    return mask,a,residual-debit_strength[:,None,None]*debit_mask


@torch.no_grad()
def conserved_strengths(masks,active,response,tolerance,provisional):
    """Fit nuisance strengths for fixed neural masks; this is NOT a train loss.

    GPU affine/nonnegative feasibility iterations may return unresolved. They
    never claim that an unresolved proposal has no feasible decomposition.
    No mask, count or observation is changed to force equality.
    """
    with torch.autocast(device_type=response.device.type,enabled=False):
        g=(masks&active[:,:,None,None]).flatten(2).double()
        y=response.flatten(1).double()
        gram=g@g.transpose(1,2);rhs=(g*y[:,None]).sum(2)
        inverse=torch.linalg.pinv(gram,hermitian=True,rtol=torch.finfo(torch.float64).eps*g.shape[1]*4)
        fit=(inverse@rhs[:,:,None]).squeeze(2)
        # Rank-deficient mask sets can have a positive solution even when the
        # minimum-norm least-squares representative is negative.
        floor=response.double().amax((-1,-2))*torch.finfo(torch.float64).eps*64
        if bool(((fit<=0)&active).any()):
            fit=provisional.double().clone()
            for _ in range(64):
                fit=torch.where(active,torch.maximum(fit,floor[:,None]),torch.zeros_like(fit))
                fit=fit+(inverse@(rhs-(gram@fit[:,:,None]).squeeze(2))[:,:,None]).squeeze(2)
            fit=torch.where(active,torch.maximum(fit,floor[:,None]),torch.zeros_like(fit))
        else:fit=fit*active
        reconstructed=(g*fit[:,:,None]).sum(1)
        # Only scalar-strength correction for floating-point overshoot; never
        # per-pixel rescaling of masks or clipping of residuals.
        ratios=torch.where(reconstructed>0,y/reconstructed.clamp_min(torch.finfo(torch.float64).tiny),torch.ones_like(y))
        factor=ratios.amin(1).clamp(0,1)
        factor=torch.where(factor<1,factor*(1-torch.finfo(torch.float64).eps*g.shape[1]*4),torch.ones_like(factor))
        fit=fit*factor[:,None]
        reconstructed=(g*fit[:,:,None]).sum(1)
        residual=y-reconstructed
        nonempty=g.sum(2)>0
        accepted=((residual.abs().amax(1)<=tolerance)&(residual.amin(1)>=0)
                  &((fit>0)|~active).all(1)&(nonempty|~active).all(1))
        return fit,residual.reshape_as(response),accepted


class SequentialCountShapeField(DisentangledDecompositionField):
    """v5 encoder/count warm-start; learned recurrent single-layer decoder.

    max_slots is a safety horizon, not 32 simultaneously extracted layers.
    The old intensity head is frozen and used only in the warm-started count
    encoder's internal refinement. Final strengths satisfy the known-response feasibility check.
    """
    def __init__(self, *, roundoff_ulps=8,residual_gradient='none',adaptive_checkpoint_min_layers=0, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(roundoff_ulps,int) or not 0<=roundoff_ulps<=64:
            raise ValueError('roundoff_ulps must be an integer in [0,64]')
        if type(adaptive_checkpoint_min_layers) is not int or adaptive_checkpoint_min_layers<0:
            raise ValueError('adaptive checkpoint threshold must be a nonnegative integer')
        self.adaptive_checkpoint_min_layers=adaptive_checkpoint_min_layers
        self._checkpoint_this_rollout=False
        self.roundoff_ulps=roundoff_ulps
        if residual_gradient not in ('none','straight_through'):
            raise ValueError('unknown residual gradient protocol')
        self.residual_gradient=residual_gradient
        width=self.mask_embedding.out_features
        self.state_encoder=nn.Sequential(nn.Linear(4,self.hidden),nn.GELU(),nn.Linear(self.hidden,self.hidden))
        self.context_feature=nn.Linear(width,self.hidden)
        self.residual_pixels=nn.Sequential(_ConvBlock(3,max(8,width//2)),nn.Conv2d(max(8,width//2),width,1))
        for parameter in self.intensity_head.parameters():parameter.requires_grad_(False)

    def _context(self,response,obstacle,window):
        """Encode the original observation once; reuse it for every step."""
        b,h,w=response.shape
        scale=response.float().amax((-1,-2)).clamp_min(1e-12)
        u=response.float()/scale[:,None,None]
        axis=(torch.arange(h,device=response.device,dtype=torch.float32)+.5)/h
        yy,xx=torch.meshgrid(2*axis-1,2*axis-1,indexing='ij')
        channels=[u,(response>0).float(),xx.expand_as(u),yy.expand_as(u)]
        if self.use_local_obstacle:channels.append(obstacle.float())
        pixels,*memories=self._features(torch.stack(channels,1).contiguous(memory_format=torch.channels_last),edge_channels(u))
        grids=tuple(min(limit,(h+2**level-1)//2**level) for limit,level in ((16,1),(12,2),(8,3)))
        global_grids=tuple(min(limit,(h+7)//8) for limit in (8,4,1))
        return pixels,memories,(grids,grids,global_grids),scale,u

    def _count(self,context,response,window):
        pixels,base,grids,scale,u=context
        slots=self.slot_queries[None]+self.window_encoder(window.float()/self.coordinate_scale)[:,None]
        previous=torch.zeros_like(u);masks=torch.zeros(len(response),self.max_slots,*u.shape[-2:],device=u.device)
        edges=edge_channels(u)
        for iteration in range(self.refinement_rounds):
            error=u-previous
            feature=self.error_encoder(torch.stack((error.clamp_min(0),(-error).clamp_min(0)),1))
            memory=[value+error_tokens(feature,sides)+self.memory_roles[i]
                    for i,(value,sides) in enumerate(zip(base,grids))]
            gates=[self._gate(masks,sides) for sides in grids]
            slots=slots+self.round_embedding[iteration]
            for block in self.decoder:slots=block(slots,memory,gates)
            out=super()._readout(slots,pixels,u,edges,scale,previous,response>0)
            masks=out['mask_probability'];previous=out['reconstruction']/scale[:,None,None]
        return out

    def _step(self,context,residual,window,hidden,index):
        if self.training and (self.activation_checkpoint_decomposer or self._checkpoint_this_rollout):
            return checkpoint(self._step_impl,context,residual,window,hidden,index,use_reentrant=False)
        return self._step_impl(context,residual,window,hidden,index)

    def _step_impl(self,context,residual,window,hidden,index):
        pixels,base,grids,scale,u=context
        r=(residual/scale[:,None,None]).to(pixels.dtype)
        explained=u.to(r.dtype)-r
        dynamic=self.residual_pixels(torch.stack((r,explained,(r>0).to(r.dtype)),1))
        keys=pixels+dynamic
        error=self.error_encoder(torch.stack((r,explained),1))
        memories=[value+error_tokens(error,sides)+self.memory_roles[i]
                  for i,(value,sides) in enumerate(zip(base,grids))]
        stats=torch.stack((r.float().mean((-1,-2)),r.float().amax((-1,-2)),
                           (r>0).float().mean((-1,-2)),explained.float().mean((-1,-2))),1)
        query=(self.slot_queries[index][None]+self.window_encoder(window.float()/self.coordinate_scale)
               +self.state_encoder(stats.to(pixels.dtype))+self.context_feature(keys.mean((-1,-2))))[:,None]+hidden
        for block in self.decoder:query=single_query_block(block,query,memories)
        query=self.slot_norm(query)
        logits=(torch.einsum('bsc,bchw->bshw',self.mask_embedding(query),keys)/math.sqrt(keys.shape[1])
                +self.mask_bias(query)[...,None]).squeeze(1).float()
        # Zero observed response implies every positive contribution is zero.
        # Current zero residual is NOT used to suppress a future full mask:
        # the final feasibility solve may redistribute provisional strengths.
        probability=logits.sigmoid()*(u>0)
        return logits,probability,query

    def forward(self,response,obstacle,window,candidate_xy=None,support=None,candidate_chunk=None,
                compute_location=None,verify_constraints=True):
        if compute_location:raise ValueError('sequential v6 trains decomposition only')
        if response.ndim!=3 or obstacle.shape!=response.shape or window.shape!=(len(response),3):
            raise ValueError('misaligned observation/window')
        context=self._context(response,obstacle,window)
        counts=self._count(context,response,window)
        # A few input-float32 ULPs, NOT a percentage reconstruction allowance.
        tolerance=(response.float().amax((-1,-2)).double()*torch.finfo(torch.float32).eps*self.roundoff_ulps)
        common={key:counts[key] for key in ('count_logits','global_count_logits','slot_count_logits','predicted_count')}
        return self._rollout(context,response,window,tolerance,common,verify_constraints=verify_constraints)

    def _rollout(self,context,response,window,tolerance,common,*,verify_constraints=True):
        b,h,w=response.shape
        residual=response.double();masks=[];logits=[];soft=[];strengths=[]
        hidden=torch.zeros(b,1,self.hidden,device=response.device,dtype=context[0].dtype)
        predicted_count=common['predicted_count']
        probe_horizon=getattr(self,'calibration_horizon',None)
        if probe_horizon is not None:
            if type(probe_horizon) is not int or not 1<=probe_horizon<=self.max_slots:
                raise ValueError('invalid calibration horizon')
            # Calibration only: exercise every enabled recurrent step at this
            # horizon. No optimizer update or teacher count chooses these steps.
            horizon=probe_horizon
            predicted_count=torch.full_like(predicted_count,horizon)
            common={**common,'predicted_count':predicted_count}
        else:horizon=self.max_slots if getattr(self,'calibration_full_horizon',False) else int(predicted_count.max())
        self._checkpoint_this_rollout=bool(self.training and self.adaptive_checkpoint_min_layers and
                                           horizon>=self.adaptive_checkpoint_min_layers)
        if self.training and horizon == 0:
            # A rank can autonomously predict zero layers. Keep the unused new
            # modules in the DDP graph with exactly zero contribution/gradient.
            anchor = sum(parameter.reshape(-1)[0] * 0 for module in
                         (self.state_encoder, self.context_feature, self.residual_pixels)
                         for parameter in module.parameters() if parameter.requires_grad)
            common = {**common, 'count_logits': common['count_logits'] + anchor}
        for t in range(horizon):
            enabled=t<predicted_count
            if probe_horizon is not None or getattr(self,'calibration_full_horizon',False):
                # Probe the actual full-batch worst case, even if the current
                # warm-start happens to predict only a few layers.
                z,p,next_hidden=self._step(context,residual,window,hidden,t)
            else:
                # A rare 20-layer window must not cause completed 1-layer
                # windows to execute nineteen more image/attention decoders.
                ids=enabled.nonzero(as_tuple=False).flatten()
                pixels,memories,grids,scale,u=context
                compact=(pixels[ids],[m[ids] for m in memories],grids,scale[ids],u[ids])
                zz,pp,hh=self._step(compact,residual[ids],window[ids],hidden[ids],t)
                z=response.new_full((b,h,w),-30.).index_copy(0,ids,zz.to(response.dtype))
                p=response.new_zeros((b,h,w)).index_copy(0,ids,pp.to(response.dtype))
                next_hidden=hidden.index_copy(0,ids,hh.to(hidden.dtype))
            mask,a,updated=peel(residual,p,enabled,
                straight_through=self.training and self.residual_gradient=='straight_through')
            masks.append(mask);logits.append(z);soft.append(p*enabled[:,None,None]);strengths.append(a)
            hidden=torch.where(enabled[:,None,None],next_hidden,hidden)
            residual=updated
        missing=self.max_slots-horizon
        def pack(values,shape,dtype,fill=0):
            data=torch.stack(values,1) if values else torch.empty((b,0,*shape),device=response.device,dtype=dtype)
            if missing:data=torch.cat((data,torch.full((b,missing,*shape),fill,device=response.device,dtype=dtype)),1)
            return data
        hard=pack(masks,(h,w),torch.bool)
        zs=pack(logits,(h,w),torch.float32,-30.)
        a=pack(strengths,(),torch.float64)
        active=torch.arange(self.max_slots,device=response.device)[None]<predicted_count[:,None]
        if verify_constraints:
            fit,fit_residual,accepted=conserved_strengths(hard,active,response,tolerance,a)
            final_a=torch.where(accepted[:,None],fit,a)
            final_residual=torch.where(accepted[:,None,None],fit_residual,residual)
        else:
            # The two supervised losses use only masks/counts. Defer the
            # no-grad FP64 solve until formal inference; never claim acceptance.
            final_a,final_residual=a,residual
            accepted=torch.zeros(b,device=response.device,dtype=torch.bool)
        return {**common,'mask_probability':hard.float(),'soft_mask_probability':pack(soft,(h,w),torch.float32),
                'mask_logits':zs,'intensity':final_a,'provisional_intensity':a,
                'active_slots':active,'presence_logits':torch.where(active,20.,-20.),
                'presence_probability':active.float(),'reconstruction':response.double()-final_residual,
                'residual':final_residual,'internal_residual':residual,
                'constraint_accepted':accepted,'constraint_tolerance':tolerance,
                '_constraint_verified':bool(verify_constraints),
                'executed_layers':((final_a>0)&active).sum(1),
                '_count_log_space':True,'_sequential_rollout':True}

    def decompose(self,response,obstacle,window):
        return self.forward(response,obstacle,window,compute_location=False)


def accepted_layers(output,index):
    """The public decoder returns no claimed solution for an incomplete rollout."""
    if not output.get('_constraint_verified',True) or not bool(output['constraint_accepted'][index]):return None
    on=output['active_slots'][index]
    return {'mask':output['mask_probability'][index,on].bool(),
            'intensity':output['intensity'][index,on],
            'residual':output['residual'][index]}
