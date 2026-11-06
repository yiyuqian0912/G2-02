"""Frozen Phase1 queries and tile-deduplicated, geometry-blind belief state."""
from collections import OrderedDict
import math
import numpy as np
import torch
from torch.nn import functional as F
from rind_phase1.train import load_model
from rind_phase1.sampling import grid_coordinates,require_resolved_sampling
from .core import potential


class FusionConflictError(RuntimeError):
    """The supplied fields have no common nonzero support."""


class FrozenLocator:
    def __init__(self,dataset,checkpoint,device,*,tile=64,spacing=16,chunk=256,batch_size=16,cache_size=4096):
        self.dataset=dataset;self.device=torch.device(device)
        self.tile,self.spacing,self.chunk,self.batch_size,self.cache_size=tile,spacing,chunk,batch_size,cache_size
        self.model,self.checkpoint=load_model(checkpoint)
        self.world=int(dataset.manifest['global_size']) if dataset is not None else int(self.model.coordinate_scale)
        if self.model.use_obstacle or self.model.use_scene_geometry or self.model.normalization_support!='world':
            raise ValueError('RL requires a geometry-blind Phase1 checkpoint with world normalization')
        if self.model.coordinate_scale!=self.world:raise ValueError('checkpoint/world coordinate mismatch')
        require_resolved_sampling(self.world,spacing,self.model.xy.frequencies)
        self.model.to(self.device).eval().requires_grad_(False)
        xy,self.grid=grid_coordinates(self.world,spacing);self.xy=torch.tensor(xy,dtype=torch.float32,device=self.device)
        self.cache=OrderedDict();self._packed=OrderedDict();self.queries=self.hits=self.uniform=0
        self.axis=torch.arange(tile,device=self.device);self.prior=torch.full((len(xy),),-math.log(len(xy)),device=self.device)

    def pixels(self,scene,tiles):
        # Hidden simulator state stays here. Only acquired IDs are decoded;
        # neither the full bits nor source metadata is exposed to the actor.
        if scene not in self._packed:
            raw=np.array(self.dataset._base.visibility_bits[scene,0],copy=True)
            self._packed[scene]=torch.as_tensor(raw,device=self.device,dtype=torch.uint8)
            if len(self._packed)>256:self._packed.popitem(last=False)
        bits=self._packed[scene];self._packed.move_to_end(scene)
        n=self.world//self.tile;ids=torch.as_tensor(tiles,device=self.device)
        x=(ids%n)*self.tile;y=(ids//n)*self.tile
        flat=(y[:,None,None]+self.axis[None,:,None])*self.world+x[:,None,None]+self.axis[None,None,:]
        return ((bits[flat//8]>>(flat%8).to(torch.uint8))&1).float()

    @torch.no_grad()
    def query(self,requests):
        """Only already-acquired (scene,tile) keys; batch expensive model calls."""
        result={};missing=[]
        for key in dict.fromkeys(requests):
            if key in self.cache:
                self.hits+=1;self.cache.move_to_end(key);result[key]=self.cache[key]
            else:missing.append(key)
        for start in range(0,len(missing),self.batch_size):
            keys=missing[start:start+self.batch_size]
            images=torch.empty(len(keys),self.tile,self.tile,device=self.device);by_scene={}
            for j,(scene,tile) in enumerate(keys):by_scene.setdefault(scene,[]).append((j,tile))
            for scene,pairs in by_scene.items():
                images.index_copy_(0,torch.tensor([j for j,t in pairs],device=self.device),self.pixels(scene,[t for j,t in pairs]))
            n=self.world//self.tile
            windows=torch.tensor([[tile%n*self.tile,tile//n*self.tile,self.tile] for scene,tile in keys],device=self.device,dtype=torch.float32)
            lp,coarse,informative=self.score_observed(images,windows)
            flags=informative.cpu().tolist()
            for i,key in enumerate(keys):
                response=coarse[i]
                value=(lp[i].clone() if flags[i] else self.prior,response.clone(),flags[i]);result[key]=value;self.cache[key]=value
                if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return result

    @torch.no_grad()
    def score_observed(self,images,windows):
        """Deployment API: raw ACQUIRED single-source responses and world ROIs.

        Reads neither dataset nor source metadata. In the noiseless single-
        source model, positive intensity is normalized to a unit binary mask.
        """
        images=torch.as_tensor(images,device=self.device,dtype=torch.float32)
        windows=torch.as_tensor(windows,device=self.device,dtype=torch.float32)
        if (images.ndim!=3 or images.shape[1:]!=(self.tile,self.tile) or windows.shape!=(len(images),3) or
                not bool(torch.isfinite(images).all()) or bool((images<0).any()) or
                not bool(torch.isfinite(windows).all()) or bool((windows[:,2]!=self.tile).any()) or
                bool((windows[:,:2]<0).any()) or bool((windows[:,:2]+self.tile>self.world).any())):
            raise ValueError('expected finite nonnegative acquired tile responses')
        images=(images>0).float()
        informative=(images.amax((-1,-2))-images.amin((-1,-2)))>0
        lp=self.prior[None].expand(len(images),-1).clone();active=informative.nonzero().flatten()
        if len(active):
            encoded=self.model.encode_observation(images[active],torch.zeros_like(images[active],dtype=torch.bool),windows[active])
            scores=[]
            for j in range(0,len(self.xy),self.chunk):
                xy=self.xy[None,j:j+self.chunk].expand(len(active),-1,-1)
                scores.append(self.model.score_encoded(encoded,xy).float())
            energy=torch.cat(scores,1)
            if not bool(torch.isfinite(energy).all()):raise FloatingPointError('nonfinite frozen Phase1 energy')
            lp[active]=(-energy).log_softmax(1);self.queries+=len(active)
            if not bool(torch.isfinite(lp[active]).all()):raise FloatingPointError('nonfinite frozen Phase1 log probability')
        self.uniform+=len(images)-len(active)
        return lp,F.avg_pool2d(images[:,None],self.spacing)[:,0],informative


class Belief:
    def __init__(self,locator,discount=.5,method='product'):
        if method not in ('product','minimum'):raise ValueError('unknown fusion method')
        self.locator=locator;self.discount=discount;self.method=method;self.seen=set();self.logs={}
        self.minimum=torch.full_like(locator.prior,torch.inf)
        self.informative_tiles=0
        self.response=torch.zeros(locator.grid,locator.grid,device=locator.device)
        self.probability=locator.prior.exp().clone()

    def update(self,scene,tiles,answers,*,finite_model_fields=False):
        n=self.locator.world//self.locator.tile;side=self.locator.tile//self.locator.spacing
        for tile in tiles:
            if tile in self.seen:continue
            answer=answers[(scene,tile)];lp,response=answer[:2]
            informative=bool(answer[2]) if len(answer)>2 else not torch.equal(lp,self.locator.prior)
            if self.method=='minimum' and informative:
                if lp.shape!=self.locator.prior.shape:raise ValueError('single-window candidate grid mismatch')
                if not finite_model_fields and (bool(torch.isnan(lp).any()) or
                        bool(torch.isposinf(lp).any()) or not bool(torch.isfinite(lp).any())):
                    raise FusionConflictError('invalid or empty single-window probability field')
                proposed=torch.minimum(self.minimum,lp)
                if not finite_model_fields and not bool(torch.isfinite(proposed).any()):
                    raise FusionConflictError('no common nonzero support; no uniform fallback')
                self.minimum=proposed;self.informative_tiles+=1
            elif self.method=='product':self.logs[tile]=lp
            self.seen.add(tile)
            x,y=(tile%n)*side,(tile//n)*side;self.response[y:y+side,x:x+side]=response
        # Canonical order makes the result independent of visits and overlaps.
        if self.method=='minimum' and self.informative_tiles:
            # Keep the RAW per-window minimum. Never recursively take the min
            # against a previously renormalized fused field (order-dependent).
            self.probability=self.minimum.softmax(0)
        elif self.logs:
            evidence=torch.stack([self.logs[i]-self.locator.prior for i in sorted(self.logs)]).sum(0)
            score=self.locator.prior+self.discount*evidence
            if not bool(torch.isfinite(score).any()):raise FusionConflictError('no common nonzero support')
            self.probability=score.softmax(0)
        return self.probability


def policy_inputs(contexts,locator):
    """Public state only: deliberately does not access context.truth."""
    if getattr(contexts[0].env,'mode',None)=='frontier':
        from .frontier import frontier_inputs
        return frontier_inputs(contexts,locator)
    device=locator.device;g=locator.grid;world=locator.world;scalars=[];features=[];masks=[]
    axis=(torch.arange(g,device=device)+.5)/g*2-1;yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    coverage=torch.as_tensor(np.stack([c.env.coverage for c in contexts]),device=device)
    factor=locator.tile//locator.spacing
    coverage=coverage.repeat_interleave(factor,1).repeat_interleave(factor,2).float()
    responses=torch.stack([c.belief.response for c in contexts])
    probability=torch.stack([c.belief.probability for c in contexts])
    log_field=(probability.clamp_min(1e-30).log()+math.log(g*g)).clamp(-20,20).reshape(-1,g,g)/20
    windows=torch.tensor([[c.env.window.x,c.env.window.y,c.env.window.size] for c in contexts],device=device)
    axis_world=(torch.arange(g,device=device)+.5)*locator.spacing
    view=((axis_world[None,None,:]>=windows[:,0,None,None]) & (axis_world[None,None,:]<(windows[:,0]+windows[:,2])[:,None,None]) &
          (axis_world[None,:,None]>=windows[:,1,None,None]) & (axis_world[None,:,None]<(windows[:,1]+windows[:,2])[:,None,None])).float()
    image=torch.stack((responses,coverage,log_field,view,xx.expand_as(view),yy.expand_as(view)),1)
    for c in contexts:
        e=c.env;w=e.window
        scalars.append([(e.budget-e.spent)/e.budget,e.spent/e.budget,e.steps/e.max_steps,w.x/world,w.y/world,w.size/world])
        proposals,legal=e.legal();row=[]
        for i,p in enumerate(proposals):
            row.append([0.,0.,0.,0.,0.,0.,1.] if i==0 or p is None else
                       [p.x/world,p.y/world,p.size/world,e.cost(p)/e.budget,(p.x-w.x)/world,(p.y-w.y)/world,0.])
        features.append(row);masks.append(legal)
    return (image,torch.tensor(scalars,device=device,dtype=torch.float32),
            torch.tensor(features,device=device,dtype=torch.float32),torch.as_tensor(np.stack(masks),device=device))
