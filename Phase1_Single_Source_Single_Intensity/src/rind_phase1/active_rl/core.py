"""Public acquisition state, truth-only scoring, and PPO policy components."""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ACTIONS=('stop','left','right','up','down','grow_center','grow_se','grow_sw','grow_ne','grow_nw')


@dataclass(frozen=True)
class Window:
    x:int
    y:int
    size:int


class TileEnv:
    """Geometry-only legal actions and exact native-pixel cost on aligned tiles.

    No oracle, source, geometry, truth-based stopping or candidate lookahead.
    All windows align with full atomic tiles, so tile counts give exact costs.
    """
    def __init__(self,world,tile,initial,budget,max_steps=32):
        if (world%tile or initial.size<2*tile or initial.size%(2*tile) or
                initial.size&(initial.size-1) or tile&(tile-1) or budget<initial.size**2 or budget>world**2):
            raise ValueError('incompatible tile/initial size/budget')
        self.world,self.tile,self.budget,self.max_steps=world,tile,budget,max_steps
        self.coverage=np.zeros((world//tile,world//tile),dtype=bool)
        self.window=initial;self.spent=0;self.steps=0;self.done=False;self.reason=None;self.history=[]
        self.initial_tiles=self.acquire(initial,'initial')

    def tiles(self,w):
        if (min(w.x,w.y)<0 or w.x+w.size>self.world or w.y+w.size>self.world or
                any(v%self.tile for v in (w.x,w.y,w.size))):raise ValueError('unaligned/outside window')
        n=self.world//self.tile
        return [(y*n+x) for y in range(w.y//self.tile,(w.y+w.size)//self.tile)
                for x in range(w.x//self.tile,(w.x+w.size)//self.tile)]

    def cost(self,w):
        return sum(not self.coverage.flat[i] for i in self.tiles(w))*self.tile**2

    def proposals(self,w=None):
        w=w or self.window;x,y,s=w.x,w.y,w.size;d=s//2
        raw=[None,Window(x-d,y,s),Window(x+d,y,s),Window(x,y-d,s),Window(x,y+d,s),
             Window(x-d,y-d,2*s),Window(x,y,2*s),Window(x-s,y,2*s),Window(x,y-s,2*s),Window(x-s,y-s,2*s)]
        return [p if p is None or (min(p.x,p.y)>=0 and p.x+p.size<=self.world and p.y+p.size<=self.world) else None for p in raw]

    def legal(self):
        proposals=self.proposals();mask=[True]+[p is not None and self.cost(p)<=self.budget-self.spent for p in proposals[1:]]
        return proposals,np.array(mask,dtype=bool)

    def acquire(self,w,name):
        ids=self.tiles(w);new=[i for i in ids if not self.coverage.flat[i]]
        cost=len(new)*self.tile**2
        if self.spent+cost>self.budget:raise ValueError('over budget')
        self.coverage.flat[new]=True;self.spent+=cost;self.window=w
        self.history.append(dict(action=name,window=[w.x,w.y,w.size],new_pixels=cost,spent=self.spent))
        if self.spent==self.budget:self.done=True;self.reason='budget'
        return new

    def step(self,action):
        proposals,legal=self.legal()
        if self.done or not 0<=action<10 or not legal[action]:raise ValueError('illegal action')
        self.steps+=1
        if action==0:
            self.done=True;self.reason='policy_stop';self.history.append(dict(action='stop',window=None,new_pixels=0,spent=self.spent));return []
        new=self.acquire(proposals[action],ACTIONS[action])
        if self.steps>=self.max_steps and not self.done:self.done=True;self.reason='max_steps'
        return new

    def baseline(self,kind,rng):
        proposals,legal=self.legal();novel=[i for i in range(1,10) if legal[i] and self.cost(proposals[i])>0]
        if novel:
            if kind=='random':return int(rng.choice(novel))
            grow=[i for i in novel if i>=5]
            return (grow or novel)[0]
        queue=deque([(self.window,None)]);seen={self.window}
        while queue and len(seen)<8192:
            w,first=queue.popleft()
            for i,p in enumerate(self.proposals(w)[1:],1):
                if p is None:continue
                c=self.cost(p)
                if 0<c<=self.budget-self.spent and first is not None:return first
                if c==0 and p not in seen:seen.add(p);queue.append((p,i if first is None else first))
        return 0


def potential(probability,xy,truth,radii=(16.,32.),epsilon=1e-8):
    """Privileged TRAINING REWARD/evaluation only; never a policy input."""
    distance=(xy-torch.as_tensor(truth,device=xy.device,dtype=xy.dtype)).square().sum(-1)
    masses=torch.stack([probability[distance<=r*r].sum() for r in radii])
    return (masses+epsilon).log().mean(),masses


def belief_metrics(p,xy,truth,spacing):
    score,mass=potential(p,xy,truth)
    distance=(xy-torch.as_tensor(truth,device=xy.device,dtype=xy.dtype)).square().sum(-1).sqrt()
    ordered=p.sort(descending=True).values;cut=ordered[torch.searchsorted(ordered.cumsum(0),p.new_tensor(.9)).clamp_max(len(p)-1)]
    core=p>=cut
    nearest=distance.argmin()
    return dict(score=float(score),mass16=float(mass[0]),mass32=float(mass[1]),
        map_distance=float(distance[p.argmax()]),expected_distance=float((p*distance).sum()),
        entropy=float(-(p*p.clamp_min(1e-30).log()).sum()),core90_area=float(core.sum()*spacing**2),
        core90_covers_truth=bool(core[nearest]))


class ActorCritic(nn.Module):
    def __init__(self,hidden=128):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(6,32,5,stride=2,padding=2),nn.GELU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.GELU(),nn.Conv2d(64,64,3,stride=2,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(4),nn.Flatten())
        self.state=nn.Sequential(nn.Linear(64*16+6,hidden),nn.Tanh())
        self.actor=nn.Sequential(nn.Linear(hidden+7,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.critic=nn.Linear(hidden,1)

    def forward(self,image,scalar,actions,legal):
        h=self.state(torch.cat((self.encoder(image),scalar),1))
        logits=self.actor(torch.cat((h[:,None].expand(-1,10,-1),actions),-1)).squeeze(-1)
        return torch.distributions.Categorical(logits=logits.masked_fill(~legal,-1e9)),self.critic(h).squeeze(-1)


def advantages(rewards,values,dones,bootstrap,gamma=1.,lam=.95):
    result=torch.zeros_like(rewards);last=torch.zeros_like(bootstrap)
    for t in reversed(range(len(rewards))):
        next_value=bootstrap if t==len(rewards)-1 else values[t+1]
        live=(~dones[t]).float()
        last=rewards[t]+gamma*next_value*live-values[t]+gamma*lam*live*last
        result[t]=last
    return result,result+values
