"""Truth-free deployment bridge: acquired response tiles -> next ROI request."""
from types import SimpleNamespace
import torch
from .prepare import sha
from .core import ActorCritic,TileEnv,Window,ACTIONS
from .locator import FrozenLocator,Belief,policy_inputs
from .frontier import make_env,make_policy


class UnknownFieldAgent:
    def __init__(self,policy_checkpoint,phase1_checkpoint,initial_window,budget,device='cpu'):
        saved=torch.load(policy_checkpoint,map_location='cpu',weights_only=True);c=saved['config']
        if saved.get('locator_snapshot_sha256')!=sha(phase1_checkpoint):
            raise ValueError('policy requires its matching frozen Phase1 snapshot')
        self.config=c
        self.locator=FrozenLocator(None,phase1_checkpoint,device,tile=c['tile_size'],spacing=c['spacing'],
            chunk=c['candidate_chunk'],batch_size=c['locator_batch_size'])
        self.policy=make_policy(c).to(self.locator.device).eval()
        self.policy.load_state_dict(saved['policy'])
        self.env=make_env(c,self.locator.world,Window(*initial_window),budget)
        self.context=SimpleNamespace(env=self.env,belief=Belief(self.locator,c['fusion_discount'],c.get('fusion_method','product')))
        self.pending=('initial',None,self.env.initial_tiles,self.env.window)

    def request(self):
        """Return a pending request; never inspect future responses or truth."""
        if self.pending is None:
            if self.env.done:return None
            with torch.no_grad():d,_=self.policy(*policy_inputs([self.context],self.locator));action=int(d.logits.argmax(1))
            if action==0:self.env.step(0);return None
            window=self.env.proposals()[action]
            tiles=[i for i in self.env.tiles(window) if not self.env.coverage.flat[i]]
            name=self.env.action_name(action) if getattr(self.env,'mode',None)=='frontier' else ACTIONS[action]
            self.pending=(name,action,tiles,window)
        name,action,tiles,w=self.pending;n=self.locator.world//self.locator.tile;t=self.locator.tile
        return dict(action=name,window=[w.x,w.y,w.size],new_pixels=len(tiles)*t*t,
            tiles={i:[i%n*t,i//n*t,t] for i in tiles},spent_before=self.env.spent if action is not None else 0)

    def observe(self,responses):
        """responses maps requested tile ID to its measured [size,size] response.

        Missing/extra/invalid blocks do not advance a noninitial request.
        Zero-cost reposition requests are acknowledged with observe({}).
        """
        if self.pending is None:raise RuntimeError('call request() first')
        name,action,tiles,w=self.pending
        if set(responses)!=set(tiles):raise ValueError('supply exactly the requested new tile IDs')
        answers={};t=self.locator.tile;n=self.locator.world//t
        if tiles:
            images=torch.stack([torch.as_tensor(responses[i],device=self.locator.device,dtype=torch.float32) for i in tiles])
            windows=[[i%n*t,i//n*t,t] for i in tiles]
            lp,coarse,informative=self.locator.score_observed(images,windows)
            flags=informative.cpu().tolist()
            answers={(0,i):(lp[j],coarse[j],flags[j]) for j,i in enumerate(tiles)}
        if action is not None:
            actual=self.env.step(action)
            if actual!=tiles:raise RuntimeError('acquisition bookkeeping changed')
        self.context.belief.update(0,tiles,answers,finite_model_fields=True);self.pending=None
        return self.context.belief.probability.detach().clone()

    @property
    def probability(self):return self.context.belief.probability.detach().clone()
