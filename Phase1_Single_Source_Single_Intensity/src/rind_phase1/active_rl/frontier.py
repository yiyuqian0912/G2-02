"""Connected single-tile acquisition and spatial, truth-blind action scoring.

An action selects a world tile ID from the COMPLETE four-neighbour frontier.
There is no truncation of the frontier and no preview of unobserved responses.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .core import TileEnv, Window, ActorCritic


class TileProposals:
    def __init__(self, env):
        self.env = env

    def __len__(self):
        return self.env.coverage.size + 1

    def __getitem__(self, action):
        if action == 0:
            return None
        if not 0 < action < len(self):
            raise IndexError(action)
        tile = action - 1
        n = self.env.world // self.env.tile
        return Window(tile % n * self.env.tile, tile // n * self.env.tile, self.env.tile)


class FrontierEnv:
    mode = 'frontier'

    def __init__(self, world, tile, initial, budget, max_steps=256):
        if (tile < 1 or world % tile or initial.size != tile or
                budget < tile * tile or budget > world * world or budget % (tile * tile) or
                max_steps < 1):
            raise ValueError('frontier requires one initial tile and a whole-tile pixel budget')
        self.world, self.tile, self.budget, self.max_steps = world, tile, budget, max_steps
        self.coverage = np.zeros((world // tile, world // tile), dtype=bool)
        self.window = initial
        self.seed = initial
        self.spent = self.steps = 0
        self.done = False
        self.reason = None
        self.frontier = set()
        self.history = []
        self.initial_tiles = self._acquire(initial, 'initial')

    def tiles(self, window):
        if (window.size != self.tile or min(window.x, window.y) < 0 or
                window.x + self.tile > self.world or window.y + self.tile > self.world or
                window.x % self.tile or window.y % self.tile):
            raise ValueError('frontier query must be one aligned, in-bounds tile')
        return [window.y // self.tile * (self.world // self.tile) + window.x // self.tile]

    def neighbours(self, tile):
        n = self.world // self.tile
        x, y = tile % n, tile // n
        return ([tile - 1] if x else []) + ([tile + 1] if x + 1 < n else []) + \
               ([tile - n] if y else []) + ([tile + n] if y + 1 < n else [])

    def _acquire(self, window, name):
        tile = self.tiles(window)[0]
        if self.coverage.flat[tile]:
            raise ValueError('a frontier tile must not have been observed')
        if self.spent + self.tile ** 2 > self.budget:
            raise ValueError('over budget')
        self.coverage.flat[tile] = True
        self.frontier.discard(tile)
        self.frontier.update(j for j in self.neighbours(tile) if not self.coverage.flat[j])
        self.window = window
        self.spent += self.tile ** 2
        self.history.append(dict(action=name, window=[window.x, window.y, window.size],
                                 new_pixels=self.tile ** 2, spent=self.spent))
        if self.spent == self.budget:
            self.done, self.reason = True, 'budget'
        return [tile]

    def proposals(self):
        return TileProposals(self)

    def legal(self):
        mask = np.zeros(self.coverage.size + 1, dtype=bool)
        mask[0] = True
        if not self.done and self.spent + self.tile ** 2 <= self.budget:
            ids = np.fromiter(self.frontier, dtype=np.int64)
            mask[ids + 1] = True
        return self.proposals(), mask

    def cost(self, window):
        return (not self.coverage.flat[self.tiles(window)[0]]) * self.tile ** 2

    def action_name(self, action):
        return 'stop' if action == 0 else f'tile_{action - 1}'

    def step(self, action):
        action = int(action)
        if self.done or (action != 0 and action - 1 not in self.frontier):
            raise ValueError('illegal frontier action')
        self.steps += 1
        if action == 0:
            self.done, self.reason = True, 'policy_stop'
            self.history.append(dict(action='stop', window=None, new_pixels=0, spent=self.spent))
            return []
        new = self._acquire(self.proposals()[action], self.action_name(action))
        if not self.done and self.steps >= self.max_steps:
            self.done, self.reason = True, 'max_steps'
        return new

    def baseline(self, kind, rng):
        if self.done or not self.frontier:
            return 0
        candidates = sorted(self.frontier)
        if kind == 'random':
            return int(rng.choice(candidates)) + 1
        if kind not in ('compact', 'grow'):
            raise ValueError('unknown geometric frontier baseline')
        n = self.world // self.tile
        sx, sy = self.seed.x // self.tile, self.seed.y // self.tile
        return min(candidates, key=lambda j: ((j % n - sx) ** 2 + (j // n - sy) ** 2, j)) + 1


class SpatialActorCritic(nn.Module):
    """One action logit per tile, retaining spatial content at tile resolution."""
    def __init__(self, hidden=128):
        super().__init__()
        width = max(32, hidden // 2)
        groups = math.gcd(8, width)
        self.encoder = nn.Sequential(
            nn.Conv2d(6, width, 3, padding=1), nn.GroupNorm(groups, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.GroupNorm(groups, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=4, dilation=4), nn.GroupNorm(groups, width), nn.GELU())
        self.state = nn.Sequential(nn.Linear(width + 6, hidden), nn.GELU(), nn.Linear(hidden, width))
        self.tile_score = nn.Sequential(nn.Conv2d(2 * width, width, 1), nn.GELU(), nn.Conv2d(width, 1, 1))
        self.stop_score = nn.Linear(width, 1)
        nn.init.constant_(self.stop_score.bias, -2.)
        self.critic = nn.Linear(width, 1)

    def forward(self, image, scalar, actions, legal):
        features = self.encoder(image)
        state = self.state(torch.cat((features.mean((-1, -2)), scalar), 1))
        spatial = state[:, :, None, None].expand_as(features)
        tile_logits = self.tile_score(torch.cat((features, spatial), 1)).flatten(1)
        logits = torch.cat((self.stop_score(state), tile_logits), 1)
        if logits.shape != legal.shape:
            raise ValueError('frontier policy/coverage grid mismatch')
        return torch.distributions.Categorical(logits=logits.masked_fill(~legal, -1e9)), self.critic(state).squeeze(-1)


def make_env(config, world, window, budget):
    cls = FrontierEnv if config.get('environment') == 'frontier' else TileEnv
    return cls(world, config['tile_size'], window, budget, config['max_steps'])


def make_policy(config):
    cls = SpatialActorCritic if config.get('environment') == 'frontier' else ActorCritic
    return cls(config['hidden'])


def policy_names(config):
    if config.get('environment') == 'frontier':
        return ('ppo', 'random', 'compact', 'boundary', 'initial_policy')
    return ('ppo', 'random', 'grow', 'initial_policy')


def frontier_inputs(contexts, locator):
    """Only measured response, known coverage, current belief and geometry."""
    device = locator.device
    n = locator.world // locator.tile
    factor = locator.tile // locator.spacing
    coverage = torch.as_tensor(np.stack([c.env.coverage for c in contexts]), device=device).float()
    response = torch.stack([c.belief.response for c in contexts])
    probability = torch.stack([c.belief.probability for c in contexts]).reshape(-1, locator.grid, locator.grid)
    if factor > 1:
        response = F.avg_pool2d(response[:, None], factor)[:, 0]
        probability = F.avg_pool2d(probability[:, None], factor)[:, 0] * factor ** 2
    log_field = (probability.clamp_min(1e-30).log() + math.log(n * n)).clamp(-20, 20) / 20
    axis = (torch.arange(n, device=device) + .5) / n * 2 - 1
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    current = torch.zeros_like(coverage)
    current_xy = torch.tensor([[c.env.window.y // locator.tile, c.env.window.x // locator.tile]
                               for c in contexts], device=device)
    current[torch.arange(len(contexts),device=device),current_xy[:,0],current_xy[:,1]] = 1
    image = torch.stack((response, coverage, log_field, current, xx.expand_as(current), yy.expand_as(current)), 1)
    scalar = torch.tensor([[(c.env.budget - c.env.spent) / c.env.budget, c.env.spent / c.env.budget,
                            c.env.steps / c.env.max_steps, c.env.window.x / locator.world,
                            c.env.window.y / locator.world, c.env.window.size / locator.world]
                           for c in contexts], device=device, dtype=torch.float32)
    legal = torch.as_tensor(np.stack([c.env.legal()[1] for c in contexts]), device=device)
    # Empty compatibility tensor avoids [batch, all tiles, hidden] allocation.
    return image, scalar, torch.empty(len(contexts), 0, 7, device=device), legal


@torch.no_grad()
def boundary_actions(contexts, locator):
    """Heuristic using only acquired neighbouring tiles, never future pixels."""
    image, _, _, legal = frontier_inputs(contexts, locator)
    response, known = image[:, 0:1], image[:, 1:2]
    kernel = response.new_tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])[None, None]
    count = F.conv2d(known, kernel, padding=1)
    mean = F.conv2d(response * known, kernel, padding=1) / count.clamp_min(1)
    second = F.conv2d(response.square() * known, kernel, padding=1) / count.clamp_min(1)
    mixed = 4 * response * (1 - response) * known
    score = F.conv2d(mixed, kernel, padding=1) / count.clamp_min(1)
    score = score + 4 * (second - mean.square()).clamp_min(0)
    # Deterministic, geometry-only compact expansion resolves zero-information ties.
    n = locator.world // locator.tile
    ids = torch.arange(n * n, device=locator.device)
    seed = torch.tensor([[c.env.seed.x // locator.tile,c.env.seed.y // locator.tile]
                         for c in contexts],device=locator.device)
    distance = (ids[None] % n - seed[:,0,None]).square() + (ids[None] // n - seed[:,1,None]).square()
    score[:,0] -= (distance.float()/(2*n*n)*1e-6 + ids[None].float()/(n*n)*1e-9).reshape(-1,n,n)
    logits = torch.cat((score.new_full((len(contexts), 1), -1e9), score.flatten(1)), 1)
    return logits.masked_fill(~legal, -1e9).argmax(1).tolist()
