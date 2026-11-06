"""Resumable scene coverage for RL; repetition never inflates unique counts."""
from collections import deque
import numpy as np
import torch
import torch.distributed as dist


class SceneSchedule:
    def __init__(self, scenes, required_visits, seed):
        self.scenes = list(map(int, scenes))
        if not self.scenes or len(set(self.scenes)) != len(self.scenes) or required_visits < 1:
            raise ValueError('unique nonempty scenes and positive required visits are needed')
        self.required = int(required_visits)
        self.rng = np.random.default_rng(seed)
        self.visits = dict.fromkeys(self.scenes, 0)
        self.started = set()
        self.pending = set()
        self.retry = deque()
        self.queue = deque(map(int, self.rng.permutation(self.scenes)))
        self.padding_cursor = 0
        self.under_target = set(self.scenes)

    def __len__(self):
        return len(self.scenes)

    def _assign(self, scene):
        self.pending.add(scene)
        self.started.add(scene)
        return scene

    def next_scene(self):
        while self.retry:
            scene = self.retry.popleft()
            if scene in self.under_target and scene not in self.pending:
                return self._assign(scene)
        while self.queue:
            scene = self.queue.popleft()
            if scene in self.under_target and scene not in self.pending:
                return self._assign(scene)
        eligible = sorted(self.under_target - self.pending)
        if eligible:
            self.queue.extend(map(int, self.rng.permutation(eligible)))
            return self._assign(self.queue.popleft())
        # Other environments/ranks may still be completing the last scenes.
        # Real additional trajectories keep synchronous PPO batches aligned;
        # they do not count as new independent scenes.
        for _ in self.scenes:
            scene = self.scenes[self.padding_cursor % len(self.scenes)]
            self.padding_cursor += 1
            if scene not in self.pending:
                return self._assign(scene)
        raise ValueError('num_envs exceeds the scene count of this GPU partition')

    def finish_scene(self, scene):
        if scene not in self.pending:
            raise RuntimeError('completion without an assigned scene')
        self.pending.remove(scene)
        self.visits[scene] += 1
        if self.visits[scene] >= self.required:
            self.under_target.discard(scene)

    def state_dict(self):
        return dict(scenes=self.scenes, visits=[self.visits[s] for s in self.scenes],
                    started=sorted(self.started), pending=sorted(self.pending),
                    retry=list(self.retry), queue=list(self.queue),
                    rng=self.rng.bit_generator.state, padding_cursor=self.padding_cursor)

    def load_state_dict(self, state):
        if state['scenes'] != self.scenes or len(state['visits']) != len(self.scenes):
            raise ValueError('scene coverage checkpoint identity mismatch')
        if any(type(v) is not int or v < 0 for v in state['visits']):
            raise ValueError('invalid scene completion counts')
        pool = set(self.scenes)
        for key in ('started', 'pending', 'retry', 'queue'):
            if not set(state.get(key, [])) <= pool:
                raise ValueError('coverage checkpoint contains an unknown scene')
        self.visits = dict(zip(self.scenes, state['visits']))
        self.started = set(state['started'])
        self.queue = deque(state['queue'])
        # A partially collected episode is restarted and is not falsely counted
        # as completed. The sensor budget/position of that episode may change.
        self.retry = deque(dict.fromkeys(state['pending'] + state.get('retry', [])))
        self.pending.clear()
        self.rng.bit_generator.state = state['rng']
        self.padding_cursor = state.get('padding_cursor', 0)
        self.under_target = {s for s, n in self.visits.items() if n < self.required}


def make_schedule(scenes, required, seed, saved_states=None):
    schedule = SceneSchedule(scenes, required, seed)
    states = saved_states or []
    matching = [s for s in states if s['scenes'] == schedule.scenes]
    if len(matching) == 1:
        schedule.load_state_dict(matching[0])
        return schedule
    if not states:
        return schedule
    # GPU count can change after interruption. Repartition accumulated counts
    # by scene ID; restart only unfinished scenes owned by this new rank.
    visits = {}
    started, pending = set(), set()
    for state in states:
        for s, n in zip(state['scenes'], state['visits']):
            if s in visits:
                raise ValueError('overlapping saved GPU scene partitions')
            visits[s] = int(n)
        started.update(state['started'])
        pending.update(state['pending'])
        pending.update(state.get('retry', []))
    if not set(schedule.scenes) <= visits.keys():
        raise ValueError('saved scene coverage does not cover the current partition')
    schedule.visits = {s: visits[s] for s in schedule.scenes}
    schedule.started = started & set(schedule.scenes)
    schedule.retry = deque(sorted(pending & set(schedule.scenes)))
    schedule.under_target = {s for s, n in schedule.visits.items() if n < required}
    return schedule


def coverage(schedule, device, world):
    if schedule is None:
        return None
    counts = torch.tensor([len(schedule.scenes), len(schedule.started),
        sum(v > 0 for v in schedule.visits.values()),
        sum(v >= schedule.required for v in schedule.visits.values()),
        sum(schedule.visits.values())], device=device, dtype=torch.long)
    minimum = torch.tensor(min(schedule.visits.values()), device=device, dtype=torch.long)
    if world > 1:
        dist.all_reduce(counts)
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    values = counts.cpu().tolist()
    result = dict(zip(('training_scenes', 'started_unique_scenes', 'completed_unique_scenes',
                       'scenes_reaching_target', 'completed_episodes'), values))
    result.update(required_visits=schedule.required, minimum_completed_visits=int(minimum),
                  coverage_complete=values[0] == values[3])
    return result
