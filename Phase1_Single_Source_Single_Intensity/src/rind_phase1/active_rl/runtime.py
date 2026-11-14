"""Single-writer ownership, coordinated failures and durable training limits."""
import fcntl
import json
import os
import socket
from pathlib import Path
import torch.distributed as dist


class RunLock:
    def __init__(self, output, role='train'):
        self.path=Path(output)/'.active.lock'
        self.role=role
        self.stream=None

    def acquire(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        stream=self.path.open('a+',encoding='utf-8')
        try:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            stream.seek(0);owner=stream.read().strip();stream.close()
            raise RuntimeError(f'experiment is already active: {self.path.parent}; owner={owner}') from error
        stream.seek(0);stream.truncate()
        json.dump(dict(pid=os.getpid(),host=socket.gethostname(),role=self.role),stream)
        stream.flush();self.stream=stream

    def release(self):
        if self.stream is not None:
            fcntl.flock(self.stream.fileno(),fcntl.LOCK_UN)
            self.stream.close();self.stream=None

    def __enter__(self):
        self.acquire();return self

    def __exit__(self,*exc):
        self.release()


def rank_zero_action(action,rank,world):
    """Every rank observes a rank-zero startup/output failure before proceeding."""
    status=[None];caught=None
    if rank==0:
        try:action()
        except Exception as error:
            caught=error;status[0]=f'{type(error).__name__}: {error}'
    if world>1:dist.broadcast_object_list(status,src=0)
    if status[0] is not None:
        raise RuntimeError('rank-zero action failed: '+status[0]) from caught


def resume_checkpoint(output):
    output=Path(output)
    for name in ('last.pt','best.pt'):
        if (output/name).is_file():return output/name
    return None


def merge_training_limits(config,output,protocol_sha,checkpoint_config=None):
    path=Path(output)/'training_schedule.json'
    sources=[checkpoint_config or {}]
    if path.is_file():
        saved=json.loads(path.read_text())
        if saved.get('protocol_sha256')!=protocol_sha:
            raise ValueError('training schedule/protocol mismatch')
        sources.append(saved)
    for source in sources:
        for name,minimum in (('updates',1),('minimum_scene_visits',0)):
            if name not in source:continue
            value=source[name]
            if type(value) is not int or value<minimum:
                raise ValueError('invalid stored training limit: '+name)
            config[name]=max(config.get(name,minimum),value)


def training_limits(config,protocol_sha):
    return dict(protocol_sha256=protocol_sha,updates=config['updates'],
                minimum_scene_visits=config.get('minimum_scene_visits',0))
