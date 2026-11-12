"""Supervised process pool for label work: crash- and hang-tolerant replacement for multiprocessing.Pool.

Why: the labeller runs native (numba / CUDA) code. A native fault (abort, segfault) kills the worker process, and
multiprocessing.Pool then waits forever for the lost task. This happened in the first 1K release pass.

Design:
* one task per worker at a time, over a private pipe;
* a worker that dies is reported to `on_failure(task, dict(kind="crash", exitcode=...))` and replaced;
* a task running longer than `timeout_of(task)` seconds is killed and reported as kind="timeout"; the worker is
  replaced;
* `fn` is given as "module:function"; it is called as fn(task, engine, kw) inside the worker and must return a
  JSON-serialisable dict (it should catch its own Python exceptions).
"""
import importlib
import os
import signal
import sys
import time
from collections import deque
from multiprocessing.connection import wait


def _worker_main(conn, fn_path, engine, kw, nice, sys_path):
    for p in reversed(sys_path):
        if p not in sys.path:
            sys.path.insert(0, p)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if nice:
        os.nice(nice)
    mod, name = fn_path.split(":")
    fn = getattr(importlib.import_module(mod), name)
    while True:
        t = conn.recv()
        if t is None:
            return
        conn.send(fn(t, engine, kw))


class Supervisor:
    def __init__(self, ctx, engines, fn_path, kw=None, nice=0, sys_path=()):
        self.ctx, self.fn_path, self.kw, self.nice, self.sys_path = ctx, fn_path, kw or {}, nice, list(sys_path)
        self.slots = [dict(engine=e) for e in engines]
        for s in self.slots:
            self._spawn(s)

    def _spawn(self, s):
        a, b = self.ctx.Pipe()
        p = self.ctx.Process(target=_worker_main, daemon=True,
                             args=(b, self.fn_path, s["engine"], self.kw, self.nice, self.sys_path))
        p.start(); b.close()
        s.update(proc=p, conn=a, task=None, t0=None)

    def _retire(self, s, kill):
        if kill and s["proc"].is_alive():
            s["proc"].kill()
        s["proc"].join(timeout=30)
        try:
            s["conn"].close()
        except OSError:
            pass
        self._spawn(s)

    def terminate(self):
        for s in self.slots:
            if s["proc"].is_alive():
                s["proc"].kill()

    def run(self, tasks, timeout_of, on_result, on_failure, poll=5.0):
        q = deque(tasks)
        while True:
            for s in self.slots:
                if s["task"] is None and q:
                    s["task"] = q.popleft(); s["t0"] = time.perf_counter()
                    s["conn"].send(s["task"])
            busy = [s for s in self.slots if s["task"] is not None]
            if not busy:
                break
            wait([s["conn"] for s in busy] + [s["proc"].sentinel for s in busy], timeout=poll)
            now = time.perf_counter()
            for s in busy:
                t, el = s["task"], now - s["t0"]
                got = None
                try:
                    if s["conn"].poll():
                        got = s["conn"].recv()
                except (EOFError, OSError):
                    got = None
                if got is not None:
                    s["task"] = None
                    on_result(got)
                elif not s["proc"].is_alive():
                    s["task"] = None
                    on_failure(t, dict(kind="crash", exitcode=s["proc"].exitcode, wall=el, engine=s["engine"]))
                    self._retire(s, kill=False)
                elif el > timeout_of(t):
                    s["task"] = None
                    on_failure(t, dict(kind="timeout", limit=timeout_of(t), wall=el, engine=s["engine"]))
                    self._retire(s, kill=True)
        for s in self.slots:
            try:
                s["conn"].send(None)
            except OSError:
                pass
        for s in self.slots:
            s["proc"].join(timeout=30)
