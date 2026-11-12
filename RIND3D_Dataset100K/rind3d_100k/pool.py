"""Streaming supervised worker pool (single node, many GPUs).

It generalises `rind3d_1k.supervise` (written after the 1K worker-pool stall):
* each worker slot has its own environment, e.g. `CUDA_VISIBLE_DEVICES=<gpu>`, which is applied before any CUDA
  library is imported;
* each slot runs its own entry function (physical rendering or labelling);
* tasks are submitted one at a time (no unbounded buffering);
* every task has a wall-clock limit and every worker an RSS limit. A dead worker (abort, segfault, OOM-kill), a timeout
  or a memory overrun is reported once as a failure event, and the worker is replaced;
* Python exceptions are returned by the entry function itself as error results.
"""
import importlib
import os
import signal
import sys
import time
from multiprocessing.connection import wait


def _worker_main(conn, fn_path, engine, kw, env, nice, sys_path):
    os.environ.update({k: str(v) for k, v in env.items()})
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


class WorkerPool:
    def __init__(self, ctx, specs, sys_path=(), nice=0, rss_limit_bytes=None):
        """specs: list of dict(name, kind, fn, engine, kw, env)."""
        self.ctx, self.sys_path, self.nice = ctx, list(sys_path), nice
        self.rss_limit = rss_limit_bytes
        self.slots = []
        for sp in specs:
            s = dict(sp); s.setdefault("kw", {}); s.setdefault("env", {})
            self.slots.append(s); self._spawn(s)
        self._last_mem = 0.0

    def _spawn(self, s):
        a, b = self.ctx.Pipe()
        p = self.ctx.Process(target=_worker_main, daemon=True,
                             args=(b, s["fn"], s["engine"], s["kw"], s["env"], self.nice, self.sys_path))
        p.start(); b.close()
        s.update(proc=p, conn=a, task=None, t0=None, limit=None, restarts=s.get("restarts", -1) + 1)

    def _retire(self, s, kill):
        if kill and s["proc"].is_alive():
            s["proc"].kill()
        s["proc"].join(timeout=30)
        try:
            s["conn"].close()
        except OSError:
            pass
        self._spawn(s)

    def idle(self, kind):
        return [s for s in self.slots if s["kind"] == kind and s["task"] is None]

    def busy(self, kind=None):
        return [s for s in self.slots if s["task"] is not None and (kind is None or s["kind"] == kind)]

    def submit(self, s, task, limit):
        s["task"] = task; s["t0"] = time.perf_counter(); s["limit"] = float(limit)
        s["conn"].send(task)

    def poll(self, timeout=1.0):
        """Returns a list of (slot, task, kind, payload) with kind in {"result", "crash", "timeout", "memory"}."""
        busy = self.busy()
        if not busy:
            time.sleep(min(timeout, 0.2)); return []
        wait([s["conn"] for s in busy] + [s["proc"].sentinel for s in busy], timeout=timeout)
        now = time.perf_counter(); ev = []
        check_mem = self.rss_limit and now - self._last_mem > 5.0
        if check_mem:
            self._last_mem = now
        for s in busy:
            t, el = s["task"], now - s["t0"]
            got = None
            try:
                if s["conn"].poll():
                    got = s["conn"].recv()
            except (EOFError, OSError):
                got = None
            if got is not None:
                s["task"] = None; ev.append((s, t, "result", got))
            elif not s["proc"].is_alive():
                s["task"] = None
                ev.append((s, t, "crash", dict(exitcode=s["proc"].exitcode, wall=el)))
                self._retire(s, kill=False)
            elif el > s["limit"]:
                s["task"] = None
                ev.append((s, t, "timeout", dict(limit=s["limit"], wall=el)))
                self._retire(s, kill=True)
            elif check_mem and _rss(s["proc"].pid) > self.rss_limit:
                s["task"] = None
                ev.append((s, t, "memory", dict(rss=_rss(s["proc"].pid), limit=self.rss_limit, wall=el)))
                self._retire(s, kill=True)
        return ev

    def terminate(self):
        for s in self.slots:
            if s["proc"].is_alive():
                s["proc"].kill()

    def shutdown(self, grace=30):
        for s in self.slots:
            if s["task"] is None:
                try:
                    s["conn"].send(None)
                except OSError:
                    pass
        t0 = time.time()
        for s in self.slots:
            s["proc"].join(timeout=max(0.1, grace - (time.time() - t0)))
        self.terminate()


def _rss(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0
