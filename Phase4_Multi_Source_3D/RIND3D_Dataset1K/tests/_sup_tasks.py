"""Worker functions for test_supervise.py (imported inside spawned workers)."""
import os
import time


def job(t, engine, kw):
    if t == "abort":
        os.abort()
    if t == "segv":
        import ctypes
        ctypes.string_at(0)
    if t == "hang":
        time.sleep(1e9)
    return dict(task=t, engine=engine, kw=kw, pid=os.getpid())
