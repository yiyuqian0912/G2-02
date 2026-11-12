"""Single-node coordinator for RIND3D-100K: physical scenes and labels on N GPUs plus M CPU workers.

**Design:**
* One coordinator process owns every append-only record:
  - `manifest/physical.jsonl`;
  - the label journal and its index;
  - `labels/failures.jsonl`;
  - `logs/progress.jsonl`.
  Workers write only their own scene container, atomically.
* Scenes are independent tasks. Within an epoch of `epoch_scenes` slots, scenes run in any order on any GPU. An epoch
  is planned (`plans/epoch_e.json`) only after every slot of epochs <= e - lag is final, so plans are deterministic.
* Labels are independent (scene, window, source) tasks. They are queued as soon as their scene is accepted:
  - L256 tasks go to GPU-assisted workers first;
  - L64 and L128 tasks go to CPU workers first;
  - any idle worker takes any task.
* Supervision (`pool.WorkerPool`): each task has a wall-clock limit and each worker an RSS limit. A crash, timeout or
  memory overrun is a failure event:
  - **physical:** the same attempt is retried (up to `phys_retries`), so a transient fault never changes the sampling
    law;
  - **labels:** the task is quarantined, i.e. recorded in failures.jsonl and visible in coverage, and is not retried
    in this run (use `--retry-failed`).
  No label is ever synthesised.
* `STOP` file or SIGTERM: no new tasks; in-flight tasks get `stop_grace` seconds; then the coordinator exits cleanly.
  Rerunning the same command resumes from the manifest and journals.
"""
import json
import multiprocessing as mp
import os
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from .controller import plan_epoch, epoch_of
from .pool import WorkerPool, _rss
from .storage import (Journal, journal_index, read_record, decode_payload, labels_path, task_key, atomic_write)

HERE = Path(__file__).resolve().parent.parent


def _jsonl(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def detect_resources():
    cpus = len(os.sched_getaffinity(0))
    quota = None
    try:
        q = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q[0] != "max":
            quota = int(q[0]) / int(q[1])
    except Exception:
        pass
    mem = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            mem = int(line.split()[1]) * 1024
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        gpus = [dict(index=int(a), name=b.strip(), mem_mb=int(c)) for a, b, c in (l.split(",") for l in out if l.strip())]
    except Exception:
        gpus = []
    return dict(cpus=cpus, cgroup_cpus=quota, mem_available=mem, gpus=gpus, host=socket.gethostname())


class Coordinator:
    def __init__(self, root, contract, opts):
        self.root = Path(root); self.cfg = contract; self.o = opts
        for d in ("manifest", "plans", "scenes", "labels/journal", "labels/scenes", "logs"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        cp = self.root / "CONTRACT.json"
        if cp.exists():
            old = json.loads(cp.read_text())
            if old["contract"] != contract["contract"] or old["seed"] != contract["seed"]:
                raise SystemExit("root holds a different contract")
        else:
            atomic_write(cp, json.dumps(contract, indent=1).encode())
        self.a, self.b = opts["slots"]
        self.man_path = self.root / "manifest" / "physical.jsonl"
        self.recs = {}
        for r in _jsonl(self.man_path):
            self.recs.setdefault(int(r["slot"]), []).append(r)
        self.fail_path = self.root / "labels" / "failures.jsonl"
        self.failed = {r["key"]: r for r in _jsonl(self.fail_path)}
        self.jidx = journal_index(self.root)
        self.done_keys = set(self.jidx)
        self.plans = {}
        self.stop = False; self.stop_t = None
        self.t_start = time.time(); self.n_scene = 0; self.n_label = 0; self.n_fail = 0
        self.scene_tasks = {}            # slot -> set(keys) still open in this run
        self.lq_heavy = deque(); self.lq_light = deque()
        self.queued = set()

    # ------------------------------------------------------------------ physical bookkeeping
    def final(self, slot):
        rs = self.recs.get(slot, [])
        return any(r.get("final") for r in rs)

    def accepted(self, slot):
        rs = [r for r in self.recs.get(slot, []) if r.get("status") == "accepted"]
        return rs[-1] if rs else None

    def next_attempt(self, slot):
        rs = self.recs.get(slot, [])
        if not rs:
            return 0, 0
        last = rs[-1]
        if last.get("status") == "failed" or last.get("kind") in ("crash", "timeout", "memory"):
            tries = sum(1 for r in rs if r["attempt"] == last["attempt"] and r.get("status") != "rejected")
            return last["attempt"], tries
        return last["attempt"] + 1, 0

    def plan(self, epoch):
        if epoch in self.plans:
            return self.plans[epoch]
        p = self.root / "plans" / f"epoch_{epoch:05d}.json"
        if p.exists():
            self.plans[epoch] = {int(k): v for k, v in json.loads(p.read_text()).items()}
            return self.plans[epoch]
        lag = int(self.cfg["controller"].get("plan_lag_epochs", 1)); E = int(self.cfg["controller"]["epoch_scenes"])
        lo = max(self.a, 0)
        for e in range(0, epoch - lag + 1):
            for slot in range(max(e * E, lo), min((e + 1) * E, self.b)):
                if not self.final(slot):
                    return None          # an earlier epoch is still running: barrier
        acc = [self.accepted(s) for s in self.recs if self.accepted(s)]
        plans = plan_epoch(self.cfg, epoch, acc, self.o.get("prior_manifest"))
        atomic_write(p, json.dumps({str(k): v for k, v in plans.items()}).encode())
        self.plans[epoch] = plans
        return plans

    def physical_queue(self):
        """Pending (slot, attempt) of the earliest epochs whose plan is available (bounded look-ahead)."""
        E = int(self.cfg["controller"]["epoch_scenes"]); out = []
        for epoch in range(epoch_of(self.a, self.cfg), epoch_of(self.b - 1, self.cfg) + 1):
            slots = [s for s in range(max(epoch * E, self.a), min((epoch + 1) * E, self.b)) if not self.final(s)]
            if not slots:
                continue
            pl = self.plan(epoch)
            if pl is None:
                break
            for s in slots:
                att, tries = self.next_attempt(s)
                out.append((s, att, tries, pl[s]))
            break                        # one epoch at a time (the next one is planned when this one is final)
        return out

    # ------------------------------------------------------------------ labels bookkeeping
    def enqueue_labels(self, rec):
        slot = int(rec["slot"])
        if not self.o.get("labels", True):
            return
        ls = self.o.get("label_slots")
        if ls and not (ls[0] <= slot < ls[1]):
            return
        keys = set()
        W = rec.get("windows") or []
        for w, (L, mask, _m) in enumerate(W):
            for j in range(int(rec["k"])):
                if not (int(mask) >> j) & 1:
                    continue
                key = task_key(slot, w, j)
                if key in self.done_keys or key in self.queued:
                    continue
                if key in self.failed and not self.o.get("retry_failed"):
                    continue
                t = (str(self.root), slot, w, j, int(L), True)
                (self.lq_heavy if int(L) >= 256 else self.lq_light).append(t)
                self.queued.add(key); keys.add(key)
        if keys:
            self.scene_tasks[slot] = keys
        else:
            self.consolidate(slot, rec)

    def consolidate(self, slot, rec=None):
        """Write labels/scenes/<shard>/<slot>.labels.npz once every informative task of the scene is resolved."""
        rec = rec or self.accepted(slot)
        if rec is None:
            return
        W = rec.get("windows") or []
        have = {}
        for w, (L, mask, _m) in enumerate(W):
            for j in range(int(rec["k"])):
                key = task_key(slot, w, j)
                if key in self.jidx:
                    have[key] = read_record(self.jidx[key])[1]
        cells, los, his, kinds, offs, metas, status = [], [], [], [], [0], [], {}
        for w, (L, mask, _m) in enumerate(W):
            for j in range(int(rec["k"])):
                key = task_key(slot, w, j)
                if not (int(mask) >> j) & 1:
                    status[key] = "not_computed"; continue
                if key in have:
                    d = decode_payload(have[key])
                    cells.append(d["cell"]); los.append(d["lo"]); his.append(d["hi"]); kinds.append(d["kind"])
                    offs.append(offs[-1] + len(d["cell"])); metas.append(d["meta"]); status[key] = d["meta"]["status"]
                elif key in self.failed:
                    status[key] = {"timeout": "timeout", "memory": "memory_limit"}.get(self.failed[key]["kind"], "engine_failure")
                else:
                    status[key] = "unavailable"
        import io
        buf = io.BytesIO()
        np.savez(buf, cell=np.concatenate(cells) if cells else np.zeros(0, np.int32),
                 lo=np.concatenate(los) if los else np.zeros(0), hi=np.concatenate(his) if his else np.zeros(0),
                 kind=np.concatenate(kinds) if kinds else np.zeros(0, np.int8), offsets=np.array(offs, np.int64),
                 metas=np.frombuffer(json.dumps(metas).encode(), np.uint8),
                 status=np.frombuffer(json.dumps(status).encode(), np.uint8))
        atomic_write(labels_path(self.root, slot), buf.getvalue())
        self.scene_tasks.pop(slot, None)

    # ------------------------------------------------------------------ main loop
    def run(self):
        o = self.o; cfg = self.cfg
        res = detect_resources()
        gpus = o["gpus"] if o.get("gpus") is not None else [g["index"] for g in res["gpus"]]
        nphys = int(o.get("phys_per_gpu", 1)) * len(gpus) if o.get("physical", True) else 0
        nlg = int(o.get("label_gpu_per_gpu", 2)) * len(gpus) if o.get("labels", True) else 0
        cpu_budget = int(res["cgroup_cpus"] or res["cpus"])
        if o.get("label_cpu", "auto") == "auto":
            nlc = max(0, cpu_budget - nphys - nlg - int(o.get("reserve_cpus", 2)))
            if o.get("rss_gb"):
                nlc = min(nlc, max(0, int(res["mem_available"] / (float(o["rss_gb"]) * 2 ** 30)) - nphys - nlg))
        else:
            nlc = int(o["label_cpu"])
        if not o.get("labels", True):
            nlc = 0
        base_env = dict(OMP_NUM_THREADS=1, NUMBA_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
                        PYTHONDONTWRITEBYTECODE=1)
        for k in ("NUMBA_CACHE_DIR", "RIND_NVRTC_LIBRARY", "CUPY_CACHE_DIR"):
            if os.environ.get(k):
                base_env[k] = os.environ[k]
        t1 = cfg["labels"]["tier1"]
        lkw = dict(tier=1, tier1={kk: v for kk, v in t1.items() if kk != "gpu_cap"},
                   tier2={kk: v for kk, v in cfg["labels"]["tier2"].items() if kk != "gpu_cap"},
                   gpu={"gpu_cap": int(o.get("gpu_cap") or t1.get("gpu_cap", 1 << 22))})
        specs = []
        for i in range(nphys):
            g = gpus[i % len(gpus)]
            specs.append(dict(name=f"phys{i}", kind="phys", fn="rind3d_100k.worker:physical_task", engine="gpu",
                              kw=cfg, env=dict(base_env, CUDA_VISIBLE_DEVICES=g)))
        for i in range(nlg):
            g = gpus[i % len(gpus)]
            specs.append(dict(name=f"lgpu{i}", kind="label", gpu=True, fn="rind3d_100k.labels:label_task", engine="gpu",
                              kw=lkw, env=dict(base_env, CUDA_VISIBLE_DEVICES=g)))
        for i in range(nlc):
            specs.append(dict(name=f"lcpu{i}", kind="label", gpu=False, fn="rind3d_100k.labels:label_task", engine="cpu",
                              kw=lkw, env=dict(base_env, CUDA_VISIBLE_DEVICES="")))
        layout = dict(resources=res, gpus=gpus, physical_workers=nphys, label_gpu_workers=nlg, label_cpu_workers=nlc,
                      slots=[self.a, self.b], options={k: v for k, v in o.items() if k != "prior_manifest"})
        self.log(dict(event="start", **layout))
        sys_path = [str(HERE), str(HERE.parent / "RIND3D_Dataset1K"), str(HERE.parent / "RIND3D_Posterior" / "src")]
        pool = WorkerPool(mp.get_context("spawn"), specs, sys_path=sys_path, nice=int(o.get("nice", 0)),
                          rss_limit_bytes=float(o["rss_gb"]) * 2 ** 30 if o.get("rss_gb") else None)
        journal = Journal(self.root, f"{socket.gethostname()}-{os.getpid()}-{int(time.time())}")

        def _sig(*_):
            self.stop = True; self.stop_t = time.time() - float(o.get("stop_grace", 600))
        signal.signal(signal.SIGTERM, _sig)
        # resume: queue labels of accepted scenes
        for slot in sorted(self.recs):
            if self.a <= slot < self.b:
                r = self.accepted(slot)
                if r is not None:
                    self.enqueue_labels(r)
        inflight_phys = set(); last_prog = 0.0; deadline = time.time() + float(o.get("max_hours", 1e9)) * 3600
        phys_lim = float(o.get("phys_timeout", 1800)); lim = {int(k): float(v) for k, v in cfg["labels"]["timeouts_s"].items()}
        try:
            while True:
                if (self.root / "STOP").exists() or time.time() > deadline:
                    if not self.stop:
                        self.stop = True; self.stop_t = time.time()
                        self.log(dict(event="stop_requested"))
                if not self.stop:
                    # physical
                    idle = pool.idle("phys") if o.get("physical", True) else []
                    if idle:
                        for (s, att, tries, pl) in self.physical_queue():
                            if not idle:
                                break
                            if s in inflight_phys:
                                continue
                            if tries > int(o.get("phys_retries", 2)):
                                self.record(dict(slot=s, attempt=att, status="failed", final=True, reason="retries_exhausted"))
                                continue
                            pool.submit(idle.pop(0), dict(root=str(self.root), slot=s, attempt=att, k=pl["k"],
                                                          c_table=pl["c_table"], plan=pl["plan"]), phys_lim)
                            inflight_phys.add(s)
                    # labels
                    for sl in pool.idle("label"):
                        q1, q2 = (self.lq_heavy, self.lq_light) if sl.get("gpu") else (self.lq_light, self.lq_heavy)
                        q = q1 if q1 else q2
                        if not q:
                            break
                        t = q.popleft()
                        pool.submit(sl, t, lim.get(t[4], 7200))
                if self.stop and (not pool.busy() or time.time() - self.stop_t > float(o.get("stop_grace", 600))):
                    break
                if not self.stop and not pool.busy() and not self.lq_heavy and not self.lq_light and \
                        (not o.get("physical", True) or all(self.final(s) for s in range(self.a, self.b))):
                    break
                for (sl, t, kind, payload) in pool.poll(1.0):
                    if sl["kind"] == "phys":
                        inflight_phys.discard(int(t["slot"]))
                        if kind == "result":
                            rec = payload
                        else:
                            rec = dict(slot=int(t["slot"]), attempt=int(t["attempt"]), status="failed", kind=kind, **payload)
                        rec.update(k_chosen=t["k"], c_table=t["c_table"], controller_plan=t["plan"])
                        if rec["status"] == "accepted":
                            rec["final"] = True
                        elif rec["status"] == "rejected":
                            rec["final"] = int(rec["attempt"]) + 1 >= int(cfg["max_attempts"])
                        else:
                            rec["final"] = False
                        self.record(rec)
                        if rec["status"] == "accepted":
                            self.n_scene += 1; self.enqueue_labels(rec)
                    else:
                        key = task_key(t[1], t[2], t[3])
                        self.queued.discard(key)
                        if kind == "result" and "arrays" in payload:
                            self.jidx[key] = journal.append(key, payload["arrays"], payload["meta"])
                            self.done_keys.add(key); self.n_label += 1
                        else:
                            fr = dict(key=key, slot=t[1], window=t[2], source=t[3], L=t[4], kind=kind if kind != "result" else "error",
                                      worker=sl["name"], engine=sl["engine"], time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                      **({kk: v for kk, v in payload.items() if kk in ("error", "trace", "exitcode", "wall", "limit", "rss")}))
                            with open(self.fail_path, "a") as f:
                                f.write(json.dumps(fr) + "\n")
                            self.failed[key] = fr; self.n_fail += 1
                        open_ = self.scene_tasks.get(t[1])
                        if open_ is not None:
                            open_.discard(key)
                            if not open_:
                                self.consolidate(t[1])
                if time.time() - last_prog > float(o.get("progress_s", 30)):
                    last_prog = time.time(); self.progress(pool)
        finally:
            self.progress(pool, final=True)
            journal.close()
            pool.shutdown(grace=5)
        return 0

    def record(self, rec):
        rec.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(self.man_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self.recs.setdefault(int(rec["slot"]), []).append(rec)

    def log(self, d):
        d = dict(d, t=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(self.root / "logs" / "progress.jsonl", "a") as f:
            f.write(json.dumps(d, default=str) + "\n")
        print(json.dumps(d, default=str), flush=True)

    def progress(self, pool, final=False):
        el = max(time.time() - self.t_start, 1e-9)
        fin = sum(1 for s in range(self.a, self.b) if self.final(s))
        acc = sum(1 for s in range(self.a, self.b) if self.accepted(s))
        rss = sum(_rss(s["proc"].pid) for s in pool.slots)
        self.log(dict(event="final" if final else "progress", elapsed_s=round(el, 1), scenes_final=fin, scenes_accepted=acc,
                      scenes_this_run=self.n_scene, labels_this_run=self.n_label, label_failures_this_run=self.n_fail,
                      scenes_per_hour=round(3600 * self.n_scene / el, 2), labels_per_s=round(self.n_label / el, 3),
                      queue_heavy=len(self.lq_heavy), queue_light=len(self.lq_light),
                      busy_phys=len(pool.busy("phys")), busy_label=len(pool.busy("label")), workers_rss_gb=round(rss / 2 ** 30, 2)))
