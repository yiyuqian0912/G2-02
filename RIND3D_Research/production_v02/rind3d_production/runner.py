"""Idempotent, resumable production runner with shard publication.

  run.sh rind3d_production.runner create ROOT CONFIG     # freeze config + contract into ROOT (refuses overwrite)
  run.sh rind3d_production.runner run ROOT [--max-new N] [--max-runtime S]
  run.sh rind3d_production.runner status ROOT

Layout under ROOT:
  config.json, contract.json            frozen at create; every later invocation verifies their SHA256
  journal.jsonl                          append-only, fsynced (runner and worker events)
  staging/shard-NNNNN/<slot>-<attempt>/  in-progress scene (deleted and redone on resume; never published)
  staging/shard-NNNNN/scenes/<scene_id>/ sealed scenes (SEAL.json; SHA256 of every file re-checked on resume)
  staging/shard-NNNNN/slots/<slot>.json  slot outcome (sealed scene id, or failure with reason)
  shards/shard-NNNNN/                    published shard (atomic rename; immutable): scenes/, index/, MANIFEST.json
Failure isolation: a worker subprocess handles the pending slots of one shard; a worker death or a per-slot
timeout costs that slot one attempt; a slot is retried at most `max_attempts` times with a new attempt number
(new geometry draw only for world-fitting/source-budget failures, recorded) and is otherwise recorded FAILED.
Slots are never re-assigned to different geometry to improve yield; FAILED slots stay in the shard record.
"""
import argparse
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .env import ROOT as PKG_ROOT, certificate, sha256_file, sha256_json

PY = sys.executable


def journal(root, event):
    event = dict(event, time=time.strftime("%Y-%m-%dT%H:%M:%S"), pid=os.getpid())
    with open(Path(root)/"journal.jsonl", "a") as f:
        f.write(json.dumps(event, default=float)+"\n")
        f.flush()
        os.fsync(f.fileno())


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(obj, indent=1, default=float)+"\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_cfg(root):
    root = Path(root)
    cfg = json.loads((root/"config.json").read_text())
    frozen = json.loads((root/"FROZEN.json").read_text())
    if sha256_file(root/"config.json") != frozen["config_SHA256"] or sha256_file(root/"contract.json") != frozen["contract_SHA256"]:
        raise SystemExit("config/contract changed after freeze: refusing to run (create a new versioned campaign)")
    if "seed_file" in cfg:                       # private master seed (added after the canary; production releases)
        path = Path(cfg["seed_file"])
        if sha256_file(path) != frozen.get("seed_file_SHA256"):
            raise SystemExit("private seed file changed or missing")
        value = int(path.read_text().strip(), 16)
        if value.bit_length() < 200:
            raise SystemExit("private master seed must be a 256-bit hexadecimal value")
        cfg["seed"] = value
    return cfg


def create(root, config):
    root = Path(root)
    if (root/"FROZEN.json").exists():
        raise SystemExit("campaign already exists")
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config, root/"config.json")
    shutil.copy2(PKG_ROOT/"RIND3D_PRODUCTION_CONTRACT.json", root/"contract.json")
    cert = certificate()
    if not cert["matches_current_build"]:
        raise SystemExit("producer build does not match its scientific certification")
    cfgd = json.loads((root/"config.json").read_text())
    seed_sha = sha256_file(cfgd["seed_file"]) if "seed_file" in cfgd else None
    atomic(root/"FROZEN.json", {"config_SHA256": sha256_file(root/"config.json"), "contract_SHA256": sha256_file(root/"contract.json"),
                                "seed_file_SHA256": seed_sha,
                                "certificate": cert, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "code_SHA256": {str(p.relative_to(PKG_ROOT)): sha256_file(p) for p in sorted((PKG_ROOT/"rind3d_production").glob("*.py"))}})
    journal(root, {"event": "created"})


def shard_of(cfg, slot):
    return slot // cfg["shard_size"]


def shard_dir(root, s, published=False):
    return Path(root)/("shards" if published else "staging")/f"shard-{s:05d}"


def verify_sealed(scene_dir):
    seal = json.loads((scene_dir/"SEAL.json").read_text())
    for rel, v in seal["files"].items():
        p = scene_dir/rel
        if not p.is_file() or p.stat().st_size != v["bytes"] or sha256_file(p) != v["SHA256"]:
            return False
    return True


def slot_state(root, cfg, slot):
    s = shard_of(cfg, slot)
    if shard_dir(root, s, True).exists():
        return "published"
    p = shard_dir(root, s)/"slots"/f"{slot}.json"
    if p.exists():
        rec = json.loads(p.read_text())
        if rec["status"] == "SEALED":
            if verify_sealed(shard_dir(root, s)/"scenes"/rec["scene_id"]):
                return "sealed"
            raise SystemExit(f"sealed scene of slot {slot} failed SHA verification: preserved for diagnosis, refusing to continue")
        return rec["status"].lower()
    return "pending"


# ---------------------------------------------------------------- worker (one subprocess per shard batch)
def worker(root, shard, slots):
    from rind3d.rendering.backends import BackendFactory
    from .production import run_scene
    cfg = load_cfg(root)
    sdir = shard_dir(root, shard)
    (sdir/"scenes").mkdir(parents=True, exist_ok=True)
    (sdir/"slots").mkdir(parents=True, exist_ok=True)
    with BackendFactory(geometry_cache=1, cuda_batch=32) as factory:
        for slot in slots:
            attempts = [json.loads(l) for l in (Path(root)/"journal.jsonl").read_text().splitlines()
                        if f'"slot": {slot},' in l and '"event": "slot_attempt_failed"' in l]
            attempt = len(attempts)
            journal(root, {"event": "slot_started", "slot": slot, "attempt": attempt, "shard": shard})
            t0 = time.perf_counter()
            try:
                stage, rec = run_scene(cfg, slot, sdir, factory, attempt)
                target = sdir/"scenes"/rec["scene_id"]
                if target.exists():
                    if verify_sealed(target):
                        shutil.rmtree(stage)                     # identical deterministic re-run of a sealed scene
                    else:
                        raise RuntimeError("conflicting existing scene directory")
                else:
                    os.replace(stage, target)
                atomic(sdir/"slots"/f"{slot}.json", {"slot": slot, "status": "SEALED", "scene_id": rec["scene_id"], "attempt": attempt,
                                                    "record": {k: v for k, v in rec.items() if k != "files"}})
                journal(root, {"event": "slot_sealed", "slot": slot, "attempt": attempt, "scene_id": rec["scene_id"],
                               "wall_seconds": rec["wall_seconds"], "profile": rec["profile"], "K": rec["K"]})
                print(f"slot {slot} sealed {rec['profile']} K={rec['K']} {rec['family']} {rec['wall_seconds']:.1f}s", flush=True)
            except Exception as e:                                # noqa: BLE001 - recorded, never hidden
                reason = f"{type(e).__name__}: {e}"
                journal(root, {"event": "slot_attempt_failed", "slot": slot, "attempt": attempt, "reason": reason[:500],
                               "wall_seconds": time.perf_counter()-t0})
                partial = sdir/f"{slot:06d}-{attempt}"
                if partial.exists():                             # kept for diagnosis, outside every shard
                    dest = Path(root)/"failed-attempts"/f"{slot:06d}-{attempt}"
                    dest.parent.mkdir(exist_ok=True)
                    shutil.rmtree(dest, ignore_errors=True)
                    os.replace(partial, dest)
                if attempt+1 >= cfg["max_attempts"]:
                    atomic(sdir/"slots"/f"{slot}.json", {"slot": slot, "status": "FAILED", "reason": reason[:2000], "attempts": attempt+1})
                print(f"slot {slot} attempt {attempt} failed: {reason[:200]}", flush=True)


# ---------------------------------------------------------------- shard publication
def publish(root, cfg, shard):
    sdir = shard_dir(root, shard)
    slots = range(shard*cfg["shard_size"], min((shard+1)*cfg["shard_size"], cfg["slots"]))
    recs = [json.loads((sdir/"slots"/f"{s}.json").read_text()) for s in slots]
    sealed = [r for r in recs if r["status"] == "SEALED"]
    (sdir/"index").mkdir(exist_ok=True)
    rows, wins = [], []
    for i, r in enumerate(sealed):
        scene = sdir/"scenes"/r["scene_id"]
        pub = json.loads((scene/"public"/"scene.json").read_text())
        rows.append({"ordinal": i, "slot": r["slot"], "scene_id": r["scene_id"], "geometry_id": pub["geometry_id"],
                     "split": pub["split"], "profile": pub["profile"], "family": pub["family"], "windows": pub["windows"]})
        w = np.load(scene/"public"/"windows.npy")
        out = np.zeros(len(w), dtype=[("scene_ordinal", "<u4"), ("window", "<u4"), ("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("L", "<u2")])
        out["scene_ordinal"] = i
        for k in ("window", "x", "y", "z", "L"):
            out[k] = w[k]
        wins.append(out)
    np.save(sdir/"index"/"windows.npy", np.concatenate(wins) if wins else np.zeros(0))
    with open(sdir/"index"/"scenes.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r)+"\n")
    atomic(sdir/"index"/"slots.json", {"slots": recs})
    files = {str(p.relative_to(sdir)): {"bytes": p.stat().st_size, "SHA256": sha256_file(p)}
             for p in sorted(sdir.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}
    atomic(sdir/"MANIFEST.json", {"shard": shard, "contract": json.loads((Path(root)/"contract.json").read_text())["contract_id"],
                                  "scenes": len(sealed), "failed_slots": [r["slot"] for r in recs if r["status"] != "SEALED"],
                                  "files": files, "bytes": int(sum(v["bytes"] for v in files.values()))})
    for leftover in sdir.iterdir():
        if leftover.is_dir() and leftover.name not in ("scenes", "slots", "index"):
            raise RuntimeError(f"unpublishable leftover in shard staging: {leftover}")
    target = shard_dir(root, shard, True)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(sdir, target)
    journal(root, {"event": "shard_published", "shard": shard, "scenes": len(sealed), "manifest_SHA256": sha256_file(target/"MANIFEST.json")})
    print(f"shard {shard} published: {len(sealed)} scenes", flush=True)


# ---------------------------------------------------------------- GPU sampler
class GpuSampler(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path, self.stop = path, False

    def run(self):
        while not self.stop:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout.strip()
                with open(self.path, "a") as f:
                    f.write(f"{time.time():.1f},{out}\n")
            except Exception:                                    # noqa: BLE001
                pass
            time.sleep(5)


def run(root, max_new=None, max_runtime=None):
    root = Path(root)
    cfg = load_cfg(root)
    started = time.perf_counter()
    journal(root, {"event": "run_started", "max_new": max_new, "max_runtime": max_runtime})
    gpu = GpuSampler(root/"gpu-samples.csv")
    gpu.start()
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    new = 0
    n_shards = (cfg["slots"] + cfg["shard_size"] - 1)//cfg["shard_size"]
    try:
        for shard in range(n_shards):
            if shard_dir(root, shard, True).exists():
                continue
            sdir = shard_dir(root, shard)
            if sdir.exists():                                    # resume: move partial attempts aside, keep sealed scenes
                for p in sdir.iterdir():
                    if p.is_dir() and p.name not in ("scenes", "slots", "index"):
                        dest = root/"failed-attempts"/("interrupted-"+p.name)
                        dest.parent.mkdir(exist_ok=True)
                        shutil.rmtree(dest, ignore_errors=True)
                        os.replace(p, dest)
                        journal(root, {"event": "partial_attempt_moved_aside", "path": str(dest.relative_to(root))})
            slots = range(shard*cfg["shard_size"], min((shard+1)*cfg["shard_size"], cfg["slots"]))
            while True:
                pending = [s for s in slots if slot_state(root, cfg, s) == "pending"]
                if not pending:
                    break
                if stop["flag"] or (max_runtime and time.perf_counter()-started > max_runtime) or (max_new is not None and new >= max_new):
                    journal(root, {"event": "run_stopped", "reason": "signal/cap", "new_scenes": new})
                    return
                batch = pending[:max(1, (max_new - new)) if max_new is not None else len(pending)]
                cmd = [PY, "-m", "rind3d_production.runner", "worker", str(root), str(shard), *map(str, batch)]
                proc = subprocess.Popen(cmd, env=os.environ.copy())
                deadline = time.perf_counter() + cfg["caps"]["scene_seconds"]*len(batch) + 60
                last, slot_started = None, time.perf_counter()
                while proc.poll() is None:
                    time.sleep(1)
                    cur = [s for s in batch if slot_state(root, cfg, s) == "pending"]
                    if cur and cur[0] != last:
                        last, slot_started = cur[0], time.perf_counter()
                    if cur and time.perf_counter()-slot_started > cfg["caps"]["scene_seconds"]:
                        proc.kill()
                        journal(root, {"event": "slot_attempt_failed", "slot": cur[0], "reason": "timeout (per-scene wall cap)",
                                       "attempt": None})
                        break
                    if stop["flag"] or time.perf_counter() > deadline:
                        proc.terminate()
                        proc.wait(30)
                        break
                proc.wait()
                done = [s for s in batch if slot_state(root, cfg, s) == "sealed"]
                new += len([s for s in done if s in batch])
                if proc.returncode not in (0, None):
                    still = [s for s in batch if slot_state(root, cfg, s) == "pending"]
                    if still:
                        s0 = still[0]
                        fails = sum(1 for l in (root/"journal.jsonl").read_text().splitlines()
                                    if f'"slot": {s0},' in l and "slot_attempt_failed" in l)
                        if not any(f'"slot": {s0},' in l and "slot_attempt_failed" in l and "timeout" in l
                                   for l in (root/"journal.jsonl").read_text().splitlines()[-3:]):
                            journal(root, {"event": "slot_attempt_failed", "slot": s0, "reason": f"worker died (returncode {proc.returncode})",
                                           "attempt": None})
                            fails += 1
                        if fails >= cfg["max_attempts"]:
                            atomic(sdir/"slots"/f"{s0}.json", {"slot": s0, "status": "FAILED", "reason": "worker death/timeout", "attempts": fails})
                if stop["flag"]:
                    return
            publish(root, cfg, shard)
    finally:
        gpu.stop = True
        ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        journal(root, {"event": "run_finished", "new_scenes": new, "wall_seconds": time.perf_counter()-started,
                       "children_cpu_seconds": ru.ru_utime+ru.ru_stime, "children_max_RSS_MiB": ru.ru_maxrss/1024})


def status(root):
    root = Path(root)
    cfg = load_cfg(root)
    states = {}
    for s in range(cfg["slots"]):
        states.setdefault(slot_state(root, cfg, s), []).append(s)
    print(json.dumps({k: len(v) for k, v in states.items()}), json.dumps({k: v[:10] for k, v in states.items() if k != "published"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("root")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--max-runtime", type=float, default=None)
    a = ap.parse_args()
    if a.cmd == "create":
        create(a.root, a.rest[0])
    elif a.cmd == "run":
        run(a.root, a.max_new, a.max_runtime)
    elif a.cmd == "status":
        status(a.root)
    elif a.cmd == "worker":
        worker(a.root, int(a.rest[0]), [int(x) for x in a.rest[1:]])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
