"""Linux CPU certification campaign. One coordinator lock, isolated child per attempt, immutable receipts."""
import argparse
import ctypes
import fcntl
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import tempfile
from time import monotonic,process_time,sleep,time
from uuid import uuid4

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import inputs
import evidence
from cbc_run import warm_jit
from cbc_search import solve
from run import write_result

TOL=Fraction(1,20)


def durable(path,data):
    """Publish a complete new record, never overwrite an existing scientific identity."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        tmp=Path(f.name)
        try:
            f.write(evidence.canonical(data))
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.link(tmp,path)
        sync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def sync_dir(d):
    fd=os.open(d,os.O_DIRECTORY|os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def make_plan(scenes,build):
    tasks={}
    files={}
    for path in sorted(scenes.glob("*/*.npz")):
        data=path.read_bytes()
        with np.load(io.BytesIO(data),allow_pickle=False) as z:
            meta=json.loads(z["meta"].tobytes())
            slot=meta["slot"]
            windows=z["windows"]
        if slot in files or len(windows)!=12 or meta["contract"]!="rind3d-100k-contract-v3":
            raise ValueError("unexpected scene or window contract")
        files[slot]=dict(path=str(path.relative_to(scenes)),sha=evidence.digest(data))
        for w,row in enumerate(windows):
            L=int(row["L"])
            mask=int(row["mask"])
            if not 0<mask<1<<int(meta["k"]) or L!=(64,128,256)[w//4]:
                raise ValueError("invalid source mask/size")
            for j in range(int(meta["k"])):
                if mask&(1<<j):
                    key=f"{slot:06d}_w{w:02d}_s{j}"
                    obs,identity=inputs.load(io.BytesIO(data),w,j)
                    if identity["slot"]!=slot or identity["family"]!=meta["family"]:
                        raise ValueError("inconsistent scene identity")
                    tasks[key]=identity
    if not tasks: raise ValueError("empty catalog")
    return dict(schema="cbc-campaign/1",backend="python-integer-fraction",build=build,
                tolerance="1/20",files=files,tasks=tasks)


def artifact_status(path,rec,identity,build):
    raw=path.read_bytes()
    if evidence.digest(raw)!=rec["sha256"]: raise ValueError("issued artifact digest mismatch")
    data=json.loads(raw)
    if data["problem"]!=identity or data["status"]!=rec["status"] or data["tolerance"]!="1/20":
        raise ValueError("wrong artifact identity/status")
    if data["status"]=="CERTIFIED":
        evidence.check_artifact(data,identity,expected_build=build,materialize=False)
    elif data["status"]!="INCOMPLETE" or any(k in data for k in ("wire","certificate","wire_sha256","cells")):
        raise ValueError("invalid incomplete artifact")
    return data


def worker(folder,parent,blob,identity,seconds,cap,mem):
    # Child can write only its dedicated attempt in our program; it never publishes a receipt.
    if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=parent:
        os._exit(125)
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,signal.SIG_DFL)
    resource.setrlimit(resource.RLIMIT_AS,(mem<<20,mem<<20))
    t,c=monotonic(),process_time()
    try:
        obs,actual=inputs.load(io.BytesIO(blob),identity["window"],identity["channel"])
        if actual!=identity: raise ValueError("decoded input differs from frozen manifest")
        key=f"{actual['slot']:06d}_w{actual['window']:02d}_s{actual['channel']}"
        wire,d=solve(obs,key,TOL,seconds,cap)
        path=folder/"label.json"
        write_result(path,identity,TOL,wire,d)
        rec=dict(status=d["status"],tv=d["tv"],reason=d["search_reason"],sha256=evidence.digest(path.read_bytes()),
                 stats=d["stats"],support=0 if wire is None else len(wire)//12,
                 lower=d.get("lower_volume"),live=d.get("unknown_volume"))
        if wire:
            p=np.frombuffer(wire,dtype=[("id","<u4"),("value","<f8")])["value"]
            rec["entropy_nats"]=float(-np.dot(p,np.log(p)))
    except MemoryError:
        rec=dict(status="RESOURCE_INCOMPLETE",reason="memory cap")
    except Exception as e:
        rec=dict(status="ERROR",reason=f"{type(e).__name__}: {e}")
    rec.update(worker_wall=monotonic()-t,worker_cpu=process_time()-c)
    durable(folder/"worker.json",rec)
    os._exit(0)


def finalize(folder,identity,build,system):
    worker_file=folder/"worker.json"
    if worker_file.exists():
        rec=json.loads(worker_file.read_bytes())
        if rec["status"] in ("CERTIFIED","INCOMPLETE"):
            artifact_status(folder/"label.json",rec,identity,build)
        elif rec["status"] not in ("RESOURCE_INCOMPLETE","ERROR"):
            raise ValueError("unknown worker status")
    else:
        rec=dict(status="INTERRUPTED",reason=system.get("termination","worker left no durable completion"))
    rec.update(system)
    durable(folder/"receipt.json",rec)
    return rec


def run(args):
    root=args.root
    root.mkdir(parents=True,exist_ok=True)
    with (root/"owner.lock").open("ab") as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        build=json.loads(evidence._build_bytes())
        manifest=root/"manifest.json"
        if args.action=="init":
            if manifest.exists(): raise FileExistsError(manifest)
            durable(manifest,make_plan(args.scenes,build))
            print(manifest,flush=True)
            return
        manifest_bytes=manifest.read_bytes()
        if evidence.digest(manifest_bytes)!=args.manifest_sha: raise ValueError("manifest changed")
        plan=json.loads(manifest_bytes)
        if plan["build"]!=build: raise ValueError("different numerical build; use the frozen checkout")
        inputs_by_slot={}
        for slot,info in plan["files"].items():
            data=(args.scenes/info["path"]).read_bytes()
            if evidence.digest(data)!=info["sha"]: raise ValueError("input file hash mismatch")
            inputs_by_slot[int(slot)]=data
        if args.workers*args.mem_mib>args.pool_mib: raise ValueError("aggregate worker reservation exceeds pool")
        pending=[]
        for key,identity in plan["tasks"].items():
            task=root/"tasks"/key
            complete=False
            same=False
            tries=sorted(task.glob("*/start.json"))
            for start in tries:
                folder=start.parent
                request=json.loads(start.read_bytes())
                if request["key"]!=key or request["manifest_sha"]!=args.manifest_sha:
                    raise ValueError("conflicting attempt identity")
                receipt=folder/"receipt.json"
                if not receipt.exists():
                    finalize(folder,identity,build,dict(recovered=True,termination="previous coordinator interrupted"))
                rec=json.loads(receipt.read_bytes())
                if rec["status"]=="ERROR": raise ValueError(f"{key}: {rec['reason']}")
                if rec["status"] in ("CERTIFIED","INCOMPLETE"):
                    artifact_status(folder/"label.json",rec,identity,build)
                if complete: raise ValueError("attempt after immutable issued certificate")
                complete=rec["status"]=="CERTIFIED"
                same |= (rec["status"]=="INCOMPLETE" and request["seconds"]==args.seconds and request["cap"]==args.cap)
            if not complete and not same and len(tries)<args.max_tries:
                last=max((int(d.name) for d in task.glob("[0-9]*") if d.is_dir()),default=0)
                pending.append((key,identity,last+1))
        if args.action=="verify":
            print(f"verified receipts for manifest of {len(plan['tasks'])} channels")
            return
        if args.limit is not None: pending=pending[:args.limit]
        session=uuid4().hex
        start_t=monotonic()
        start_cpu=process_time()
        stop=False
        def pause(sig,frame):
            nonlocal stop
            stop=True
        for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,pause)
        warm=warm_jit()
        # Compile the read-only signatures once; search output is discarded and its cost charged.
        if pending:
            key,identity,_=pending[0]
            obs,_=inputs.load(io.BytesIO(inputs_by_slot[identity["slot"]]),identity["window"],identity["channel"])
            solve(obs,key,TOL,1,256)
            del obs
        parent=os.getpid()
        active={}
        killed=set()
        submitted=0
        committed=0
        failures=[]
        try:
            while active or (submitted<len(pending) and not stop):
                while len(active)<args.workers and submitted<len(pending) and not stop:
                    key,ident,attempt=pending[submitted]
                    folder=root/"tasks"/key/f"{attempt:03d}"
                    folder.mkdir(parents=True,exist_ok=False)
                    sync_dir(folder.parent); sync_dir(folder.parent.parent); sync_dir(root)
                    durable(folder/"start.json",dict(key=key,session=session,seconds=args.seconds,cap=args.cap,
                            manifest_sha=args.manifest_sha,mem_mib=args.mem_mib,started=time()))
                    blob=inputs_by_slot[ident["slot"]]
                    # Share canonical integer geometry read-only through fork; no visibility decision is shared.
                    with np.load(io.BytesIO(blob),allow_pickle=False) as z:
                        inputs._geometry_rows(z["geometry"].tobytes())
                    t=monotonic()
                    pid=os.fork()
                    if pid==0:
                        try: worker(folder,parent,blob,ident,args.seconds,args.cap,args.mem_mib)
                        finally: os._exit(124)
                    active[pid]=(folder,ident,t)
                    submitted+=1
                if not active: break
                for pid,(folder,ident,t) in list(active.items()):
                    done,status,usage=os.wait4(pid,os.WNOHANG)
                    if not done:
                        if monotonic()-t>args.watchdog:
                            os.kill(pid,signal.SIGKILL)
                            killed.add(pid)
                        continue
                    del active[pid]
                    system={"termination":"hard watchdog"} if pid in killed else {}
                    killed.discard(pid)
                    rec=finalize(folder,ident,build,dict(**system,exit=os.waitstatus_to_exitcode(status),
                        wall=monotonic()-t,cpu=usage.ru_utime+usage.ru_stime,rss_kib=usage.ru_maxrss))
                    committed+=1
                    if rec["status"]=="ERROR":
                        failures.append(rec["reason"])
                        stop=True
                if active: sleep(.035)
        finally:
            for pid in active:
                try: os.kill(pid,signal.SIGKILL)
                except ProcessLookupError: pass
            for pid in active: os.waitpid(pid,0)
            durable(root/f"session-{session}.json",dict(submitted=submitted,committed=committed,wall=monotonic()-start_t,
                    parent_cpu=process_time()-start_cpu,jit_warm=warm,paused=stop,errors=failures,workers=args.workers))
        if failures: raise RuntimeError(failures)
        print(f"{committed} attempts committed",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=("init","run","verify"))
    ap.add_argument("scenes",type=Path)
    ap.add_argument("root",type=Path)
    ap.add_argument("--seconds",type=int,default=10)
    ap.add_argument("--cap",type=int,default=4096)
    ap.add_argument("--workers",type=int,default=5)
    ap.add_argument("--max-tries",type=int,default=4)
    ap.add_argument("--mem-mib",type=int,default=4608)
    ap.add_argument("--pool-mib",type=int,default=23552)
    ap.add_argument("--watchdog",type=int,default=105)
    ap.add_argument("--limit",type=int)
    ap.add_argument("--manifest-sha",help="independently pinned campaign manifest SHA256, required after init")
    args=ap.parse_args()
    if min(args.seconds,args.workers,args.cap,args.mem_mib,args.max_tries,args.watchdog)<=0:
        ap.error("positive resource limits required")
    if args.action!="init" and evidence.digest((args.root/"manifest.json").read_bytes())!=args.manifest_sha:
        ap.error("pinned manifest hash required")
    run(args)
