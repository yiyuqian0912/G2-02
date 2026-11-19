"""Blind, one-process research labeler. Multiple cases can share a warmed JIT."""
import argparse
import base64
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import tempfile
from time import perf_counter, process_time

import inputs
from kernel import Kernel
from ledger import decode
from solver import Budget, Limit, choose_pairs


def search(obs, key, tolerance, seconds, cap, pairs, strategy, order):
    if seconds<=0 or cap<1 or pairs<0 or not 0<=tolerance<=1:
        raise ValueError("invalid budget or tolerance")
    budget = Budget(seconds,cap)
    kernel = Kernel(obs,budget)
    timings = {}
    payload = bound = None
    parked = set()
    def checkpoint():
        start = perf_counter()
        result = kernel.snapshot()
        timings["bound_seconds"] = timings.get("bound_seconds",0) + perf_counter()-start
        return result
    reason = "unresolved_frontier"
    try:
        start = perf_counter()
        seed = int.from_bytes(sha256(key.encode()).digest()[:4],"big")
        edges = choose_pairs(obs.cls,pairs,seed,strategy) if pairs else []
        for dark,lit in edges:
            kernel.condition_pair(dark,lit)
        timings["discovery_seconds"] = perf_counter()-start
        kernel.exclude_window()
        while True:
            budget.check()
            work = [h for h in kernel.work_handles() if h not in parked]
            if not work:
                break
            handle = max(work,key=kernel.priority) if order=="largest" else work[0]
            if not kernel.advance(handle):
                parked.add(handle)
            if kernel.positive:
                payload,bound = checkpoint()
                if bound<=tolerance:
                    reason = "tv_target"
                    break
    except Limit as exc:
        reason = str(exc)
    # The frontier is owned by the kernel, including the piece whose transition
    # failed. This final bounded-size accounting pass may finish after the soft limit.
    if reason != "tv_target":
        payload,bound = checkpoint()
    accepted = bound is not None and bound<=tolerance
    return (payload if accepted else None),dict(
        status="CERTIFIED" if accepted else "INCOMPLETE", reason=reason,
        tv=None if bound is None else str(bound), live=len(kernel.handles()),
        positive=kernel.positive, stats=dict(kernel.stats), **timings)


def write_result(path, identity, tolerance, payload, details):
    result = dict(format="astra-cbc-research-1",problem=identity,tolerance=str(tolerance),**details)
    if payload is not None:
        result.update(cells=len(decode(payload)), wire=base64.b64encode(payload).decode("ascii"),
                      wire_sha256=sha256(payload).hexdigest())
    import evidence
    if details.get("solver") == evidence.VERSION:
        result["format"] = "astra-xlf-certified-interface/1"
        if result["status"] == "CERTIFIED":
            evidence.check_seal(result["certificate"], payload, result["tv"], tolerance, identity)
            result["artifact_sha256"] = evidence.artifact_digest(identity, result["certificate"])
    raw = json.dumps(result,sort_keys=True,allow_nan=False,separators=(",",":")).encode()
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        tmp = Path(f.name)
        try:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        rec = json.loads(tmp.read_bytes())
        if rec != result:
            raise ArithmeticError("serialized envelope changed")
        if rec["status"]=="CERTIFIED":
            restored = base64.b64decode(rec["wire"],validate=True)
            if rec["format"] == "astra-xlf-certified-interface/1":
                evidence.check_artifact(rec, identity)
            if rec.get("certificate") is not None:
                from evidence import check_seal
                check_seal(rec["certificate"], restored, rec["tv"], tolerance, rec["problem"])
            if restored!=payload or sha256(restored).hexdigest()!=rec["wire_sha256"]:
                raise ArithmeticError("wire mismatch")
            if not 0<=Fraction(rec["tv"])<=tolerance or len(decode(restored))!=rec["cells"]:
                raise ArithmeticError("invalid release")
        elif "wire" in rec:
            raise ArithmeticError("incomplete label cannot carry a distribution")
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("container",type=Path)
    parser.add_argument("window",type=int)
    parser.add_argument("channel",type=int)
    parser.add_argument("output",type=Path)
    parser.add_argument("--tolerance",type=Fraction,default=Fraction(1,20))
    parser.add_argument("--seconds",type=float,default=8)
    parser.add_argument("--cap",type=int,default=192)
    parser.add_argument("--pairs",type=int,default=24)
    parser.add_argument("--strategy",choices=("interface","random_classes"),default="interface")
    parser.add_argument("--order",choices=("largest","fifo"),default="largest")
    args=parser.parse_args()
    t,c=perf_counter(),process_time()
    obs,identity=inputs.load(args.container,args.window,args.channel)
    key=f"{identity['slot']:06d}_w{args.window:02d}_s{args.channel}"
    payload,details=search(obs,key,args.tolerance,args.seconds,args.cap,args.pairs,args.strategy,args.order)
    write_result(args.output,identity,args.tolerance,payload,details)
    print(json.dumps(dict(key=key,wall=perf_counter()-t,cpu=process_time()-c,
                          peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**details)))


if __name__=="__main__":
    main()
