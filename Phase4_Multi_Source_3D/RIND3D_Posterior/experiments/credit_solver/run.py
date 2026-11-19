"""Historical search for diagnostic callers; both CLI aliases invoke the maintained CBC solver."""
import base64
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

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
    import evidence
    if details.get("solver") != evidence.VERSION or details.get("status") not in ("CERTIFIED","INCOMPLETE"):
        raise ValueError("expected a current solver result")
    result = dict(format="astra-xlf-certified-interface/1", problem=identity, tolerance=str(tolerance), **details)
    if payload is not None:
        result.update(cells=len(decode(payload)), wire=base64.b64encode(payload).decode("ascii"),
                      wire_sha256=sha256(payload).hexdigest())
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
            evidence.check_artifact(rec, identity)
            if restored != payload:
                raise ArithmeticError("wire mismatch")
        elif "wire" in rec:
            raise ArithmeticError("incomplete label cannot carry a distribution")
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)



if __name__ == "__main__":
    from cbc_run import main
    main()
