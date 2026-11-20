"""Real single-channel recovery on a disposable child campaign; never signals an unrelated process."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from campaign import durable
import evidence


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("frozen_driver","scenes","full","child"):
        ap.add_argument(name,type=Path)
    a=ap.parse_args()
    key="000001_w07_s0"
    plan=json.loads((a.full/"manifest.json").read_bytes())
    plan["tasks"]={key:plan["tasks"][key]}
    plan["files"]={"1":plan["files"]["1"]}
    a.child.mkdir(parents=True,exist_ok=False)
    mp=a.child/"manifest.json"
    durable(mp,plan)
    command=[sys.executable,str(a.frozen_driver),"run",str(a.scenes),str(a.child),
             "--manifest-sha",evidence.digest(mp.read_bytes()),"--workers","1","--seconds","20"]
    kw=dict(stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    # Hold the first worker in its attempt by killing its coordinator immediately after fork.
    proc=subprocess.Popen(command,**kw)
    folder=a.child/"tasks"/key
    try:
        until=time.monotonic()+40
        while time.monotonic()<until:
            if proc.poll() is not None: raise RuntimeError(proc.communicate())
            if (folder/"001/start.json").exists():
                kids=Path(f"/proc/{proc.pid}/task/{proc.pid}/children").read_text().split()
                if kids: break
            time.sleep(.004)
        else: raise RuntimeError("no active owned worker")
        # Probe the same nonblocking exclusive lock without starting a competing writer.
        import fcntl
        with (a.child/"owner.lock").open("ab") as owner:
            try: fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: pass
            else: raise AssertionError("second owner acquired campaign")
        proc.kill()
        proc.wait(10)
    finally:
        if proc.poll() is None: proc.kill();proc.wait()
    until=time.monotonic()+15
    while any(Path(f"/proc/{pid}/status").exists() and
              "State:\tZ" not in Path(f"/proc/{pid}/status").read_text() for pid in kids):
        if time.monotonic()>until: raise RuntimeError("surviving active writer")
        time.sleep(.05)
    # Empty directory left by a pre-start crash also must not collide with a later attempt.
    (folder/'002').mkdir()
    subprocess.run(command,check=True)
    rec=json.loads((folder/"001/receipt.json").read_bytes())
    assert rec["status"]=="INTERRUPTED"
    committed=folder/"003/label.json"
    original=committed.read_bytes()
    rt=json.loads(original)
    full=json.loads((a.full/"tasks"/key/"001/label.json").read_bytes())
    assert rt['status']=='CERTIFIED' and rt["wire"]==full["wire"] and rt["tv"]==full["tv"]
    subprocess.run(command,check=True)
    assert list(folder.glob("*/label.json"))==[committed] and committed.read_bytes()==original
    committed.write_bytes(original[:-12]) # disposable fixture for a failed integrity check
    refused=subprocess.run(command[:2]+["verify"]+command[3:],**kw)
    if refused.returncode==0 or b"digest mismatch" not in refused.stderr:
        raise RuntimeError("corruption was not rejected")
    committed.write_bytes(original)
    subprocess.run(command[:2]+["verify"]+command[3:],check=True)
    print(json.dumps(dict(key=key,killed_active_children=len(kids),interrupted=rec,
          retries=1,wire_equal=True,bound_equal=True,committed_sha=evidence.digest(original),
          corrupt_copy_refused=True,repeat_resume_new_attempts=0,parallel_owner_refused=True)))
if __name__=="__main__": main()
