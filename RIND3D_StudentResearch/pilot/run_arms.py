"""Run pilot arms sequentially (train -> evaluate best checkpoint on pilot-eval), resumable.
usage: run_arms.py <feat_dir> <runs_dir> <arms.json> [--train-only | --eval-only]
arms.json: list of arm dicts (train.py keys). A finished arm has runs/<name>/eval.npz and is skipped."""
import os
import json, subprocess, sys, time
from pathlib import Path

PY = os.environ.get("RIND3D_TORCH_PY", sys.executable)   # torch interpreter (RIND3D_HANDOFF/env.sh)
HERE = Path(__file__).resolve().parent
feat, runs, arms = sys.argv[1], Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
for arm in arms:
    d = runs / arm["name"]
    if (d / "eval.npz").exists():
        continue
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if "--eval-only" not in sys.argv and not (d / "done.json").exists():
        r = subprocess.run([PY, str(HERE / "train.py"), feat, str(d), json.dumps(arm)], stdout=open(d / "train.out", "a"),
                           stderr=subprocess.STDOUT)
        if r.returncode:
            print(json.dumps(dict(arm=arm["name"], train_failed=r.returncode)), flush=True); continue
    if "--train-only" in sys.argv or not (d / "done.json").exists():
        print(json.dumps(dict(arm=arm["name"], trained=(d / "done.json").exists(), wall=time.time() - t0)), flush=True); continue
    r = subprocess.run([PY, str(HERE / "evaluate.py"), feat, str(d / "best.pt"), str(d / "eval.npz")],
                       stdout=open(d / "eval.out", "a"), stderr=subprocess.STDOUT)
    print(json.dumps(dict(arm=arm["name"], eval_rc=r.returncode, wall=time.time() - t0)), flush=True)
