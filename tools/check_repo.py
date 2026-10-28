#!/usr/bin/env python3
"""Run source/shell and phase-specific CPU checks without installing raw data."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "Phase1_Single_Source_Single_Intensity"
PHASE2 = ROOT / "Phase2_Multi_Source"


def run(args, cwd, env):
    print(f"Check: {cwd.name} | {' '.join(map(str, args))}", flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--phase", choices=("all", "phase1", "phase2"), default="all")
    args = parser.parse_args()
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = env.get("RIND_CHECK_CACHE_ROOT", str(ROOT / "runtime_cache/checks/numba"))
    Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    env["NUMBA_NUM_THREADS"] = "2"
    # The same check runs inside a transfer package without a .git directory.
    scripts = set(ROOT.glob('*.sh')) | set(PHASE1.glob('*.sh')) | set(PHASE2.glob('*.sh'))
    for directory in (ROOT / 'tools', PHASE1 / 'scripts', PHASE2 / 'scripts', PHASE2 / 'dataset_update'):
        scripts.update(directory.rglob('*.sh'))
    scripts = sorted(scripts)
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
    print(f"Shell syntax: {len(scripts)} entry points passed", flush=True)
    run([args.python, "-m", "unittest", "discover", "-s", "tools", "-p", "test_*.py", "-q"], ROOT, env)
    for phase, selected in ((PHASE1, "phase1"), (PHASE2, "phase2")):
        if args.phase not in ("all", selected):
            continue
        paths = [phase, phase / "src", phase / "vendor/rind-dataset", PHASE1 / "src", PHASE2 / "src",
                 PHASE2 / "research/implicit_slots", PHASE2 / "research/layered_residual"]
        current = {**env, "PYTHONPATH": os.pathsep.join(map(str, paths))}
        run([args.python, "-m", "unittest", "discover", "-s", "tests", "-q"], phase, current)
        if selected == "phase1":
            run([args.python, "-m", "unittest", "part_e.test_part_e", "part_e.test_integration", "-q"], phase, current)
        else:
            run([args.python, "-m", "unittest", "dataset_update.test_dataset_union", "-q"], phase, current)
            run([args.python, "-m", "unittest", "discover", "-s", "research/implicit_slots", "-p", "test_*.py", "-q"], phase, current)
    print("Repository checks passed.", flush=True)


if __name__ == "__main__":
    main()
