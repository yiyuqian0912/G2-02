"""PREPARED, NOT RUN BY THE AUDIT. FastLabel labels for a scene subset, written to an EXTERNAL directory (never into the
release). Input of the Student pilot (P1_STUDENT_PILOT.md).

Usage (main agent; GPU + CPU; 15 shards like the production runner):
  export RIND_NVRTC_LIBRARY=/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12
  export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/home/yushu/.cache/rind3d_numba
  for i in $(seq 0 14); do nice -n 10 /home/yushu/Projects/RIND/G2-02-3d-100k/.venv/bin/python p1_label_subset.py \
      --out /home/yushu/Projects/RIND/RIND3D_Audit_runs/p1_labels --train 160 --val 40 --shard $i/15 & done; wait
Estimated cost: 200 scenes x ~160 informative channels = ~32k labels; at the HOLDOUT throughput (12.2 labels/s with
15 workers) about 45 min of the full laptop. Disk: ~47 KB/label delta-encoded, here raw npz ~0.1-2 MB for broad labels
-> budget ~10 GB. The delta budget of these research labels is separate from the 1K release budget (state it).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Dataset1K"); sys.path.insert(0, "/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Posterior/src")
from rind3d_1k.loader import split_of
ROOT = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1/scenes")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--train", type=int, default=160)
    ap.add_argument("--val", type=int, default=40); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
    out = Path(a.out); assert not str(out).startswith(str(ROOT.parent)), "never write into the release"
    i, n = map(int, a.shard.split("/"))
    from rind3d_1k.fastlabel import label_task
    pick = {"train": [], "val": []}
    for d in sorted(ROOT.iterdir()):
        sp = split_of(json.loads((d / "scene.json").read_text())["scene_id"])
        if sp in pick and len(pick[sp]) < (a.train if sp == "train" else a.val):
            pick[sp].append(d)
    tasks = []
    for sp, ds in pick.items():
        for d in ds:
            W = np.load(d / "public" / "windows.npy")
            for r in W:
                for j in range(int(r["k"])):
                    if (int(r["mask"]) >> j) & 1:
                        tasks.append((sp, d, int(r["window"]), j))
    tasks.sort(key=lambda t: (t[1].name, t[2], t[3]))
    for t, (sp, d, w, j) in enumerate(tasks):
        if t % n != i:
            continue
        p = out / sp / d.name / f"w{w:03d}_s{j}.npz"
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time(); lab = label_task(str(d), w, j)
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, cell=lab["cell"], lo=lab["lo"], hi=lab["hi"], kind=lab["kind"], tv_bound=lab["tv_bound"],
                            meta=json.dumps(dict(status=lab["status"], path=lab["meta"].get("path"), wall=time.time() - t0,
                                                 certificate=lab["meta"].get("certificate")), default=float))
        tmp.rename(p)


if __name__ == "__main__":
    main()
