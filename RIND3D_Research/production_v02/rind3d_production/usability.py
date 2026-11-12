"""Training-usability checks (contract C8) on a published root.

1 student-local random access throughput (one process, uniform random windows of each side, cold LRU)
2 a 2-worker torch DataLoader pass over a bounded random subset (NumPy payloads in workers, collation in parent)
3 TeacherPhysics: batched certified visibility for the true sources equals the stored fields; hypothetical
  candidate batch throughput; full-window 64^3 cost per candidate
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from .loader import ProductionDataset, collate_numpy
from .teacher import window_centres


def main(root):
    root = Path(root)
    out = {}
    rng = np.random.default_rng(5)
    for prof, name in (("rind3d-csg-single-binary-v0.2", "SB"), ("rind3d-csg-multi-additive-v0.2", "MA")):
        ds = ProductionDataset(root, profiles=[prof], subsets=("natural", "informative", "extreme", "rare", "nonlocal"))
        sizes = {}
        for L in (64, 128, 256):
            idx = [i for i in rng.permutation(len(ds))[:4000] if ds._scene(int(ds.index[i][0]))["public_windows"][int(ds.index[i][1])]["L"] == L][:150]
            ds.cache.clear()
            t = time.perf_counter()
            for i in idx:
                ds[int(i)]
            dt = time.perf_counter()-t
            sizes[str(L)] = {"windows": len(idx), "windows_per_second": len(idx)/dt}
        out[f"student_local_random_access_{name}"] = sizes
        out[f"windows_{name}"] = len(ds)
    import torch
    from torch.utils.data import DataLoader, Subset
    ds = ProductionDataset(root, subsets=("natural",))
    sub = Subset(ds, rng.permutation(len(ds))[:256].tolist())
    t = time.perf_counter()
    n = 0
    for batch in DataLoader(sub, batch_size=8, num_workers=2, collate_fn=collate_numpy):
        n += len(batch["window_id"])
    out["dataloader_2_workers"] = {"windows": n, "seconds": time.perf_counter()-t, "torch": torch.__version__}
    tds = ProductionDataset(root, view="teacher", research_access=True, subsets=("informative",))
    checks, cand = [], []
    for i in rng.permutation(len(tds))[:12]:
        s = tds[int(i)]
        phys = tds.physics(int(i))
        free = np.flatnonzero(~s["observed_solid"].ravel())
        take = rng.choice(free, min(2048, len(free)), replace=False)
        xyz = window_centres(s["window_origin"], s["window_size"])[take]
        vis = phys.visibility(s["sources"], xyz)
        checks.append(bool(all(np.array_equal(vis[k] == 1, s["channels"][k].ravel()[take]) for k in range(s["K"]))))
        hyp = []
        while len(hyp) < 32:
            p = rng.random(3)
            if phys.legal([p])[0]:
                hyp.append(p)
        t = time.perf_counter()
        phys.visibility(np.array(hyp), xyz)
        cand.append(32*len(xyz)/(time.perf_counter()-t))
    out["teacher"] = {"true_source_queries_equal_stored": checks, "all_equal": all(checks),
                      "candidate_pairs_per_second_median": float(np.median(cand))}
    (root/"usability.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
