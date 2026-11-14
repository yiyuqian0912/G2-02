"""Release statistics: m histograms (raw and omega-weighted), by side / k / family, overlap, coverage, rejections,
runtime, storage.  Usage: release_stats.py <root> -> <root>/RELEASE_STATS.json"""
import sys, json, collections
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
man = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines()]
acc = [r for r in man if r["status"] == "accepted"]
rej = collections.Counter(r.get("reason") or r.get("error", "")[:60] for r in man if r["status"] != "accepted")
M = np.zeros(4); Mw = np.zeros(4); byL = {L: np.zeros(4) for L in (64, 128, 256)}; byk = collections.defaultdict(lambda: np.zeros(4))
byfam = collections.defaultdict(lambda: np.zeros(4)); kh = collections.Counter(); fam = collections.Counter()
phys_wall = []; storage = 0; cov = collections.Counter(); ov = np.zeros(5)
for r in acc:
    d = root / "scenes" / f"{r['slot']:06d}"
    if not (d / "scene.json").exists():
        continue
    sc = json.loads((d / "scene.json").read_text()); W = np.load(d / "public" / "windows.npy")
    kh[sc["k"]] += 1; fam[sc["family"]] += 1; phys_wall.append(r.get("wall", 0))
    for w in W:
        m = int(w["m"]) - 1; L = int(w["Lp"])
        M[m] += 1; byL[L][m] += 1; byk[sc["k"]][m] += 1; byfam[sc["family"]][m] += 1
        Mw[m] += sc["Lambda_hat"][str(L)]
    storage += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    lab = d / "labels" / "LABELS.json"
    if lab.exists():
        for k_, v in json.loads(lab.read_text())["coverage"].items():
            cov[k_] += v
    for wf in json.loads((d / "privileged" / "windows_full.json").read_text()):
        ov[:len(wf["overlap_hist"])] += wf["overlap_hist"]
out = dict(attempts=len(man), accepted=len(acc), rejections=dict(rej), k_hist=dict(kh), families=dict(fam),
           windows=int(M.sum()), m_hist=M.tolist(), m_share=(M / max(M.sum(), 1)).round(4).tolist(),
           m_share_weighted=(Mw / max(Mw.sum(), 1e-12)).round(4).tolist(),
           m_by_side={L: v.tolist() for L, v in byL.items()}, m_by_k={k: v.tolist() for k, v in byk.items()},
           m_by_family={f: v.tolist() for f, v in byfam.items()}, overlap_hist_free_voxels=ov.tolist(),
           informative_channels=float(M @ np.arange(1, 5)), label_coverage=dict(cov),
           physical_wall_s=dict(mean=float(np.mean(phys_wall)) if phys_wall else 0, total=float(np.sum(phys_wall))),
           storage_bytes=storage)
(root / "RELEASE_STATS.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
