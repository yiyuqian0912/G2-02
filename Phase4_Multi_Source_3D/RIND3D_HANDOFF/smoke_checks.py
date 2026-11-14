"""RIND3D 1K reference pipeline: lightweight end-to-end smoke checks (read-only on the data).

usage:  source RIND3D_HANDOFF/env.sh
        $RIND3D_LABEL_PY RIND3D_HANDOFF/smoke_checks.py [n_scenes=3]
Checks (a CUDA GPU with NVRTC is needed for check 1):
  1. local response: re-render stored release windows (SOLID + every source channel, L = 64/128/256) with the certified
     CUDA backend from the canonical geometry and the true sources, tile by tile (64^3 tiles, no full-world field), and
     compare bit-for-bit with the stored public crops;
  2. certificates: re-verify every stored label of the chosen scenes with backfill/verify_backfill.check_scene
     (sha256 vs labels_log, array sanity, certificate fields and thresholds, true source inside the support);
  3. loader: flatten(population="Tstar") yields every certified status as supervision "posterior";
  4. seals: the chosen scenes' SEAL.json manifests still match.
Prints one JSON line per check and exits non-zero on any failure."""
import json, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "RIND3D_Dataset1K"), str(REPO / "RIND3D_Dataset1K" / "backfill"), str(REPO / "RIND3D_Posterior" / "src")]
DATA = Path(os.environ.get("RIND3D_1K_DATA", "/home/yushu/Projects/RIND/RIND3D_1K_data"))
REL = DATA / "release_1k_v1"
n_sc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
scenes = [REL / "scenes" / f"{i:06d}" for i in (10, 500, 999)][:n_sc]
ok_all = True


def report(name, ok, **kw):
    global ok_all
    ok_all &= bool(ok)
    print(json.dumps(dict(check=name, ok=bool(ok), **kw), default=str), flush=True)


# 1. local response re-render
from rind3d_1k.labels import SceneObs
from rind3d_1k.fastlabel.pipeline import scene_cache, process_factory
from rind3d_1k.physical import POLICY
fac = process_factory()
for d in scenes:
    S = SceneObs(d); scn = scene_cache(d)
    src = json.loads((d / "privileged" / "sources.json").read_text())["sources_hex"]
    picked = []
    for L in (64, 128, 256):
        ws = [int(r["window"]) for r in S.W if int(r["L"]) == L]
        picked.append(ws[0])
    mism = 0; bits = 0
    for w in picked:
        L = int(S.W[w]["L"]); o = np.asarray(S.origin(w)); n = L // 64
        offs = [(a, b, c) for c in range(n) for b in range(n) for a in range(n)]
        for j in range(int(S.W[w]["k"])):
            p = np.array([float.fromhex(v) for v in src[j]])
            r = fac.source(scn["bundle"], p, "cuda_certified", POLICY)
            out = r.packed_batch(np.array([o + np.array(t) * 64 for t in offs], np.int32), 64, 1024)
            r.close()
            ref = np.zeros((L, L, L), bool)
            for (a, b, c), (pl, _) in zip(offs, out):
                ref[c * 64:(c + 1) * 64, b * 64:(b + 1) * 64, a * 64:(a + 1) * 64] = \
                    np.unpackbits(np.frombuffer(pl, np.uint8), bitorder="little").reshape(64, 64, 64).astype(bool)
            stored = S.crop(w, j)
            mism += int(np.sum(ref != stored)); bits += ref.size
    report("local_response_rerender", mism == 0, scene=d.name, windows=picked, bits=bits, mismatches=mism)

# 2. certificate re-verification
import verify_backfill as VB
pre = {}
logged = {}
for line in open(REL / "labels_log.jsonl"):
    r = json.loads(line)
    if "sha256" in r and r["scene"] in {d.name for d in scenes}:
        logged[(r["scene"], int(r["window"]), int(r["source"]))] = r["sha256"]
for d in scenes:
    name, seal_bad, rows = VB.check_scene((str(d), {k: v for k, v in logged.items() if k[0] == d.name}, pre, 159802))
    probs = [r for r in rows if r["problems"] and not (r.get("preexisting") is None and r["problems"] == ["no logged sha256"]
                                                      and r.get("engine") != "fastlabel")]
    st = {}
    for r in rows:
        st[r.get("status")] = st.get(r.get("status"), 0) + 1
    report("certificates", not probs and all(r.get("true_in_support") for r in rows), scene=name, labels=len(rows),
           statuses=st, problems=len(probs))
    report("seals", not seal_bad, scene=name, mismatched=seal_bad)

# 3. loader defaults
from rind3d_1k import loader as LD
ds = LD.RIND3D1K(REL, scenes=[scenes[0].name])
sup = {}
for ex in LD.flatten(ds, population="Tstar", with_inputs=False):
    sup.setdefault(ex["supervision"], {}).setdefault(ex["status"], 0)
    sup[ex["supervision"]][ex["status"]] += 1
report("loader_default_accepts_certified", set(sup) == {"posterior"}, scene=scenes[0].name, supervision=sup)
sys.exit(0 if ok_all else 1)
