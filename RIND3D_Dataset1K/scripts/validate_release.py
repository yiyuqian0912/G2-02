"""Validate every accepted scene of a campaign root; optional physics spot checks on a seeded sample of scenes.
Usage: validate_release.py <root> [--physics N_SCENES] -> <root>/VALIDATION.json"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from rind3d_1k.validate import validate_scene

root = Path(sys.argv[1])
scenes = sorted(p for p in (root / "scenes").iterdir() if (p / "SEAL.json").exists())
res = [validate_scene(d) for d in scenes]
out = dict(scenes=len(res), ok=sum(r["ok"] for r in res), failed=[r for r in res if not r["ok"]][:50],
           m_hist=np.sum([r["m_hist"] for r in res], axis=0).tolist() if res else [],
           labels={k: sum(r["labels"].get(k, 0) for r in res) for k in set(k for r in res for k in r["labels"])})
if "--physics" in sys.argv:
    from rind3d_1k.validate import physics_spot_check
    n = int(sys.argv[sys.argv.index("--physics") + 1])
    pick = np.random.default_rng(0).choice(len(scenes), size=min(n, len(scenes)), replace=False)
    ph = [physics_spot_check(scenes[i], n_points=2048, n_windows=3, independent_points=32) for i in sorted(pick)]
    out["physics"] = dict(scenes=len(ph), checked=sum(p["checked"] for p in ph), mismatches=sum(p["mismatches"] for p in ph),
                          independent_checked=sum(p["independent_checked"] for p in ph),
                          independent_mismatches=sum(p["independent_mismatches"] for p in ph))
(root / "VALIDATION.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "failed"}))
