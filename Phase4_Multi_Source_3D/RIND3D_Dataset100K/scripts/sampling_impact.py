"""Sampling-impact canary for contract v3 (docs/SAMPLING_IMPACT_V3.md).

usage: sampling_impact.py --arm v2|v3rule_v2seed|v3 --slots 0-71 --out DIR [--keep-scenes] [--thin T] [--families a,b] [--name NAME]
Arms:
  v2              contract v2 as declared (window rule "mixed");
  v3rule_v2seed   contract v2's seed, kappa and streams with the v3 rule (identical worlds, sources and proposal streams
                  to arm v2: a paired comparison of the rule alone; analysis only, not a production contract);
  v3              contract v3 as declared (its own seed).
--thin T sets the v3 thin-slab rule (minimum thickness T voxels; 0 = remove it) on any arm, e.g. v2 with T = 2 isolates
the thin-slab change, v3rule_v2seed with T = 1, 2, 3 compares the minimum thickness on identical streams.
--families restricts to slots whose family is listed (round robin slot % 9). --name overrides the arm directory.
Writes DIR/<arm>/records.jsonl (one physical record per slot: status, timing, work, per-side rate statistics, windows)
and, with --keep-scenes, the scene containers under DIR/<arm>/scenes/. Sequential, one GPU; the hardware is recorded."""
import argparse, json, platform, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def hardware():
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    return dict(gpu=gpu, cpu=cpu, python=platform.python_version(), host=platform.node())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["v2", "v3rule_v2seed", "v3"])
    ap.add_argument("--slots", default="0-71"); ap.add_argument("--out", required=True)
    ap.add_argument("--keep-scenes", action="store_true")
    ap.add_argument("--thin", type=float, default=None); ap.add_argument("--families", default=None)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    from rind3d.rendering.backends import BackendFactory
    from rind3d_100k import sampling as SM
    from rind3d_100k.physical import run_scene
    from rind3d_100k.storage import scene_path, atomic_write
    C = HERE / "contract"
    if a.arm == "v3":
        cfg = json.loads((C / "RIND3D_100K_CONTRACT_v3.json").read_text())
    else:
        cfg = json.loads((C / "RIND3D_100K_CONTRACT_v2.json").read_text())
        if a.arm == "v3rule_v2seed":
            v3 = json.loads((C / "RIND3D_100K_CONTRACT_v3.json").read_text())
            cfg["eligibility"] = v3["eligibility"]
            cfg["contract"] = "analysis:v3-rule-on-v2-seed"
    if a.thin is not None:
        v3 = json.loads((C / "RIND3D_100K_CONTRACT_v3.json").read_text())
        if a.thin > 0:
            cfg["geometry"] = dict(v3["geometry"], thin_min_thickness_vox=float(a.thin))
        else:
            cfg.pop("geometry", None)
        cfg["contract"] = cfg["contract"] + f"+thin{a.thin:g}" if not cfg["contract"].startswith("rind3d") else f"analysis:{cfg['contract']}+thin{a.thin:g}"
    lo, hi = (int(x) for x in a.slots.split("-"))
    out = Path(a.out) / (a.name or a.arm); out.mkdir(parents=True, exist_ok=True)
    fams = set(a.families.split(",")) if a.families else None
    (out / "run.json").write_text(json.dumps(dict(arm=a.arm, name=a.name or a.arm, contract=cfg["contract"], seed=cfg["seed"],
                                                  rule=SM.eligibility(cfg), geometry=cfg.get("geometry"), families=sorted(fams) if fams else None,
                                                  slots=[lo, hi], hardware=hardware())))
    fac = BackendFactory()
    done = set()
    rp = out / "records.jsonl"
    if rp.exists():
        done = {json.loads(l)["slot"] for l in rp.read_text().splitlines() if l.strip()}
    for slot in range(lo, hi + 1):
        if slot in done or (fams and cfg["families"][slot % len(cfg["families"])] not in fams):
            continue
        k, _ = SM.kappa_draw(cfg, slot)
        for attempt in range(int(cfg["max_attempts"])):
            t = time.perf_counter()
            rec, payload = run_scene(cfg, slot, k, None, fac, attempt)
            rec["wall"] = time.perf_counter() - t
            if payload is not None or attempt + 1 == int(cfg["max_attempts"]):
                break
            with open(out / "rejected_attempts.jsonl", "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        if payload is not None and a.keep_scenes:
            atomic_write(scene_path(out, slot), payload)
        with open(rp, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(json.dumps(dict(slot=slot, status=rec["status"], family=rec.get("family"), k=k, wall=round(rec["wall"], 1))), flush=True)


if __name__ == "__main__":
    main()
