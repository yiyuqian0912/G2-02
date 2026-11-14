"""Resumable physical-stage runner: run_physical.py <campaign.json> <root> <slot_from> <slot_to> [k_plan]
campaign.json: {"seed", "resolution", "cap": {"64","128","256"}, "rate_n", "c_table": {L: [c1..c4]}, "k_cycle"}.
Each finished slot appends one line to <root>/manifest.jsonl (accepted / rejected / failed); finished slots are
skipped on restart; a rejected scene is retried with attempt+1 up to max_attempts (all attempts recorded)."""
import sys, json, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rind3d.rendering.backends import BackendFactory
from rind3d_1k.physical import run_scene, _rng
from rind3d_1k.controller import Controller


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    root = Path(sys.argv[2]); root.mkdir(parents=True, exist_ok=True)
    a, b = int(sys.argv[3]), int(sys.argv[4])
    man = root / "manifest.jsonl"
    done = set()
    if man.exists():
        for l in man.read_text().splitlines():
            r = json.loads(l)
            if r["status"] in ("accepted",) or r.get("final"):
                done.add(r["slot"])
    fac = BackendFactory()
    ctl = None
    if cfg.get("controller"):
        # deterministic controller state: prior from independent canary manifests, then every accepted release scene in
        # slot order (only scenes BEFORE the current slot are ever used)
        ctl = Controller()
        for pm in cfg.get("controller_prior_manifests", []):
            for l in Path(pm).read_text().splitlines():
                r = json.loads(l)
                if r["status"] == "accepted":
                    for s_ in r["sides"]:
                        ctl.nA[int(s_["Lp"])][r["k"]] += __import__("numpy").asarray(s_["rate_m"][:r["k"]], float)
                        ctl.rate[int(s_["Lp"])][r["k"]][0] += s_["rate_A"]; ctl.rate[int(s_["Lp"])][r["k"]][1] += 1024
        if man.exists():
            acc = sorted((json.loads(l) for l in man.read_text().splitlines()), key=lambda r: r["slot"])
            for r in acc:
                if r["status"] == "accepted" and r["slot"] < a:
                    ctl.update(r)
    for slot in range(a, b):
        if slot in done:
            if ctl is not None:
                for l in man.read_text().splitlines():
                    r = json.loads(l)
                    if r["slot"] == slot and r["status"] == "accepted":
                        ctl.update(r)
            continue
        if ctl is not None:
            u = float(_rng(cfg["seed"], slot, 56000, 0).random())
            k, c_table, plan = ctl.choose(u=u, floor=cfg.get("k_floor", 0.15))
        else:
            k = cfg["k_cycle"][slot % len(cfg["k_cycle"])] if "k_cycle" in cfg else cfg["k_of_slot"][str(slot)]
            c_table = cfg["c_table"]; plan = None
        for attempt in range(cfg.get("max_attempts", 3)):
            t = time.perf_counter()
            try:
                rec = run_scene(cfg, slot, k, c_table, root, fac, attempt)
            except Exception as e:
                rec = dict(slot=slot, attempt=attempt, status="failed", error=repr(e)[:500],
                           trace=traceback.format_exc()[-1500:])
            rec["wall"] = time.perf_counter() - t
            if plan is not None:
                rec["controller_plan"] = plan; rec["c_table"] = c_table; rec["k_chosen"] = k
            rec["final"] = rec["status"] == "accepted" or attempt == cfg.get("max_attempts", 3) - 1
            with open(man, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps({kk: rec.get(kk) for kk in ("slot", "attempt", "status", "reason", "k", "family", "m_hist", "wall", "error")}), flush=True)
            if rec["status"] == "accepted":
                if ctl is not None:
                    ctl.update(rec)
                    (root / "controller_state.json").write_text(json.dumps(dict(after_slot=slot, **ctl.state())))
                break


if __name__ == "__main__":
    main()
