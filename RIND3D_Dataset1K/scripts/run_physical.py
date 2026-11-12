"""Resumable physical-stage runner: run_physical.py <campaign.json> <root> <slot_from> <slot_to> [k_plan]
campaign.json: {"seed", "resolution", "cap": {"64","128","256"}, "rate_n", "c_table": {L: [c1..c4]}, "k_cycle"}.
Each finished slot appends one line to <root>/manifest.jsonl (accepted / rejected / failed); finished slots are
skipped on restart; a rejected scene is retried with attempt+1 up to max_attempts (all attempts recorded)."""
import sys, json, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rind3d.rendering.backends import BackendFactory
from rind3d_1k.physical import run_scene


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
    for slot in range(a, b):
        if slot in done:
            continue
        k = cfg["k_cycle"][slot % len(cfg["k_cycle"])] if "k_cycle" in cfg else cfg["k_of_slot"][str(slot)]
        c_table = cfg["c_table"]
        for attempt in range(cfg.get("max_attempts", 3)):
            t = time.perf_counter()
            try:
                rec = run_scene(cfg, slot, k, c_table, root, fac, attempt)
            except Exception as e:
                rec = dict(slot=slot, attempt=attempt, status="failed", error=repr(e)[:500],
                           trace=traceback.format_exc()[-1500:])
            rec["wall"] = time.perf_counter() - t
            rec["final"] = rec["status"] == "accepted" or attempt == cfg.get("max_attempts", 3) - 1
            with open(man, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps({kk: rec.get(kk) for kk in ("slot", "attempt", "status", "reason", "k", "family", "m_hist", "wall", "error")}), flush=True)
            if rec["status"] == "accepted":
                break


if __name__ == "__main__":
    main()
