"""Run lab arms sequentially: train (resumable) then evaluate on the dev split. Re-reads the queue file between arms, so
arms can be appended while it runs. Facet arms wait until the facet table exists.

usage: run_queue.py <queue.txt>     one config path (relative to the lab directory) per line; '#' comments
An arm is done when runs/<name>/eval_dev.npz exists. Launch failures are retried (3 attempts)."""
import json, subprocess, sys, time
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))
from slab.paths import OUT, facets_file   # noqa: E402

PY = str(LAB / ".venv" / "bin" / "python")


def arms(qf):
    out = []
    for line in Path(qf).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.append(LAB / line)
    return out


def run(cmd, log):
    for attempt in range(3):
        with open(log, "a") as f:
            r = subprocess.run(cmd, cwd=LAB, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode == 0:
            return True
        time.sleep(10)
    return False


def main():
    qf = sys.argv[1]; failed = set()
    while True:
        todo = None
        for p in arms(qf):
            cfg = json.loads(p.read_text()); d = OUT / "runs" / cfg["name"]
            if (d / "eval_dev.npz").exists() or cfg["name"] in failed:
                continue
            if cfg.get("model") == "facet" and not facets_file(cfg.get("facets", "v1")).exists():
                continue
            todo = (p, cfg, d); break
        if todo is None:
            pending = [p for p in arms(qf) if not (OUT / "runs" / json.loads(p.read_text())["name"] / "eval_dev.npz").exists()]
            if not pending or all(json.loads(p.read_text())["name"] in failed for p in pending):
                print(json.dumps(dict(queue="empty", failed=sorted(failed))), flush=True); return
            time.sleep(60); continue
        p, cfg, d = todo; d.mkdir(parents=True, exist_ok=True); t0 = time.time()
        ok = (d / "final.pt").exists() or run([PY, "-m", "slab.train", str(p)], d / "train.out")
        ok = ok and run([PY, "-m", "slab.evaluate", str(d)], d / "eval.out")
        if not ok:
            failed.add(cfg["name"])
        print(json.dumps(dict(arm=cfg["name"], ok=ok, wall=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
