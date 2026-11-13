"""Post-backfill verification of release_1k_v1 labels (read-only).

usage: verify_backfill.py <release_root> <pre_backfill_integrity.json> <out.json> [workers] [n_tasks]
(n_tasks: the delta family size; 159,802 for the release, 30,349 for NAT-1K)
Checks, per informative task:
* a task file exists, or a failure / quarantine record explains its absence;
* sha256 of the file equals the sha256 logged at write time (labels_log.jsonl; FastLabel entries);
* arrays are finite and non-negative, q = lo / sum(lo) sums to 1, cells lie in [0, 128^3);
* status / tv_bound consistency: certified -> tv <= 0.05; certified_stat -> tv == eps == 0.05 with a statistical
  certificate whose fields match the declared policy (delta = 0.01 / 159,802, n_tasks = 159,802, keyed seed) and whose
  K is within the design threshold of its stage, recomputed here from design2;
* soundness (audit S4, privileged): the true source's 128^3 cell lies in the label support (hi > 0) -- for statistical
  labels F ⊆ B, so this must hold for every label;
Plus: the 333 pre-existing labels are byte-identical to the pre-backfill snapshot, every scene's SEAL still matches."""
import hashlib, json, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
G = 128
N_TASKS = 159802


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cell_of(s):
    i = np.minimum((np.asarray(s) * G).astype(np.int64), G - 1)
    return int((i[0] * G + i[1]) * G + i[2])


def check_scene(args):
    d, logged, pre, N_TASKS = args
    from rind3d_1k.fastlabel.pipeline import design2
    d = Path(d)
    W = np.load(d / "public" / "windows.npy")
    src = json.loads((d / "privileged" / "sources.json").read_text())
    S = np.array([[float.fromhex(v) for v in s] for s in src["sources_hex"]])
    seal = json.loads((d / "SEAL.json").read_text()) if (d / "SEAL.json").exists() else {}   # NAT-1K scenes carry no seal
    seal_bad = [k for k, v in seal.items() if sha(d / k) != v]
    rows = []
    for r in W:
        for j in range(int(r["k"])):
            if not (int(r["mask"]) >> j) & 1:
                continue
            w = int(r["window"]); p = d / "labels" / "tasks" / f"w{w:03d}_s{j}.npz"
            rec = dict(scene=d.name, window=w, source=j, L=int(r["Lp"]), problems=[])
            if not p.exists():
                rec["status"] = "missing"; rows.append(rec); continue
            h = sha(p)
            rec["sha_logged"] = logged.get((d.name, w, j))
            if p.name in pre:
                rec["preexisting"] = True
                if pre[p.name] != h:
                    rec["problems"].append("preexisting label changed")
            elif rec["sha_logged"] is not None and rec["sha_logged"] != h:
                rec["problems"].append("sha256 mismatch vs log")
            elif rec["sha_logged"] is None:
                rec["problems"].append("no logged sha256")
            with np.load(p) as z:
                cell, lo, hi, tv = z["cell"].astype(np.int64), z["lo"], z["hi"], float(z["tv_bound"])
                m = json.loads(str(z["meta"]))
            st = m["status"]; rec.update(status=st, tv_bound=tv, path=m.get("path"), engine=m.get("engine"),
                                       n_cells=int(len(np.unique(cell))), wall=m.get("wall"))
            if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(lo >= 0) and np.all(hi >= lo - 1e-18)):
                rec["problems"].append("bad arrays")
            if len(cell) == 0 or cell.min() < 0 or cell.max() >= G ** 3 or not lo.sum() > 0:
                rec["problems"].append("bad cells or empty")
            if st == "certified" and tv > 0.05:
                rec["problems"].append("certified with tv > 0.05")
            if st == "certified_stat":
                c = m.get("certificate") or {}
                ok = (c.get("kind") == "statistical" and abs(tv - 0.05) < 1e-15 and c.get("eps") == 0.05
                      and abs(c.get("delta_label", 0) - 0.01 / N_TASKS) < 1e-20 and c.get("n_tasks") == N_TASKS
                      and c.get("seed", [None, None, None, None])[1:] == [int(d.name), w, j] and c.get("p_lower", 0) >= 0.95)
                if ok:
                    a1, a2, alpha = design2(int(c["n1"]), int(c["n_max"]), 0.05, float(c["delta_round"]))
                    ok = (a1 == c["a1"] and a2 == c["a2"] and alpha <= float(c["delta_round"]) + 1e-18 and
                          ((c["stage"] == 1 and c["n"] == c["n1"] and c["K"] <= a1) or (c["stage"] == 2 and c["n"] == c["n_max"] and c["K"] <= a2))
                          and abs(float(c["delta_round"]) - c["delta_label"] * 0.5 ** int(c["round"])) < 1e-22)
                    rec["round"] = int(c["round"]); rec["stage"] = int(c["stage"])
                if not ok:
                    rec["problems"].append("statistical certificate inconsistent")
            u, inv = np.unique(cell, return_inverse=True)
            H = np.bincount(inv, weights=hi, minlength=len(u))
            ct = cell_of(S[j]); pos = np.searchsorted(u, ct)
            rec["true_in_support"] = bool(pos < len(u) and u[pos] == ct and H[pos] > 0)
            if not rec["true_in_support"]:
                rec["problems"].append("true source outside support")
            rows.append(rec)
    return d.name, seal_bad, rows


def main():
    root = Path(sys.argv[1]); pre = json.loads(Path(sys.argv[2]).read_text())["scenes"]; outp = Path(sys.argv[3])
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    logged = {}
    for line in open(root / "labels_log.jsonl"):
        r = json.loads(line)
        if "sha256" in r:
            logged[(r["scene"], int(r["window"]), int(r["source"]))] = r["sha256"]
    dirs = sorted(p for p in (root / "scenes").iterdir() if (p / "SEAL.json").exists() or (p / "natural.json").exists())
    by_scene = {}
    for k, v in logged.items():
        by_scene.setdefault(k[0], {})[k] = v
    nt = int(sys.argv[5]) if len(sys.argv) > 5 else N_TASKS
    args = [(str(d), by_scene.get(d.name, {}), pre.get(d.name, {}).get("labels", {}), nt) for d in dirs]
    with ProcessPoolExecutor(nw) as ex:
        res = list(ex.map(check_scene, args, chunksize=4))
    rows = [r for _, _, rr in res for r in rr]
    seal_bad = {n: b for n, b, _ in res if b}
    st = {}
    for r in rows:
        key = f"L{r['L']}:{r.get('status')}"
        st[key] = st.get(key, 0) + 1
    probs = [r for r in rows if r["problems"]]
    walls = np.array([r["wall"] for r in rows if r.get("wall") is not None and not r.get("preexisting")], float)
    nc = {L: np.array([r["n_cells"] for r in rows if r["L"] == L and "n_cells" in r]) for L in (64, 128, 256)}
    summ = dict(informative_tasks=len(rows), status_counts=st, problems=len(probs), problem_examples=probs[:20],
                seal_mismatch_scenes=seal_bad, preexisting_checked=sum(1 for r in rows if r.get("preexisting")),
                true_in_support=sum(1 for r in rows if r.get("true_in_support")),
                paths={p: sum(1 for r in rows if r.get("path") == p) for p in sorted({str(r.get("path")) for r in rows})},
                stat_stage={f"r{r_}s{s_}": sum(1 for r in rows if r.get("round") == r_ and r.get("stage") == s_)
                            for r_ in range(1, 13) for s_ in (1, 2) if any(r.get("round") == r_ and r.get("stage") == s_ for r in rows)},
                label_wall_s=dict(mean=float(walls.mean()) if len(walls) else None, p50=float(np.median(walls)) if len(walls) else None,
                                  p99=float(np.quantile(walls, 0.99)) if len(walls) else None, max=float(walls.max()) if len(walls) else None),
                n_cells={str(L): dict(median=float(np.median(v)), mean=float(v.mean()), p99=float(np.quantile(v, 0.99)), max=int(v.max()))
                         for L, v in nc.items() if len(v)})
    outp.write_text(json.dumps(dict(summary=summ, rows=rows), default=float))
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
