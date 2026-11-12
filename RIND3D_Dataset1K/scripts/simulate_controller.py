"""Offline feasibility of the balancing controller: replay scenes drawn (bootstrap, by k) from a canary's rate-stream
statistics. Usage: simulate_controller.py <manifest.jsonl> <n_scenes> [seed]"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rind3d_1k.controller import Controller, QUOTA, SIDES

man = [json.loads(l) for l in open(sys.argv[1]) if json.loads(l)["status"] == "accepted"]
n = int(sys.argv[2]); rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
byk = {k: [r for r in man if r["k"] == k] for k in range(1, 5)}
ctl = Controller()
for r in man:                      # canary statistics as the controller's starting knowledge (rates only)
    k = r["k"]
    for s in r["sides"]:
        ctl.nA[int(s["Lp"])][k] += np.asarray(s["rate_m"][:k], float)
        ctl.rate[int(s["Lp"])][k][0] += s["rate_A"]; ctl.rate[int(s["Lp"])][k][1] += 1024
ks, props, bySide = [], [], {L: np.zeros(4) for L in SIDES}
for t in range(n):
    k, c, info = ctl.choose(u=float(rng.random()), floor=float(sys.argv[4]) if len(sys.argv) > 4 else 0.10)
    sc = byk[k][rng.integers(len(byk[k]))]
    mh = np.zeros(4); P = 0
    sides = []
    for s in sc["sides"]:
        L = int(s["Lp"]); p = np.asarray(s["rate_m"][:k], float); p = p / max(p.sum(), 1)
        cc = np.asarray(c[str(L)][:k]); pi = p * cc
        PA = s["rate_A"] / 1024
        acc = rng.multinomial(QUOTA[L], pi / pi.sum())
        mh[:k] += acc; bySide[L][:k] += acc
        P += QUOTA[L] / max(PA * pi.sum(), 1e-9)
        sides.append(dict(Lp=L, rate_m=s["rate_m"], rate_A=s["rate_A"]))
    ctl.update(dict(k=k, m_hist=mh.tolist(), sides=sides))
    ks.append(k); props.append(P)
N = ctl.N
print(json.dumps(dict(scenes=n, m_hist=N.tolist(), m_share=(N / N.sum()).round(4).tolist(),
                      k_hist=np.bincount(ks, minlength=5)[1:].tolist(),
                      proposals_per_scene=dict(mean=float(np.mean(props)), p95=float(np.percentile(props, 95)), max=float(np.max(props))),
                      by_side={L: (v / v.sum()).round(3).tolist() for L, v in bySide.items()},
                      informative_labels=float(N @ np.arange(1, 5)), channels_total=float(64 * np.sum(ks)))))
