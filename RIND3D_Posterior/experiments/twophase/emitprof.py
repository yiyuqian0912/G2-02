import sys, time
exec(open('twophase.py').read().split('def main():')[0])
from rind3d_posterior import label_window
sc, w, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
S = SceneObs(R / sc); cls = S.cls(w, s)
for eps in (2.0 ** -26, 0.0):
    c0 = time.process_time()
    r = label_window(S.P, S.K, cls, S.origin(w), 1024, engine="cpu", nq_cap=1 << 25, ne_cap=1 << 26, shell_eps=eps)
    print(sc, w, s, "shell_eps", eps, "cpu %.2f" % (time.process_time() - c0), "s1 %.2f s2 %.2f" % (r["seconds"]["stage1"], r["seconds"]["stage2"]), "records", len(r["cell"]), "nodes", r["stats"]["nodes"], "evals %.2e" % r["stats"]["entry_evals"], "in", r["stats"]["in"], flush=True)
