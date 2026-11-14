"""Rendering strategies for the physical stage (Section 10): A full certified per-source fields + exact count index
(as produced), B independent local rendering of every proposal window (certified CUDA 64^3 tiles, one call per
tile), C batched local rendering (up to 32 tiles per call, one bound source per channel).  For each scene the
proposals actually made by the selection (selection_log) are re-rendered locally; local responses are compared
bit-for-bit with the full-field crops on a subset.  Usage: render_benchmark.py <scene_dir> [max_proposals_per_side]"""
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rind3d.geometry import load_geometry
from rind3d.rendering.backends import BackendFactory
from rind3d.pack import RINDPack
from rind3d_1k.physical import POLICY

d = Path(sys.argv[1]); cap = int(sys.argv[2]) if len(sys.argv) > 2 else 200
sc = json.loads((d / "scene.json").read_text()); N = sc["resolution"]; k = sc["k"]
bundle = load_geometry(d / "privileged" / "geometry.npz")
src = json.loads((d / "privileged" / "sources.json").read_text())
S = np.array([[float.fromhex(v) for v in s] for s in src["sources_hex"]])
logs = np.load(d / "privileged" / "selection_log.npz")
fac = BackendFactory()
T = sc["timing"]
A = dict(fields=T["fields"], count_index=T["count_index"], selection=T["selection"], materialize=T["materialize"])
res = dict(scene=d.name, k=k, A_full_field=A, A_total=sum(A.values()))
tiles_per = lambda L: (L // 64) ** 3
B_s = 0.0; C_s = 0.0; nprop = 0; tiles = 0; mism = 0; checked = 0
srcs = [fac.source(bundle, S[j], "cuda_certified", POLICY) for j in range(k)]
try:
    resp = [RINDPack(d / "privileged" / "fields" / f"src{j}" / "response.rindb") for j in range(k)]
    for Lp in (64, 128, 256):
        log = logs[f"L{Lp}"]; L = Lp * N // 1024
        sel = log[:cap]; nprop += len(sel)
        origins = []
        for o in sel[:, :3].astype(int):
            for a in range(0, L, 64):
                for b in range(0, L, 64):
                    for c in range(0, L, 64):
                        origins.append((int(o[0] + a), int(o[1] + b), int(o[2] + c)))
        tiles += len(origins)
        for j in range(k):
            t = time.perf_counter()
            for o in origins[:64]:
                srcs[j].packed_batch([o], 64, N)
            per_tile_B = (time.perf_counter() - t) / min(64, len(origins))
            B_s += per_tile_B * len(origins)
            t = time.perf_counter(); out = []
            for i in range(0, len(origins), 32):
                out += srcs[j].packed_batch(origins[i:i + 32], 64, N)
            C_s += time.perf_counter() - t
            for (o, pr) in list(zip(origins, out))[:: max(1, len(origins) // 16)]:
                lit = np.unpackbits(np.frombuffer(pr[0], np.uint8), bitorder="little").astype(bool).reshape(64, 64, 64)
                ref = resp[j].read_window(*o, 64)
                mism += int(np.sum(lit != ref)); checked += lit.size
finally:
    for s in srcs:
        s.close()
scale = {str(Lp): len(logs[f"L{Lp}"]) / max(1, min(cap, len(logs[f"L{Lp}"]))) for Lp in (64, 128, 256)}
res.update(proposals_rendered=nprop, tiles=tiles, B_local_independent_s=B_s, C_local_batched_s=C_s,
           proposals_total={str(Lp): int(len(logs[f"L{Lp}"])) for Lp in (64, 128, 256)},
           note="B/C cover only the selection proposals listed (first %d per side); rate streams excluded" % cap,
           local_vs_field_bits_checked=checked, local_vs_field_mismatches=mism)
print(json.dumps(res))
