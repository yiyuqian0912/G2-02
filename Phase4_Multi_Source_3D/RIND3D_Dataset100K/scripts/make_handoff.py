#!/usr/bin/env python3
"""Build the portable RIND3D-100K server handoff archive from the committed Git revision.

  make_handoff.py --out /path/outdir [--with-deps DIR] [--allow-dirty]

The archive contains, from `git archive HEAD` (tracked files only, no data, venvs or secrets):
* RIND3D_Generator, minus experiments/ and docs/assets/;
* RIND3D_Posterior;
* RIND3D_Dataset1K (its rind3d_1k package is reused by the 100K pipeline);
* RIND3D_Dataset100K, with the handoff fixtures, controller prior, environment scripts and docs;
* docs/RIND3D.

Optional `deps/`: the CGAL 6.2.1 source tarball and the Boost 1.92 package, for offline builds.

Outputs:
* `rind3d_100k_handoff_<commit12>.tar.gz`;
* `rind3d_100k_handoff_<commit12>.sha256`;
* `HANDOFF_MANIFEST.json`, a copy of the manifest also stored inside the archive: commit, branch, remote, sha256 of
  every file, contract and fixture hashes, lock hash.
"""
import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent
PATHS = ["RIND3D_Generator", "RIND3D_Posterior", "RIND3D_Dataset1K", "RIND3D_Dataset100K", "docs/RIND3D"]
EXCLUDE = [":(exclude)RIND3D_Generator/experiments", ":(exclude)RIND3D_Generator/docs/assets"]


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--with-deps"); ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dirty = git("status", "--porcelain", "--", *PATHS)
    if dirty and not a.allow_dirty:
        raise SystemExit("uncommitted changes in handoff paths:\n" + dirty)
    commit = git("rev-parse", "HEAD"); short = commit[:12]
    sub = git("rev-parse", "--show-prefix")              # "Phase4_Multi_Source_3D/": archive that subtree, so the package
    tree = f"HEAD:{sub}" if sub else "HEAD"              # layout (RIND3D_*/ at its root) is unchanged by the Phase4 move;
    top = git("rev-parse", "--show-toplevel")            # run from the top level: pathspecs are then relative to the subtree
    raw = subprocess.run(["git", "-C", top, "archive", "--format=tar", f"--prefix=rind3d_100k_{short}/", tree, "--", *PATHS, *EXCLUDE],
                         capture_output=True, check=True).stdout
    src = tarfile.open(fileobj=io.BytesIO(raw))
    files = {}
    members = []
    for m in src.getmembers():
        data = src.extractfile(m).read() if m.isfile() else None
        if data is not None:
            files[m.name.split("/", 1)[1]] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        members.append((m, data))
    deps = []
    if a.with_deps:
        for name in ("CGAL-6.2.1.tar.xz", "boost-1.92.0-1-x86_64.pkg.tar.zst"):
            p = Path(a.with_deps) / name
            if p.exists():
                data = p.read_bytes(); deps.append((f"deps/{name}", data))
                files[f"deps/{name}"] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    sh = lambda rel: files.get(rel, {}).get("sha256")
    manifest = dict(
        package="rind3d-100k-handoff", created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), commit=commit,
        branch=git("rev-parse", "--abbrev-ref", "HEAD"), remote=git("remote", "get-url", "origin"),
        contract=json.loads((HERE / "contract" / "RIND3D_100K_CONTRACT_v3.json").read_text())["contract"],
        contract_sha256=sh("RIND3D_Dataset100K/contract/RIND3D_100K_CONTRACT_v3.json"),
        contract_v1_sha256=sh("RIND3D_Dataset100K/contract/RIND3D_100K_CONTRACT_v1.json"),
        fixtures_index_sha256=sh("RIND3D_Dataset100K/handoff/fixtures/labels/INDEX.json"),
        fastlabel_fixtures_sha256=sh("RIND3D_Dataset100K/handoff/fixtures/fastlabel_v2/INDEX.json"),
        requirements_lock_sha256=sh("RIND3D_Dataset100K/handoff/env/requirements-lock.txt"),
        engine_version="fastlabel-v3.4 (reference engine: tpe-b4/p2-cpu+g2-gpu-pass-v3.1)", n_files=len(files), total_bytes=sum(v["bytes"] for v in files.values()),
        offline_deps=[d[0] for d in deps], files=files,
        entry_points={"setup": "RIND3D_Dataset100K/handoff/env/setup_env.sh", "readme": "RIND3D_Dataset100K/handoff/README_H20.md",
                      "runbook": "RIND3D_Dataset100K/docs/H20_CANARY_RUNBOOK.md",
                      "probe": "RIND3D_Dataset100K/scripts/probe_hardware.py", "smoke": "RIND3D_Dataset100K/scripts/smoke_test.py",
                      "bench_labels": "RIND3D_Dataset100K/scripts/bench_labels.py", "bench_scaling": "RIND3D_Dataset100K/scripts/bench_scaling.py",
                      "estimate": "RIND3D_Dataset100K/scripts/estimate.py", "launch": "RIND3D_Dataset100K/scripts/run_node.py",
                      "report": "RIND3D_Dataset100K/scripts/report.py", "verify": "RIND3D_Dataset100K/scripts/verify_root.py"})
    mbytes = json.dumps(manifest, indent=1).encode()
    name = f"rind3d_100k_handoff_{short}.tar.gz"
    with tarfile.open(out / name, "w:gz") as tf:
        for m, data in members:
            tf.addfile(m, io.BytesIO(data) if data is not None else None)
        for rel, data in deps + [("HANDOFF_MANIFEST.json", mbytes)]:
            ti = tarfile.TarInfo(f"rind3d_100k_{short}/{rel}"); ti.size = len(data); ti.mtime = int(time.time()); ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(data))
    (out / "HANDOFF_MANIFEST.json").write_bytes(mbytes)
    digest = hashlib.sha256((out / name).read_bytes()).hexdigest()
    (out / f"rind3d_100k_handoff_{short}.sha256").write_text(f"{digest}  {name}\n")
    print(json.dumps(dict(archive=str(out / name), sha256=digest, bytes=(out / name).stat().st_size, commit=commit,
                          files=len(files), offline_deps=[d[0] for d in deps])))


if __name__ == "__main__":
    main()
