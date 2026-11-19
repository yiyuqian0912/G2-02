#!/usr/bin/env python3
"""RIND3D-100K node launcher (physical scenes and/or labels on one machine with 1..N GPUs). Resumable.

Examples:
  run_node.py --root /scratch/r3d100k --slots 0:64 --gpus all                # canary, everything
  run_node.py --root R --slots 0:100000 --gpus 0-15 --label-cpu auto --rss-gb 6
  run_node.py --root R --no-labels --slots 0:2000                            # physical only
  run_node.py --root R --no-physical --label-slots 0:2000                    # labels only
Stop safely:  touch R/STOP   (or SIGTERM). Resume: rerun the same command (remove STOP first).
Options can also be given in a JSON config file (--config), command-line flags override it."""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "RIND3D_Dataset1K"), str(REPO / "RIND3D_Posterior" / "src")]


def parse_range(s, default):
    if not s:
        return default
    a, b = s.split(":")
    return int(a), int(b)


def parse_gpus(s):
    if s in (None, "", "all"):
        return None
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def default_env(root):
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(root) / ".cache" / "numba"))
    os.environ.setdefault("CUPY_CACHE_DIR", str(Path(root) / ".cache" / "cupy"))
    if not os.environ.get("RIND_NVRTC_LIBRARY"):
        import site
        for sp in site.getsitepackages():
            for c in sorted(Path(sp).glob("nvidia/cuda_nvrtc/lib/libnvrtc.so*")):
                if ".alt." not in c.name:
                    os.environ["RIND_NVRTC_LIBRARY"] = str(c)
                    break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--contract", default=str(HERE / "contract" / "RIND3D_100K_CONTRACT_v3.json"),
                    help="contract JSON (default: v3, the production contract; v2 and v1 for regression and "
                         "reproducibility only)")
    ap.add_argument("--config")
    ap.add_argument("--slots"); ap.add_argument("--label-slots")
    ap.add_argument("--gpus"); ap.add_argument("--phys-per-gpu", type=int); ap.add_argument("--label-gpu-per-gpu", type=int)
    ap.add_argument("--label-cpu"); ap.add_argument("--reserve-cpus", type=int); ap.add_argument("--rss-gb", type=float)
    ap.add_argument("--gpu-cap", type=int); ap.add_argument("--nice", type=int)
    ap.add_argument("--no-labels", action="store_true"); ap.add_argument("--no-physical", action="store_true")
    ap.add_argument("--retry-failed", action="store_true"); ap.add_argument("--max-hours", type=float)
    ap.add_argument("--stop-grace", type=float); ap.add_argument("--phys-timeout", type=float)
    ap.add_argument("--prior-manifest"); ap.add_argument("--epoch-scenes", type=int); ap.add_argument("--progress-s", type=float)
    ap.add_argument("--label-sample", help="benchmark only: n64,n128,n256,seed stratified sample of label tasks")
    ap.add_argument("--label-engine", choices=("tpe", "fastlabel"),
                    help="label engine (default: the contract's labels.engine; v2: fastlabel, v1: tpe)")
    ap.add_argument("--fastlabel-n-tasks", type=int, help="planned label count of the campaign (delta = 0.01 / n)")
    ap.add_argument("--fastlabel-base-seed", type=int, help="base seed of the FastLabel label streams")
    a = ap.parse_args()
    conf = json.loads(Path(a.config).read_text()) if a.config else {}
    contract = json.loads(Path(conf.get("contract", a.contract)).read_text())
    if a.epoch_scenes or conf.get("epoch_scenes"):
        contract["planning" if "planning" in contract else "controller"]["epoch_scenes"] = int(a.epoch_scenes or conf["epoch_scenes"])
    slots = parse_range(a.slots or conf.get("slots"), (0, int(contract["n_scenes"])))
    o = dict(slots=slots, gpus=parse_gpus(a.gpus or conf.get("gpus")),
             phys_per_gpu=a.phys_per_gpu if a.phys_per_gpu is not None else conf.get("phys_per_gpu", 1),
             label_gpu_per_gpu=a.label_gpu_per_gpu if a.label_gpu_per_gpu is not None else conf.get("label_gpu_per_gpu", 2),
             label_cpu=a.label_cpu or conf.get("label_cpu", "auto"),
             reserve_cpus=a.reserve_cpus if a.reserve_cpus is not None else conf.get("reserve_cpus", 2),
             rss_gb=a.rss_gb or conf.get("rss_gb", 6.0), gpu_cap=a.gpu_cap or conf.get("gpu_cap"),
             nice=a.nice if a.nice is not None else conf.get("nice", 0),
             labels=not (a.no_labels or conf.get("no_labels")), physical=not (a.no_physical or conf.get("no_physical")),
             retry_failed=a.retry_failed or conf.get("retry_failed", False), max_hours=a.max_hours or conf.get("max_hours", 1e9),
             stop_grace=a.stop_grace or conf.get("stop_grace", 600), phys_timeout=a.phys_timeout or conf.get("phys_timeout", 1800),
             prior_manifest=a.prior_manifest or conf.get("prior_manifest"), progress_s=a.progress_s or conf.get("progress_s", 30),
             label_slots=parse_range(a.label_slots or conf.get("label_slots"), None),
             label_sample=[int(x) for x in a.label_sample.split(",")] if a.label_sample else None,
             label_engine=a.label_engine or conf.get("label_engine"),
             fastlabel={k: v for k, v in dict(n_tasks=a.fastlabel_n_tasks or conf.get("fastlabel_n_tasks"),
                                              base_seed=a.fastlabel_base_seed or conf.get("fastlabel_base_seed")).items()
                        if v is not None})
    from rind3d_100k.profile import check_certified_profile
    check_certified_profile(contract, a.label_engine or conf.get("label_engine"))   # curved / unknown families refused
    default_env(a.root)
    from rind3d_100k.coordinator import Coordinator
    sys.exit(Coordinator(a.root, contract, o).run())


if __name__ == "__main__":
    main()
