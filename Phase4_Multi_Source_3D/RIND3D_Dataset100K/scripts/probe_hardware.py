#!/usr/bin/env python3
"""Server hardware and software probe for RIND3D-100K. Writes a JSON report (stdout and --out).

  probe_hardware.py --out probe.json [--paths /data/r3d,/scratch] [--io-mb 2048] [--bench] [--gpus 0,1]

Reports:
* CPU: model, logical CPUs, affinity, cgroup quota, NUMA nodes and their CPUs;
* RAM: total, available, cgroup limit, swap;
* GPUs: model, memory, compute capability, driver, clocks, PCIe, MIG mode, NUMA affinity, `nvidia-smi topo -m`;
* storage: df of each path, and with --io-mb sequential fsync write and read throughput;
* schedulers: Slurm, PBS, LSF, Flux, plus active job variables;
* software: Python, numpy, numba, cupy, CUDA runtime and driver versions, NVRTC library, rind3d native modules.
With --bench, per GPU: FP64 and FP32 FMA throughput (cupy RawKernel loops), device memory bandwidth, and the certified
tile renderer on a synthetic scene. The labeller benchmark is separate (bench_labels.py) because it needs the
fixtures."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sh(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def cpu_info():
    d = dict(logical=os.cpu_count(), affinity=len(os.sched_getaffinity(0)), machine=platform.machine())
    lscpu = sh("lscpu")
    if lscpu:
        for line in lscpu.splitlines():
            k, _, v = line.partition(":")
            if k.strip() in ("Model name", "Socket(s)", "Core(s) per socket", "Thread(s) per core", "NUMA node(s)", "L3 cache"):
                d[k.strip()] = v.strip()
    nodes = {}
    base = Path("/sys/devices/system/node")
    if base.exists():
        for n in sorted(base.glob("node[0-9]*")):
            try:
                nodes[n.name] = (n / "cpulist").read_text().strip()
            except OSError:
                pass
    d["numa_cpulists"] = nodes
    try:
        q = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        d["cgroup_cpu_max"] = None if q[0] == "max" else int(q[0]) / int(q[1])
    except Exception:
        d["cgroup_cpu_max"] = None
    return d


def mem_info():
    d = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            d[k] = int(v.split()[0]) * 1024
    try:
        v = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        d["cgroup_memory_max"] = None if v == "max" else int(v)
    except Exception:
        d["cgroup_memory_max"] = None
    return d


def gpu_info():
    q = ("index,name,uuid,memory.total,memory.used,compute_cap,driver_version,clocks.max.sm,clocks.max.mem,power.limit,"
         "pcie.link.gen.max,pcie.link.width.max,mig.mode.current,persistence_mode")
    out = sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader")
    gpus = []
    if out:
        keys = q.split(",")
        for line in out.splitlines():
            gpus.append(dict(zip(keys, [x.strip() for x in line.split(",")])))
    return dict(gpus=gpus, topology=sh("nvidia-smi topo -m"), smi_version=sh("nvidia-smi --version"))


def storage_info(paths, io_mb):
    out = {}
    for p in paths:
        p = Path(p)
        try:
            p.mkdir(parents=True, exist_ok=True)
            du = shutil.disk_usage(p)
            d = dict(total=du.total, free=du.free, fs=sh(f"df -T {p} | tail -1"))
            if io_mb:
                f = p / f".r3d_iotest_{os.getpid()}"
                buf = os.urandom(1 << 20)
                t = time.perf_counter()
                with open(f, "wb") as fh:
                    for _ in range(io_mb):
                        fh.write(buf)
                    fh.flush(); os.fsync(fh.fileno())
                d["write_MBps"] = io_mb / (time.perf_counter() - t)
                t = time.perf_counter()
                with open(f, "rb") as fh:
                    while fh.read(1 << 24):
                        pass
                d["read_MBps_cached"] = io_mb / (time.perf_counter() - t)
                f.unlink()
            out[str(p)] = d
        except Exception as e:
            out[str(p)] = dict(error=repr(e))
    return out


def software_info():
    d = dict(python=sys.version.split()[0], executable=sys.executable)
    for mod in ("numpy", "numba", "cupy", "zstandard", "z3"):
        try:
            m = __import__(mod); d[mod] = getattr(m, "__version__", None) or getattr(m, "get_version_string", lambda: "?")()
        except Exception as e:
            d[mod] = f"unavailable: {e!r}"[:200]
    try:
        import cupy as cp
        d["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion(); d["cuda_driver"] = cp.cuda.runtime.driverGetVersion()
        d["cupy_devices"] = cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        d["cuda"] = f"unavailable: {e!r}"[:200]
    try:
        import rind._exact2d, rind._exact3d, rind3d._native, rind3d._convex, rind3d._research  # noqa
        d["rind3d_natives"] = "ok"
    except Exception as e:
        d["rind3d_natives"] = f"missing: {e!r}"[:200]
    d["RIND_NVRTC_LIBRARY"] = os.environ.get("RIND_NVRTC_LIBRARY")
    d["nvcc"] = sh("nvcc --version | tail -1"); d["gcc"] = sh("gcc --version | head -1"); d["cmake"] = sh("cmake --version | head -1")
    return d


def scheduler_info():
    tools = {t: shutil.which(t) for t in ("sbatch", "srun", "squeue", "sinfo", "qsub", "bsub", "flux", "apptainer", "singularity", "docker", "podman")}
    env = {k: v for k, v in os.environ.items() if k.startswith(("SLURM_", "PBS_", "LSB_", "FLUX_"))}
    return dict(tools=tools, job_env=env, sinfo=sh("sinfo -s") if tools["sinfo"] else None,
                ulimit=sh("ulimit -a"))


FMA_SRC = r'''
extern "C" __global__ void fma64(double* x, int iters){ int i=blockIdx.x*blockDim.x+threadIdx.x; double a=x[i], b=1.0000001, c=1e-9;
  for(int k=0;k<iters;k++){ a=fma(a,b,c); a=fma(a,b,c); a=fma(a,b,c); a=fma(a,b,c);} x[i]=a; }
extern "C" __global__ void fma32(float* x, int iters){ int i=blockIdx.x*blockDim.x+threadIdx.x; float a=x[i], b=1.0000001f, c=1e-9f;
  for(int k=0;k<iters;k++){ a=fmaf(a,b,c); a=fmaf(a,b,c); a=fmaf(a,b,c); a=fmaf(a,b,c);} x[i]=a; }
'''


def gpu_bench(dev):
    import cupy as cp
    out = {}
    with cp.cuda.Device(dev):
        mod = cp.RawModule(code=FMA_SRC)
        props = cp.cuda.runtime.getDeviceProperties(dev)
        n = props["multiProcessorCount"] * 2048
        for name, dt in (("fma64", cp.float64), ("fma32", cp.float32)):
            x = cp.ones(n, dt); f = mod.get_function(name); iters = 2000 if dt == cp.float64 else 8000
            f((n // 256,), (256,), (x, cp.int32(10))); cp.cuda.Device().synchronize()
            t = time.perf_counter(); f((n // 256,), (256,), (x, cp.int32(iters))); cp.cuda.Device().synchronize()
            dtt = time.perf_counter() - t
            out[name + "_TFLOPS"] = 2 * 4 * iters * n / dtt / 1e12
        a = cp.empty(1 << 28, cp.uint8); b = cp.empty_like(a)
        cp.copyto(b, a); cp.cuda.Device().synchronize()
        t = time.perf_counter()
        for _ in range(10):
            cp.copyto(b, a)
        cp.cuda.Device().synchronize()
        out["copy_GBps"] = 2 * 10 * a.nbytes / (time.perf_counter() - t) / 1e9
        out["name"] = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        out["sm_count"] = props["multiProcessorCount"]; out["cc"] = f'{props["major"]}.{props["minor"]}'
    return out


def renderer_bench():
    """Certified tile renderer throughput on a synthetic scene (child process; CUDA_VISIBLE_DEVICES selects the GPU)."""
    here = Path(__file__).resolve().parent.parent
    sys.path[:0] = [str(here), str(here.parent / "RIND3D_Dataset1K")]
    import numpy as np
    from rind3d.rendering.backends import BackendFactory
    from rind3d_1k.families import generate
    from rind3d_1k.physical import POLICY, sample_sources
    bundle, _ = generate(20261215, 7, "occluder_clusters", 0)
    fac = BackendFactory(cuda_batch=32); ctx = fac.context(bundle, POLICY)
    S, _ = sample_sources(ctx, 20261215, 7, 1, 0)
    t = time.perf_counter(); src = fac.source(bundle, S[0], "cuda_certified", POLICY); setup = time.perf_counter() - t
    rng = np.random.default_rng(0)
    src.packed_batch(rng.integers(0, 960, (32, 3)), 64, 1024)
    t = time.perf_counter(); n = 0
    for _ in range(8):
        src.packed_batch(rng.integers(0, 960, (32, 3)), 64, 1024); n += 32
    dt = time.perf_counter() - t
    pts = rng.random((20000, 3)); t = time.perf_counter(); src.cpu.native.points(np.ascontiguousarray(pts), False); dp = time.perf_counter() - t
    return dict(setup_s=setup, tile64_ms=1e3 * dt / n, voxels_per_s=n * 64 ** 3 / dt, cpu_exact_point_us=1e6 * dp / len(pts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out"); ap.add_argument("--paths", default=""); ap.add_argument("--io-mb", type=int, default=0)
    ap.add_argument("--bench", action="store_true"); ap.add_argument("--gpus", default="")
    ap.add_argument("--_renderer", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._renderer:
        print(json.dumps(renderer_bench())); return
    rep = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), host=platform.node(), os=platform.platform(),
               cpu=cpu_info(), memory=mem_info(), gpu=gpu_info(), software=software_info(), schedulers=scheduler_info(),
               storage=storage_info([p for p in a.paths.split(",") if p], a.io_mb))
    if a.bench:
        devs = [int(x) for x in a.gpus.split(",") if x != ""] or list(range(len(rep["gpu"]["gpus"])))
        rep["gpu_bench"] = {}
        for dv in devs:
            try:
                rep["gpu_bench"][dv] = gpu_bench(dv)
            except Exception as e:
                rep["gpu_bench"][dv] = dict(error=repr(e)[:300])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(dv))
            r = subprocess.run([sys.executable, __file__, "--_renderer"], capture_output=True, text=True, env=env, timeout=600)
            try:
                rep["gpu_bench"][dv]["renderer"] = json.loads(r.stdout.strip().splitlines()[-1])
            except Exception:
                rep["gpu_bench"][dv]["renderer"] = dict(error=(r.stderr or r.stdout)[-400:])
    s = json.dumps(rep, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(s)
    print(s)


if __name__ == "__main__":
    main()
