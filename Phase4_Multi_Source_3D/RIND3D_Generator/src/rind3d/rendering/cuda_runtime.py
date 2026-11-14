"""Owned CUDA context and audited NVRTC module, without a CUDA Python package.

Availability failures are distinct from compilation/numerical failures. The
backend factory may report and handle the former; it must never hide the latter.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np


class CUDAError(RuntimeError): pass
class CUDAUnavailable(CUDAError): pass


def compiler_path():
    explicit = os.environ.get('RIND_NVRTC_LIBRARY')
    if explicit: return Path(explicit)
    candidates = []
    for root in (Path('/usr/local/cuda/lib64'),Path('/opt/cuda/lib64'),Path('/usr/lib')):
        if root.exists():candidates.extend(root.glob('libnvrtc.so*'))
    for root in (Path('/usr/local'), Path('/opt'), Path.home()/'.cache/uv/archive-v0'):
        if root.exists(): candidates.extend(root.glob('*/nvidia/cu13/lib/libnvrtc.so.13'))
        if root.exists(): candidates.extend(root.glob('*/nvidia/cuda_nvrtc/lib/libnvrtc.so*'))
    candidates.extend(Path('/usr/lib').glob('libnvrtc.so*'))
    if not candidates: raise CUDAUnavailable('NVRTC not found; set RIND_NVRTC_LIBRARY to its absolute path')
    return sorted(candidates)[0]


class Runtime:
    def __init__(self, output=None):
        t = time.perf_counter()
        path = compiler_path()
        self.builtins = [C.CDLL(str(p), mode=C.RTLD_GLOBAL) for p in path.parent.glob('libnvrtc-builtins.so.[0-9]*') if '.alt.' not in p.name]
        try:
            self.nvrtc = C.CDLL(str(path))
            self.driver = C.CDLL('libcuda.so.1')
        except OSError as exc:raise CUDAUnavailable(str(exc)) from exc
        self._signature()
        status=self.driver.cuInit(0)
        if status:raise CUDAUnavailable(f'CUDA initialization unavailable: status {status}')
        count=C.c_int();self.ck('cuDeviceGetCount',C.byref(count))
        if count.value<1:raise CUDAUnavailable('No CUDA device')
        dev = C.c_int(); self.ck('cuDeviceGet', C.byref(dev), 0)
        self.context = C.c_void_p(); self.ck('cuCtxCreate_v2', C.byref(self.context), 0, dev)
        self.ptrs = []
        self.module = C.c_void_p()
        self.functions = {}
        self.timings = {'H2D_seconds': 0., 'D2H_seconds': 0., 'kernel_seconds': 0., 'sync_wall_seconds': 0.}
        self.allocated = 0; self.peak_allocated = 0
        major = C.c_int(); minor = C.c_int(); self.ck('cuDeviceComputeCapability', C.byref(major), C.byref(minor), dev)
        name = C.create_string_buffer(256); self.ck('cuDeviceGetName', name, 256, dev)
        self.major = major.value; self.minor = minor.value
        self.options = [f'--gpu-architecture=compute_{self.major}{self.minor}', '--std=c++17',
                        '--fmad=false', '--ftz=false', '--prec-div=true', '--prec-sqrt=true']
        code = Path(__file__).with_name('cuda_certified.cu').read_bytes()
        ptx, log = self.compile(code)
        if any(x in ptx for x in (b'.ftz', b'.approx', b'.f32')): raise CUDAError('Unapproved PTX arithmetic')
        for token in (b'add.rm.f64', b'add.rp.f64', b'mul.rm.f64', b'mul.rp.f64'):
            if token not in ptx: raise CUDAError('Missing explicit directed arithmetic: '+token.decode())
        self.ptx = C.create_string_buffer(ptx)
        self.ck('cuModuleLoadData', C.byref(self.module), self.ptx)
        for name2 in ('regions', 'packed', 'points', 'arithmetic'):
            f = C.c_void_p(); self.ck('cuModuleGetFunction', C.byref(f), self.module, name2.encode()); self.functions[name2] = f
        self.events = [C.c_void_p(), C.c_void_p()]
        for e in self.events: self.ck('cuEventCreate', C.byref(e), 0)
        v1 = C.c_int(); v2 = C.c_int(); self.nvrtc.nvrtcVersion(C.byref(v1), C.byref(v2))
        dv = C.c_int(); self.ck('cuDriverGetVersion', C.byref(dv))
        self.info = {'device': name.value.decode(), 'compute_capability': [major.value, minor.value],
                     'nvrtc_version': [v1.value, v2.value], 'driver_api_version': dv.value,
                     'nvcc': shutil.which('nvcc'), 'nvrtc_library': str(path),
                     'nvrtc_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'compiler_options': self.options, 'compiler_log': log,
                     'ptx_sha256': hashlib.sha256(ptx).hexdigest(), 'kernel_source_sha256': hashlib.sha256(code).hexdigest(),
                     'initial_memory': self.memory(), 'compile_context_seconds': time.perf_counter()-t}
        if output:
            output = Path(output); output.mkdir(parents=True, exist_ok=True)
            (output/'certified.ptx').write_bytes(ptx)
            (output/'runtime.json').write_text(json.dumps(self.info, indent=2))

    def _signature(self):
        d = self.driver
        signature = {
            'cuCtxCreate_v2': [C.POINTER(C.c_void_p), C.c_uint, C.c_int],
            'cuModuleLoadData': [C.POINTER(C.c_void_p), C.c_void_p],
            'cuModuleGetFunction': [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
            'cuMemAlloc_v2': [C.POINTER(C.c_uint64), C.c_size_t],
            'cuMemFree_v2': [C.c_uint64],
            'cuMemcpyHtoD_v2': [C.c_uint64, C.c_void_p, C.c_size_t],
            'cuMemcpyDtoH_v2': [C.c_void_p, C.c_uint64, C.c_size_t],
            'cuLaunchKernel': [C.c_void_p]+[C.c_uint]*7+[C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p],
            'cuEventRecord': [C.c_void_p, C.c_void_p],
            'cuEventSynchronize': [C.c_void_p],
            'cuEventElapsedTime': [C.POINTER(C.c_float), C.c_void_p, C.c_void_p],
            'cuEventDestroy_v2': [C.c_void_p], 'cuModuleUnload': [C.c_void_p], 'cuCtxDestroy_v2': [C.c_void_p],
        }
        for name, types in signature.items(): getattr(d, name).argtypes = types

    def ck(self, name, *args):
        value = getattr(self.driver, name)(*args)
        if value: raise CUDAError(f'{name}: CUDA status {value}')

    def compile(self, code):
        n = self.nvrtc; p = C.c_void_p()
        n.nvrtcCreateProgram.argtypes = [C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p, C.c_int, C.c_void_p, C.c_void_p]
        if n.nvrtcCreateProgram(C.byref(p), code, b'certified.cu', 0, None, None): raise CUDAError('nvrtcCreateProgram failed')
        try:
            opts = (C.c_char_p*len(self.options))(*(x.encode() for x in self.options))
            n.nvrtcCompileProgram.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)]
            result = n.nvrtcCompileProgram(p, len(opts), opts)
            size = C.c_size_t(); n.nvrtcGetProgramLogSize(p, C.byref(size)); buf = C.create_string_buffer(size.value)
            n.nvrtcGetProgramLog(p, buf); log = buf.value.decode()
            if result: raise CUDAError(f'NVRTC {result}: {log}')
            n.nvrtcGetPTXSize(p, C.byref(size)); buf = C.create_string_buffer(size.value); n.nvrtcGetPTX(p, buf)
            return buf.value, log
        finally: n.nvrtcDestroyProgram(C.byref(p))

    def alloc(self, count):
        p = C.c_uint64(); self.ck('cuMemAlloc_v2', C.byref(p), count)
        self.ptrs.append((p, count)); self.allocated += count; self.peak_allocated = max(self.peak_allocated, self.allocated)
        return p

    def free(self, p):
        self.ck('cuMemFree_v2', p)
        for i, (q, count) in enumerate(self.ptrs):
            if q.value == p.value: self.ptrs.pop(i); self.allocated -= count; return

    def upload(self, a, p=None):
        a = np.ascontiguousarray(a); p = self.alloc(a.nbytes) if p is None else p
        t = time.perf_counter(); self.ck('cuMemcpyHtoD_v2', p, a.ctypes.data, a.nbytes)
        self.timings['H2D_seconds'] += time.perf_counter()-t
        return p

    def download(self, p, shape, dtype):
        a = np.empty(shape, dtype=dtype); t = time.perf_counter()
        self.ck('cuMemcpyDtoH_v2', a.ctypes.data, p, a.nbytes)
        self.timings['D2H_seconds'] += time.perf_counter()-t
        return a

    def launch(self, name, grid, block, args):
        values = [x if isinstance(x, C.c_uint64) else C.c_int(x) for x in args]
        params = (C.c_void_p*len(values))(*(C.addressof(x) for x in values))
        self.ck('cuEventRecord', self.events[0], None)
        self.ck('cuLaunchKernel', self.functions[name], *grid, *block, 0, None, params, None)
        self.ck('cuEventRecord', self.events[1], None)
        t = time.perf_counter(); self.ck('cuEventSynchronize', self.events[1]); self.timings['sync_wall_seconds'] += time.perf_counter()-t
        ms = C.c_float(); self.ck('cuEventElapsedTime', C.byref(ms), *self.events)
        self.timings['kernel_seconds'] += ms.value/1000

    def memory(self):
        free = C.c_size_t(); total = C.c_size_t(); self.ck('cuMemGetInfo_v2', C.byref(free), C.byref(total))
        return {'free_bytes': free.value, 'total_bytes': total.value}

    def close(self):
        if getattr(self, 'context', None) is None: return
        for p, _ in self.ptrs[:]: self.free(p)
        for e in self.events: self.ck('cuEventDestroy_v2', e)
        self.ck('cuModuleUnload', self.module); self.ck('cuCtxDestroy_v2', self.context); self.context = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
