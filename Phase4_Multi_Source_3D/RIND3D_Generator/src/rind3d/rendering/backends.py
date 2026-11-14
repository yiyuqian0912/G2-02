"""Explicit backend capabilities and lifetime; publication is a separate gate."""
from collections import Counter,OrderedDict
import hashlib
import time
import numpy as np
from rind3d.geometry import Oracle,Unsupported,render_region

BACKENDS=('cpu_reference','cpu_native_certified','cuda_certified','auto')

class ReferenceSource:
    backend='cpu_reference'
    def __init__(self,oracle,source,reason=None):
        self.oracle=oracle;self.source=np.array(source,dtype='<f8',copy=True).reshape(3);self.source.setflags(write=False)
        oracle.validate_source(self.source);self.work=Counter();self.setup_seconds=0.;self.reason=reason
        self.source_id=hashlib.sha256(oracle.geometry.content_hash+self.source.tobytes()).hexdigest()
        self.identity={'kind':'original-canonical-reference','backend':oracle.provenance['canonical_backend'],'policy':oracle.policy_hash}
    def __getattr__(self,name):return getattr(self.oracle,name)
    @property
    def metrics(self):return dict(self.work)
    def render(self,origin,size,resolution,*,budget=None,leaf_side=8):
        y,m,count=render_region(self.oracle,self.source,origin,size,resolution,budget=budget,leaf_side=leaf_side)
        self.work['certificate_covered_voxels']+=count;return y,m,count
    def packed_batch(self,origins,side,resolution,*,budget=None):
        result=[]
        for origin in origins:
            y,m,_=self.render(origin,side,resolution,budget=budget);pair=[]
            for a in (y,m):
                pad=np.zeros((64,)*3,dtype=bool);pad[:side,:side,:side]=a;pair.append(np.packbits(pad,bitorder='little').tobytes())
            result.append(tuple(pair))
        return result
    def report(self):return {'backend':self.backend,'identity':self.identity,'fallback_reason':self.reason,'work':dict(self.work)}
    def close(self):pass

class BackendFactory:
    """One producer owns this object; DataLoader workers never create CUDA state."""
    def __init__(self,*,cuda_batch=32,geometry_cache=2,cuda_allowed=True):
        self.cuda_batch=int(cuda_batch);self.geometry_cache=int(geometry_cache)
        self.contexts=OrderedDict();self.device_geometries=OrderedDict();self.runtime=None
        self.cuda_unavailable=None;self.cuda_setup_seconds=0.;self.cuda_selftest=None
        self.cuda_allowed=bool(cuda_allowed)
    def context(self,bundle,policy=None):
        try:from .context import GeometryContext
        except ImportError as exc:raise Unsupported('Certified native extension is not installed') from exc
        from rind3d.geometry import physics_provenance
        import json
        key=(bundle.geometries[0].content_hash,hashlib.sha256(json.dumps(physics_provenance(policy),sort_keys=True).encode()).hexdigest())
        if key in self.contexts:
            value=self.contexts.pop(key);self.contexts[key]=value;return value
        value=GeometryContext(bundle,policy);self.contexts[key]=value
        while len(self.contexts)>self.geometry_cache:self.contexts.popitem(last=False)
        return value
    def _cuda(self):
        if not self.cuda_allowed:self.cuda_unavailable='CUDA fixture certification is not enabled for this profile build';return None
        if self.cuda_unavailable is not None:return None
        if self.runtime is None:
            from .cuda_runtime import Runtime,CUDAUnavailable
            from .cuda_backend import validate_arithmetic
            start=time.perf_counter()
            try:self.runtime=Runtime()
            except CUDAUnavailable as exc:self.cuda_unavailable=str(exc);return None
            self.cuda_selftest=validate_arithmetic(self.runtime);self.cuda_setup_seconds=time.perf_counter()-start
        return self.runtime
    def source(self,bundle,source,backend='auto',policy=None):
        if backend not in BACKENDS:raise ValueError('Unknown renderer backend')
        if backend=='cpu_reference':return ReferenceSource(Oracle(bundle,policy),source)
        try:context=self.context(bundle,policy)
        except Unsupported as exc:
            return ReferenceSource(Oracle(bundle,policy),source,'unsupported native scope: '+str(exc))
        if backend=='cpu_native_certified':return context.bind(source)
        runtime=self._cuda()
        if runtime is None:
            if backend=='cuda_certified':
                from .cuda_runtime import CUDAUnavailable
                raise CUDAUnavailable(self.cuda_unavailable or 'CUDA unavailable; use auto or cpu_native_certified')
            item=context.bind(source);item.selection_reason='CUDA unavailable: '+str(self.cuda_unavailable);return item
        from .cuda_backend import DeviceGeometry
        key=(context.geometry_id,context.policy_hash)
        if key not in self.device_geometries:self.device_geometries[key]=DeviceGeometry(context,runtime,self.cuda_batch)
        else:self.device_geometries.move_to_end(key)
        while len(self.device_geometries)>self.geometry_cache:
            _,old=self.device_geometries.popitem(last=False);old.close()
        return self.device_geometries[key].bind(source)
    def report(self):
        return {'cuda_initialized':self.runtime is not None,'cuda_unavailable':self.cuda_unavailable,
                'cuda_cold_setup_seconds':self.cuda_setup_seconds,'cuda_selftest':self.cuda_selftest,
                'cuda_runtime':self.runtime.info if self.runtime else None}
    def close(self):
        for item in self.device_geometries.values():item.close()
        self.device_geometries.clear()
        if self.runtime:self.runtime.close();self.runtime=None
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
