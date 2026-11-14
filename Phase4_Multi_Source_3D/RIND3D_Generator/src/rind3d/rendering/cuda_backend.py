"""Bounded packed CUDA batches with proved native or original-oracle fallback."""
from collections import Counter
from fractions import Fraction as Q
import hashlib
import time
import numpy as np
from .cuda_runtime import Runtime,CUDAError,CUDAUnavailable
from .polyhedra import shadow_planes
from .context import backend_identity

def enclose(value):
    q=Q(value);x=float(q)
    if not np.isfinite(x):raise ValueError('Nonfinite rational coefficient')
    lo=hi=x
    while Q(lo)>q:lo=float(np.nextafter(lo,-np.inf))
    while Q(hi)<q:hi=float(np.nextafter(hi,np.inf))
    if not Q(lo)<=q<=Q(hi):raise AssertionError('Coefficient enclosure failed')
    return lo,hi

def encode(bodies):
    rows=[];offsets=[0]
    for body in bodies:
        rows.extend([[v for q in row for v in enclose(q)] for row in body]);offsets.append(len(rows))
    return np.asarray(rows,dtype='<f8').reshape(-1,8),np.asarray(offsets,dtype='<i4')

def validate_arithmetic(runtime):
    """Real-device enclosure check; never a CPU substitute for a CUDA test."""
    rng=np.random.default_rng(882)
    a=np.r_[0.,-0.,np.nextafter(0.,1.),-np.nextafter(0.,1.),2.**-1022,2.**1023,np.ldexp(rng.uniform(-1,1,1024),rng.integers(-1070,1023,1024))]
    b=np.r_[0.,-1.,.5,.5,2.**-100,2.,np.ldexp(rng.uniform(-1,1,1024),rng.integers(-1070,1023,1024))]
    buffers=[runtime.upload(a),runtime.upload(b),runtime.alloc(len(a)*4*8)]
    try:
        runtime.launch('arithmetic',((len(a)+255)//256,1,1),(256,1,1),[buffers[0],buffers[1],len(a),buffers[2]])
        values=runtime.download(buffers[2],(len(a),4),np.float64)
        for i,(x,y) in enumerate(zip(a,b)):
            for lo,q,hi in ((values[i,0],Q(x)+Q(y),values[i,1]),(values[i,2],Q(x)*Q(y),values[i,3])):
                if not (lo==-np.inf or Q(lo)<=q) or not (hi==np.inf or q<=Q(hi)):
                    raise CUDAError('Real-device directed arithmetic certification failed')
    finally:
        for ptr in buffers:runtime.free(ptr)
    return {'tested_pairs':len(a),'passed':True,'subnormals':True,'overflow':True}

class DeviceGeometry:
    def __init__(self,context,runtime,batch=32):
        if not 1<=batch<=128:raise ValueError('CUDA batch must be 1..128')
        self.context=context;self.runtime=runtime;self.batch=batch
        rows,offsets=encode(context.bodies);self.rows=runtime.upload(rows);self.offsets=runtime.upload(offsets)
        self.origins=runtime.alloc(batch*12);self.labels=runtime.alloc(batch*585*2)
        self.output=[runtime.alloc(batch*32768) for _ in range(4)]
    def bind(self,source):return CUDASource(self,source)
    def close(self):
        for p in [self.rows,self.offsets,self.origins,self.labels,*self.output]:self.runtime.free(p)

class CUDASource:
    backend='cuda_certified'
    def __init__(self,device_geometry,source):
        start=time.perf_counter();self.device=device_geometry;self.context=device_geometry.context;self.runtime=device_geometry.runtime
        self.cpu=self.context.bind(source);self.source=self.cpu.source;self.source_id=self.cpu.source_id
        self.work=Counter();self.closed=False
        planes=[shadow_planes(b,self.source) for b in self.context.bodies];rows,offsets=encode(planes)
        self.rows=self.runtime.upload(rows);self.offsets=self.runtime.upload(offsets)
        self.identity={**backend_identity('cuda-directed-double'),'kernel_sha256':self.runtime.info['kernel_source_sha256'],
                       'ptx_sha256':self.runtime.info['ptx_sha256'],'compiler_options':self.runtime.options,
                       'fallback':'proved-convex-interval-GMP'}
        self.setup_seconds=time.perf_counter()-start
    def __getattr__(self,name):return getattr(self.cpu,name)
    @property
    def metrics(self):return dict(self.work)
    def close(self):
        if not self.closed:self.runtime.free(self.rows);self.runtime.free(self.offsets);self.closed=True
    def packed_batch(self,origins,side,resolution,*,budget=None):
        if self.closed:raise RuntimeError('CUDA source is closed')
        if budget:budget.check()
        origins=np.ascontiguousarray(origins,dtype='<i4').reshape(-1,3);count=len(origins);g=self.device;r=self.runtime;n=resolution
        if not 0<count<=g.batch or side not in (32,64) or n<side or n>1024 or n&(n-1):raise ValueError('Invalid CUDA raster extent')
        if np.any(origins<0) or np.any(origins+side>n):raise ValueError('Invalid CUDA origins')
        r.upload(origins,g.origins);last=int(np.log2(side//8));body_count=len(self.context.bodies)
        for level in range(last+1):
            r.launch('regions',((8**level+127)//128,count,1),(128,1,1),[g.rows,g.offsets,self.rows,self.offsets,body_count,g.origins,count,side,n,level,g.labels,0])
        r.launch('packed',(128,count,1),(256,1,1),[g.rows,g.offsets,self.rows,self.offsets,body_count,g.origins,count,side,n,last,g.labels,*g.output,0])
        values=[r.download(p,(count,32768),np.uint8) for p in g.output];labels=r.download(g.labels,(count,585,2),np.int8)
        leaves=8**last;offset=(leaves-1)//7;result=[]
        for i,origin in enumerate(origins):
            for which in (0,1):
                # Exact zero in the UNKNOWN bitset proves no CPU work is needed.
                if not np.any(values[which+2][i]):continue
                bad=np.flatnonzero(np.unpackbits(values[which+2][i],bitorder='little'))
                z,y,x=np.unravel_index(bad,(64,)*3);points=(np.stack([x,y,z],axis=1)+origin+.5)/n
                start=time.perf_counter();yy,mm,stats=self.cpu.native.points(np.ascontiguousarray(points,dtype='<f8'),False)
                actual=(yy,mm)[which]
                if np.any(actual<0):raise RuntimeError('Native exact UNKNOWN fallback did not certify')
                self.work['fallback_seconds']+=time.perf_counter()-start
                self.work['fallback_rays' if which==0 else 'fallback_points']+=len(bad)
                self.work['GMP_fallback_operations']+=stats['exact_fallbacks']
                if budget:budget.check(len(points))
                bits=np.unpackbits(values[which][i],bitorder='little');bits[bad]=actual
                values[which][i]=np.packbits(bits,bitorder='little')
            if np.any(values[0][i]&values[1][i]):raise AssertionError('CUDA solid receiver lit')
            result.append((values[0][i].tobytes(),values[1][i].tobytes()))
        self.work['bricks']+=count;self.work['voxels']+=count*side**3
        self.work['certificate_covered_voxels']+=int(np.sum(np.all(labels[:,offset:offset+leaves]>=0,axis=2)))*8**3
        if budget:budget.check()
        return result
    def report(self):
        return {'backend':self.backend,'identity':self.identity,'geometry_id':self.context.geometry_id,'source_id':self.source_id,
                'source_setup_seconds':self.setup_seconds,'work':dict(self.work),'runtime':self.runtime.info,
                'device_allocated_peak_bytes':self.runtime.peak_allocated}
