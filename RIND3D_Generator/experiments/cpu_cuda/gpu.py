"""Certified GPU halfspaces, packed output, canonical CPU UNKNOWN fallback."""
from collections import Counter
from fractions import Fraction
import hashlib
import time
import numpy as np
from rind3d.geometry import centres
from experiments.shadow_volumes.renderer import shadow_planes


def enclose(q):
    q = Fraction(q); x = float(q)
    if not np.isfinite(x): raise ValueError('Nonfinite rational conversion')
    lo = hi = x
    while Fraction(lo) > q: lo = np.nextafter(lo, -np.inf)
    while Fraction(hi) < q: hi = np.nextafter(hi, np.inf)
    if not Fraction(lo) <= q <= Fraction(hi): raise AssertionError('Invalid coefficient enclosure')
    return lo, hi


def encode(bodies):
    rows = []; offsets = [0]
    for body in bodies:
        for row in body: rows.append([v for q in row for v in enclose(q)])
        offsets.append(len(rows))
    return np.asarray(rows, dtype='<f8').reshape(-1, 8), np.asarray(offsets, dtype='<i4')


class GPUGroup:
    def __init__(self, shared, runtime, batch=32):
        t = time.perf_counter(); self.shared = shared; self.runtime = runtime; self.batch = batch
        g, offsets = encode(shared.bodies); self.rows = runtime.upload(g); self.offsets = runtime.upload(offsets)
        self.origins = runtime.alloc(batch*3*4); self.labels = runtime.alloc(batch*585*2)
        self.output = [runtime.alloc(batch*32768) for _ in range(4)]
        self.setup_seconds = time.perf_counter()-t
        self.solid_cache = {}  # Optional explicitly enabled packed cache; <=128 MiB at 1024³.
        self.cache_resolution = None

    def bind(self, source, *, reuse_mask=False, fallback_backend='canonical_cgal'):
        return GPUOracle(self, source, reuse_mask=reuse_mask, fallback_backend=fallback_backend)

    def close(self):
        for p in [self.rows, self.offsets, self.origins, self.labels]+self.output: self.runtime.free(p)
        self.solid_cache.clear()


class GPUOracle:
    def __init__(self, group, source, *, reuse_mask=False, fallback_backend='canonical_cgal'):
        t = time.perf_counter(); self.group = group; self.shared = group.shared; self.runtime = group.runtime
        self.source = np.array(source, dtype='<f8', copy=True).reshape(3); self.source.setflags(write=False)
        self.cpu = self.shared.bind(self.source)  # Exact free-source validation.
        self.source_id = self.cpu.source_id; self.work = Counter(); self.reuse_mask = reuse_mask
        if fallback_backend not in ('canonical_cgal','certified_native'):raise ValueError('Unknown fallback backend')
        self.fallback_backend=fallback_backend;self.independent_reference=None
        bodies = [shadow_planes(b, self.source) for b in self.shared.bodies]
        rows, offsets = encode(bodies)
        self.rows = self.runtime.upload(rows); self.offsets = self.runtime.upload(offsets)
        self.setup_seconds = time.perf_counter()-t; self.shadow_plane_counts = [len(b) for b in bodies]

    def __getattr__(self, name): return getattr(self.cpu, name)

    def close(self):
        self.runtime.free(self.rows); self.runtime.free(self.offsets)

    def attach_independent_reference(self, control):
        if control.shared.geometry_id!=self.shared.geometry_id or not np.array_equal(control.source,self.source):
            raise ValueError('Independent fallback reference identity mismatch')
        self.independent_reference=control.native

    def fallback(self, points, which):
        if not len(points): return np.empty(0, dtype=bool)
        if self.fallback_backend=='certified_native':
            # Separate CPU exact predicate, NOT a GPU certificate or a claim
            # that the original CGAL ran. GeometryGroup proves its scope:
            # exact inverse, strict interior/world witness, bounded full-
            # dimensional convex components, top union. Intervals + GMP.
            t=time.perf_counter();y,m,stats=self.cpu.native.points(np.ascontiguousarray(points,dtype='<f8'))
            values=(y,m)[which]
            if np.any(values<0):raise AssertionError('Unresolved native exact fallback')
            self.work['native_fallback_rays' if which==0 else 'native_fallback_points']+=len(points)
            self.work['native_exact_fallback_operations']+=stats['exact_fallbacks']
            self.work['native_fallback_seconds']+=time.perf_counter()-t
            if self.independent_reference is not None:
                t=time.perf_counter();ry,rm,reference_stats=self.independent_reference.points(np.ascontiguousarray(points,dtype='<f8'),True)
                if not np.array_equal(values,(ry,rm)[which]):raise AssertionError('Independent fallback reference disagreed')
                self.work['independent_fallback_points_checked']+=len(points)
                self.work['independent_reference_exact_clips']+=reference_stats['reference_exact_clips']
                self.work['independent_fallback_seconds']+=time.perf_counter()-t
            return values.astype(bool)
        t = time.perf_counter(); oracle = self.shared.canonical
        # Direct canonical calls: do not silently replace fallback with this
        # experiment's interval implementation or treat UNKNOWN as physical MISS.
        if which == 0:
            result = oracle.reference.segment_intersect(np.repeat(self.source[None], len(points), axis=0), points)
            values = oracle._values(result) == 0
        else:
            values = oracle._values(oracle.reference.classify_points(points)) != 0
        self.work['canonical_fallback_rays' if which == 0 else 'canonical_fallback_points'] += len(points)
        self.work['canonical_fallback_seconds'] += time.perf_counter()-t
        return values

    def point_labels(self, points, *, resolve=True):
        p = np.ascontiguousarray(points, dtype='<f8').reshape(-1, 3)
        if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1): raise ValueError('Invalid point coordinates')
        r = self.runtime; device = [r.upload(p), r.alloc(len(p)), r.alloc(len(p))]
        try:
            r.launch('points', ((len(p)+255)//256,1,1),(256,1,1),
                     [self.group.rows,self.group.offsets,self.rows,self.offsets,len(self.shared.bodies),device[0],len(p),device[1],device[2]])
            out = [r.download(d, len(p), np.int8) for d in device[1:]]
            if resolve:
                for i, a in enumerate(out):
                    unknown = np.flatnonzero(a<0)
                    if len(unknown): a[unknown] = self.fallback(p[unknown], i)
            return tuple(out)
        finally:
            for d in device: r.free(d)

    def packed_batch(self, origins, side, n):
        origins = np.ascontiguousarray(origins, dtype='<i4').reshape(-1,3); count = len(origins)
        if not 0<count<=self.group.batch or side not in (32,64) or n<side or n>1024 or n&(n-1): raise ValueError('Invalid GPU raster batch')
        if np.any(origins<0) or np.any(origins+side>n): raise ValueError('Invalid GPU origins')
        r = self.runtime; g = self.group
        if g.cache_resolution is not None and g.cache_resolution != n: g.solid_cache.clear()
        g.cache_resolution = n
        skip = self.reuse_mask and all(tuple(o) in g.solid_cache for o in origins)
        r.upload(origins, g.origins)
        last = int(np.log2(side//8))
        for level in range(last+1):
            nodes = 8**level
            r.launch('regions',((nodes+127)//128,count,1),(128,1,1),
                     [g.rows,g.offsets,self.rows,self.offsets,len(self.shared.bodies),g.origins,count,side,n,level,g.labels,int(skip)])
        r.launch('packed',(128,count,1),(256,1,1),
                 [g.rows,g.offsets,self.rows,self.offsets,len(self.shared.bodies),g.origins,count,side,n,last,g.labels,*g.output,int(skip)])
        values = [r.download(d,(count,32768),np.uint8) for d in g.output]
        labels = r.download(g.labels,(count,585,2),np.int8)
        covered = 0
        # Count certified final leaves, not inflated parent+child totals.
        leaves = 8**last; offset = (leaves-1)//7
        for i, o in enumerate(origins):
            if skip: values[1][i] = np.frombuffer(g.solid_cache[tuple(o)],dtype=np.uint8)
            for which in (0,1):
                bad = np.flatnonzero(np.unpackbits(values[which+2][i],bitorder='little'))
                if len(bad):
                    z,y,x = np.unravel_index(bad,(64,)*3)
                    points = (np.stack([x,y,z],axis=1)+o+.5)/n
                    bits = np.unpackbits(values[which][i],bitorder='little')
                    bits[bad] = self.fallback(points,which)
                    values[which][i] = np.packbits(bits,bitorder='little')
            if np.any(values[0][i]&values[1][i]): raise AssertionError('GPU solid receiver lit')
            if self.reuse_mask: g.solid_cache[tuple(o)] = values[1][i].tobytes()
            covered += int(np.sum(np.all(labels[i,offset:offset+leaves]>=0,axis=1)))*8**3
        self.work['certificate_covered_voxels'] += covered
        self.work['voxel_count'] += count*side**3; self.work['bricks'] += count
        self.work['mask_bricks_reused'] += count if skip else 0
        return [(values[0][i].tobytes(),values[1][i].tobytes()) for i in range(count)]

    def report(self):
        return {'version':'cuda-certified-interval-shadow-v2','production_ready':False,
                'geometry_id':self.shared.geometry_id,'source_id':self.source_id,
                'fallback_backend':self.fallback_backend,'independent_fallback_reference_attached':self.independent_reference is not None,
                'source_setup_seconds':self.setup_seconds,'shadow_plane_counts':self.shadow_plane_counts,
                'work':dict(self.work),'canonical_initialization_seconds':self.shared.initialization_seconds,
                'runtime_device_allocated_peak':self.runtime.peak_allocated,
                'packed_mask_host_cache_bytes':sum(len(x) for x in self.group.solid_cache.values())}
