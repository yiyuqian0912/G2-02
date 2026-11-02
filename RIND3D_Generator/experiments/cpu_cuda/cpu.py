"""Cached exact polyhedral CPU path, with lazy canonical fallback.

Only strict-interior, bounded convex components in a top-level union are in
scope. This does not supply the production geometry acceptance/clearance gate.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import threading
import time
import numpy as np
from rind3d.geometry import Oracle, POLICY, Unsupported
from experiments.shadow_volumes.renderer import extract, strings, shadow_planes
from . import _efficient_native as native


class LazyCanonical:
    def __init__(self, bundle):
        self.bundle = bundle
        self.geometry = bundle.geometries[0]
        self.policy = {}
        self.lock = threading.RLock()
        self._oracle = None
        self.initialization_seconds = 0.
        self.evaluated_sources = set()
        from rind.geometry import backend_build_id
        from rind import transcendental, expanded_geometry
        self.provenance = {
            'physics': POLICY, 'oracle_limits': {},
            'canonical_backend': backend_build_id(3),
            'function_source_sha256': {
                Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                for m in (transcendental, expanded_geometry)},
            'filter_policy': 'outward-interval-convex-union-v1'}
        self.policy_hash = hashlib.sha256(json.dumps(self.provenance, sort_keys=True).encode()).hexdigest()
        self.capability = {'supported': True, 'ready_capable': False,
                           'reason': 'Experimental polyhedral path; production acceptance gates unchanged'}

    @property
    def canonical(self):
        with self.lock:
            if self._oracle is None:
                t = time.perf_counter()
                self._oracle = Oracle(self.bundle)
                assert self._oracle.policy_hash == self.policy_hash
                self.initialization_seconds += time.perf_counter()-t
            return self._oracle


class GeometryGroup(LazyCanonical):
    def __init__(self, bundle):
        t = time.perf_counter()
        super().__init__(bundle)
        self.bodies, self.old = extract(bundle)
        # Exact boundedness proof; extract proves the strict interior witness
        # and exact nonsingular affine transform, without constructing a mesh.
        self.native = native.Geometry(strings(self.bodies))
        self.geometry_id = self.geometry.content_hash.hex()
        self.setup_seconds = time.perf_counter()-t

    def bind(self, source, *, solid=None, projected=False):
        return CPUOracle(self, source, solid=solid, projected=projected)


class CPUOracle:
    def __init__(self, shared, source, *, solid=None, projected=False):
        t = time.perf_counter()
        self.shared = shared
        self.base = shared
        self.source = np.array(source, dtype='<f8', copy=True).reshape(3)
        self.source.setflags(write=False)
        self.source_id = hashlib.sha256(bytes.fromhex(shared.geometry_id)+self.source.tobytes()).hexdigest()
        self.work = Counter()
        self.solid = solid
        self.projected = projected
        if solid is not None and (solid.geometry_id != shared.geometry_id or solid.policy_hash != shared.policy_hash):
            raise ValueError('Mask identity mismatch')
        self.planes = [shadow_planes(b, self.source) for b in shared.bodies] if projected else []
        self.native = native.Model(shared.native, self.source.tolist(), strings(self.planes))
        self.setup_seconds = time.perf_counter()-t

    def __getattr__(self, name):
        return getattr(self.shared, name)

    @property
    def metrics(self):
        return dict(self.work)

    def report(self):
        return {'version': 'cached-clipping-experiment-v1', 'production_ready': False,
                'geometry_id': self.shared.geometry_id, 'source_id': self.source_id,
                'geometry_setup_seconds': self.shared.setup_seconds,
                'source_setup_seconds': self.setup_seconds, 'projected': self.projected,
                'canonical_initialization_seconds': self.shared.initialization_seconds,
                'geometry_plane_counts': [len(b) for b in self.shared.bodies],
                'shadow_plane_counts': [len(b) for b in self.planes],
                'solid_mask_reused': self.solid is not None, 'work': dict(self.work)}

    def _same(self, source):
        return np.array_equal(np.asarray(source).reshape(-1), self.source)

    def validate_source(self, source, budget=None):
        if budget: budget.check()
        if not self._same(source):
            self.shared.canonical.validate_source(source, budget)
        # Native Model construction already used exact membership and bounds.

    def certificate(self, source, lo, hi, *, budget=None):
        if budget: budget.check()
        if not self._same(source):
            return self.shared.canonical.certificate(source, lo, hi, budget=budget)
        self.work['certificate_calls'] += 1
        return self.native.certificate(list(lo), list(hi))

    def _points(self, points, budget=None):
        p = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        if budget: budget.check(len(p))
        y, m, stats = self.native.points(p)
        self.work.update(stats)
        return y.astype(bool), m.astype(bool)

    def response(self, source, points, *, budget=None):
        return self._points(points, budget)[0] if self._same(source) else self.shared.canonical.response(source, points, budget=budget)

    def contains(self, points, *, budget=None):
        return self._points(points, budget)[1]

    def render(self, source, origin, size, n, *, budget=None, leaf_side=8, old_mode=False):
        if not self._same(source): raise ValueError('Source-dependent cache identity mismatch')
        if budget: budget.check()
        y, m, stats = self.native.raster(list(origin), size, n, leaf_side, self.projected, self.solid is not None)
        self.work.update(stats)
        if budget: budget.check(stats['point_rays']+stats['solid_points'])
        if self.solid is not None:
            if self.solid.shape != (n,)*3: raise ValueError('Mask resolution mismatch')
            m = self.solid.read_window(*origin, size)
            self.work['solid_bricks_reused'] += 1
        if np.any(y & m): raise AssertionError('Solid receiver lit')
        return y.astype(bool), m.astype(bool), stats['certificate_covered_voxels']


def optional_group(bundle):
    try:
        return GeometryGroup(bundle)
    except (Unsupported, ValueError):
        return Oracle(bundle)
