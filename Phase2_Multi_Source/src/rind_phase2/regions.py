"""Response-region graph with continuous-geometry-safe four-neighbor edges."""
from dataclasses import dataclass
import numpy as np
import rind_dataset.geometry as geo
from .region_geometry import segment_intersects_obstacle as intersects
try:
    from numba import njit
except ImportError:
    def njit(**kwargs):
        return lambda function: function

@njit(cache=True)
def find(par,a):
    while par[a]!=a:
        par[a]=par[par[a]];a=par[a]
    return a

@njit(cache=True)
def graph(free,levels,x0,y0,types,params,aabbs):
    h,w=free.shape;n=h*w;par=np.arange(n);u=np.empty(2*n,np.int64);v=np.empty(2*n,np.int64)
    count=0;blocked=0
    for y in range(h):
        for x in range(w):
            if not free[y,x]:continue
            for axis in range(2):
                xx=x+int(axis==0);yy=y+int(axis==1)
                if xx>=w or yy>=h or not free[yy,xx]:continue
                px=x0+x+.5;py=y0+y+.5;qx=x0+xx+.5;qy=y0+yy+.5;hit=False
                for j in range(len(types)):
                    if qx<aabbs[j,0] or px>aabbs[j,2] or qy<aabbs[j,1] or py>aabbs[j,3]:continue
                    if intersects(px,py,qx,qy,int(types[j]),params[j]):hit=True;break
                if hit:blocked+=1;continue
                a=y*w+x;b=yy*w+xx;u[count]=a;v[count]=b;count+=1
                if levels[y,x]==levels[yy,xx]:
                    ra=find(par,a);rb=find(par,b)
                    if ra!=rb:par[rb]=ra
    for p in range(n):par[p]=find(par,p)
    return par,u[:count],v[:count],blocked

def response_bins(y,tol):
    vals,inv=np.unique(y,return_inverse=True);labels=np.zeros(len(vals),np.int64)
    start=vals[0];label=0
    for i,a in enumerate(vals):
        if a-start>tol:label+=1;start=a
        labels[i]=label
    return labels[inv].reshape(y.shape)

@dataclass
class ResponseRegions:
    labels: np.ndarray
    values: np.ndarray
    pixels: list
    neighbors: list
    excluded_edges: int


def response_regions(response, obstacle, origin, types, params, *, tolerance=1e-12):
    """Equal-valued disconnected areas remain distinct region IDs."""
    response = np.asarray(response, dtype=np.float64)
    obstacle = np.asarray(obstacle, dtype=bool)
    if response.ndim != 2 or response.shape != obstacle.shape or not np.isfinite(response).all():
        raise ValueError("Response and obstacle must be aligned finite 2D arrays")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Region tolerance must be finite and nonnegative")
    boxes = np.array([geo.compute_obstacle_aabb(int(t), p) for t, p in zip(types, params)]).reshape(-1, 4)
    par, u, v, blocked = graph(~obstacle, response_bins(response, tolerance),
                              int(origin[0]), int(origin[1]), types, params, boxes)
    free = np.flatnonzero(~obstacle.ravel())
    roots, inverse = np.unique(par[free], return_inverse=True)
    labels = np.full(response.size, -1, dtype=np.int64)
    labels[free] = inverse
    order = np.argsort(inverse, kind="stable")
    sizes = np.bincount(inverse, minlength=len(roots))
    pixels = np.split(free[order], np.cumsum(sizes)[:-1]) if len(roots) else []
    if pixels:
        order = np.argsort([indices[0] for indices in pixels])
        remap = np.empty(len(order), dtype=np.int64)
        remap[order] = np.arange(len(order))
        labels[free] = remap[labels[free]]
        pixels = [pixels[i] for i in order]
    values = np.array([response.ravel()[indices].min() for indices in pixels])
    neighbors = [set() for _ in roots]
    if len(u):
        pairs = np.unique(np.sort(np.column_stack((labels[u], labels[v])), axis=1), axis=0)
        for a, b in pairs:
            if a != b:
                neighbors[a].add(int(b)); neighbors[b].add(int(a))
    return ResponseRegions(labels.reshape(response.shape), values, pixels,
                           [sorted(n) for n in neighbors], int(blocked))


def check_geometry_backend():
    """Compile and verify the teacher adjacency path before starting workers."""
    from rind_dataset.constants import PARAM_WIDTH
    from .region_geometry import segment_intersects_obstacle
    types = np.arange(4, dtype=np.uint8)
    params = np.zeros((4, PARAM_WIDTH), dtype=np.float64)
    params[0, :5] = [8.0, 8.0, 2.0, 4.0, 0.3]
    params[1, :5] = [20.0, 8.0, 3.0, 2.0, -0.2]
    params[2, :6] = [5.0, 20.0, 11.0, 19.0, 8.0, 25.0]
    vertices = [(18.0, 18.0), (24.0, 18.0), (24.0, 24.0),
                (21.0, 21.0), (18.0, 24.0)]
    params[3, 0] = len(vertices)
    params[3, 1:1 + 2 * len(vertices)] = np.asarray(vertices).ravel()
    rng = np.random.default_rng(20261028)
    segments = rng.uniform(0.0, 32.0, size=(32, 4))
    checks = 0
    for kind, row in zip(types, params):
        for px, py, qx, qy in segments:
            actual = segment_intersects_obstacle(px, py, qx, qy, int(kind), row)
            expected = geo.segment_intersects_obstacle(px, py, qx, qy, int(kind), row)
            if actual != expected:
                raise RuntimeError(f"region geometry differs from canonical renderer for type {kind}")
            checks += 1
    thin = np.zeros((1, PARAM_WIDTH), dtype=np.float64)
    thin[0, :5] = [1.0, 0.5, 0.1, 0.4, 0.0]
    result = response_regions(np.ones((1, 2)), np.zeros((1, 2), dtype=bool),
                              (0, 0), types[:1], thin)
    if len(result.values) != 2 or result.excluded_edges != 1:
        raise RuntimeError("thin obstacle was not excluded from free-space adjacency")
    compiled = bool(getattr(graph, "nopython_signatures", ()))
    try:
        import numba
        numba_version = numba.__version__
    except ImportError:
        numba_version = None
    return {"geometry_checks": checks, "nopython_compiled": compiled,
            "thin_obstacle_adjacency_checked": True,
            "numpy_version": np.__version__, "numba_version": numba_version}
