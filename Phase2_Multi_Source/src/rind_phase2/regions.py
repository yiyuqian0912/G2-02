"""Response-region graph with continuous-geometry-safe four-neighbor edges."""
from dataclasses import dataclass
import inspect
import numpy as np
import rind_dataset.geometry as geo
try:
    from numba import njit
    from numba.extending import register_jitable
    for name, fn in list(vars(geo).items()):
        if inspect.isfunction(fn) and fn.__module__ == geo.__name__:
            register_jitable(fn)
except ImportError:
    def njit(**kwargs):
        return lambda function: function
intersects = geo.segment_intersects_obstacle

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
