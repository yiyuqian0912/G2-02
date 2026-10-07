"""Independent boundary-cost adapter for C's scalar CandidateEvaluator contract."""
from dataclasses import dataclass
import numpy as np


def boundary(response):
    response = np.asarray(response,dtype=float)
    if response.ndim != 2 or not response.size or not np.isfinite(response).all():
        raise ValueError('finite nonempty 2D response required')
    mask = np.zeros(response.shape,dtype=bool)
    horizontal, vertical = response[:,1:] != response[:,:-1], response[1:,:] != response[:-1,:]
    mask[:,1:] |= horizontal
    mask[:,:-1] |= horizontal
    mask[1:,:] |= vertical
    mask[:-1,:] |= vertical
    return mask


def nearest_distance(points, edges):
    """Exact Euclidean nearest distances; bounded fallback when SciPy is absent."""
    points, edges = np.asarray(points),np.asarray(edges)
    if not len(edges): raise ValueError('nonempty edges required')
    try:
        from scipy.spatial import cKDTree
        return cKDTree(edges).query(points)[0]
    except ImportError:
        result = []
        for start in range(0,len(points),128):
            chunk = points[start:start+128]
            minimum = np.full(len(chunk),np.inf)
            for edge_start in range(0,len(edges),256):
                d = chunk[:,None,:]-edges[None,edge_start:edge_start+256,:]
                minimum = np.minimum(minimum,np.sum(d*d,axis=-1).min(axis=1))
            result.extend(np.sqrt(minimum))
        return np.asarray(result)


def boundary_weights(response, *, boundary_lambda=0., boundary_sigma=1.):
    if not np.isfinite(boundary_lambda) or boundary_lambda < 0 or not np.isfinite(boundary_sigma) or boundary_sigma <= 0:
        raise ValueError('lambda>=0 and sigma>0 required')
    edges = np.argwhere(boundary(response))
    shape = np.asarray(response).shape
    if not len(edges) or boundary_lambda == 0: return np.ones(shape)
    points = np.indices(shape).reshape(2,-1).T
    d = nearest_distance(points,edges).reshape(shape)
    return 1+boundary_lambda*np.exp(-d*d/(2*boundary_sigma**2))


def costs(observed, rendered, *, alpha=1., beta=0., boundary_lambda=0., boundary_sigma=1., weights=None):
    observed, rendered = np.asarray(observed,dtype=float),np.asarray(rendered,dtype=float)
    eo, er = boundary(observed),boundary(rendered)
    if observed.shape != rendered.shape: raise ValueError('response shapes differ')
    if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(beta) or beta < 0:
        raise ValueError('alpha>0 and beta>=0 required')
    if weights is None: weights = boundary_weights(observed,boundary_lambda=boundary_lambda,boundary_sigma=boundary_sigma)
    weights = np.asarray(weights)
    if weights.shape != observed.shape or not np.isfinite(weights).all() or (weights<=0).any():
        raise ValueError('positive finite pixel weights required')
    response_cost = float(np.sum(weights*np.abs(observed-rendered))/weights.sum())
    edge = None
    if beta > 0:
        a,b = np.argwhere(eo),np.argwhere(er)
        diagonal = float(np.hypot(max(observed.shape[0]-1,0),max(observed.shape[1]-1,0)))
        def directed(source,target):
            if not len(source) or diagonal == 0: return 0.
            if not len(target): return 1.
            return float(nearest_distance(source,target).mean()/diagonal)
        edge = directed(a,b)+directed(b,a)
    return {'L_resp':response_cost,'L_edge':edge,
            'physical_cost':alpha*response_cost+(beta*edge if edge is not None else 0.)}


@dataclass
class CandidateEvaluation:
    valid: bool
    physical_cost: float
    num_physics_evaluations: int = 0
    cache_hit: bool = False


class BoundaryEvaluator:
    """Use the original dataset renderer; hidden geometry stays evaluation-only."""
    def __init__(self,dataset,scene_id,view_id,*,alpha=1.,beta=0.,boundary_lambda=0.,boundary_sigma=1.,backend='auto'):
        self.dataset,self.scene_id = dataset,scene_id
        self.sample = dataset.get_observation(scene_id,view_id)
        self.scene = dataset.get_scene(scene_id)
        self.settings = dict(alpha=alpha,beta=beta,boundary_lambda=boundary_lambda,boundary_sigma=boundary_sigma)
        self.weights = boundary_weights(self.sample['response'],boundary_lambda=boundary_lambda,boundary_sigma=boundary_sigma)
        costs(self.sample['response'],self.sample['response'],weights=self.weights,**self.settings)
        self.backend,self.cache = backend,{}

    def evaluate(self,xy):
        from rind_dataset.geometry import is_driver_valid_mixed
        xy = np.asarray(xy,dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all(): raise ValueError('finite xy required')
        key = tuple(xy)
        if key in self.cache:
            value = self.cache[key]
            return CandidateEvaluation(value.valid,value.physical_cost,cache_hit=True)
        valid = bool(is_driver_valid_mixed(xy,self.scene['obstacle_types'],self.scene['obstacle_params'],
                     len(self.scene['obstacle_types']),domain_size=self.dataset.manifest['global_size']))
        if not valid:
            result = CandidateEvaluation(False,np.inf)
        else:
            rendered = self.dataset.rerender(self.scene_id,self.sample['window'],xy,backend=self.backend)
            value = costs(self.sample['response'],rendered,weights=self.weights,**self.settings)
            result = CandidateEvaluation(True,value['physical_cost'],num_physics_evaluations=1)
        self.cache[key] = result
        return result
