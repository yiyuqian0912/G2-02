"""Geometry-blind association and evidence fusion across local views.

Each input layer remains a Phase-I-style single-contribution posterior.  This
module never reads obstacle geometry or true source identity.  It links layers
from different views conservatively and fuses only layers assigned to the same
track.  Ambiguous layers remain separate instead of forcing a false match.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class LayerField:
    view_id: int
    layer_id: int
    intensity: float
    probability: np.ndarray

    def __post_init__(self):
        probability=np.asarray(self.probability,dtype=np.float64)
        total=float(probability.sum())
        if (probability.ndim!=1 or not len(probability) or
                not np.isfinite(probability).all() or np.any(probability<0) or
                not math.isfinite(total) or total<=0 or
                not np.isclose(total,1.0,rtol=0,atol=1e-5)):
            raise ValueError('probability must be one finite normalized vector')
        if not math.isfinite(self.intensity) or self.intensity<=0:
            raise ValueError('intensity must be positive and finite')
        probability=(probability/total).copy();probability.setflags(write=False)
        object.__setattr__(self,'probability',probability)


def bhattacharyya_overlap(first,second):
    first=np.asarray(first,dtype=np.float64);second=np.asarray(second,dtype=np.float64)
    first_total=float(first.sum());second_total=float(second.sum())
    if (first.shape!=second.shape or first.ndim!=1 or not len(first) or
            not np.isfinite(first).all() or not np.isfinite(second).all() or
            np.any(first<0) or np.any(second<0) or
            not np.isclose(first_total,1.0,rtol=0,atol=1e-5) or
            not np.isclose(second_total,1.0,rtol=0,atol=1e-5)):
        raise ValueError('probability fields must be finite normalized matching vectors')
    return float(np.sqrt((first/first_total)*(second/second_total)).sum())


def centered_shape_similarity(first,second):
    """Cosine similarity after removing the uninformative uniform field."""
    first=np.asarray(first,dtype=np.float64);second=np.asarray(second,dtype=np.float64)
    bhattacharyya_overlap(first,second)  # shared validation
    first=first/first.sum();second=second/second.sum()
    uniform=1.0/len(first)
    left=first-uniform;right=second-uniform
    denominator=float(np.linalg.norm(left)*np.linalg.norm(right))
    if denominator<=np.finfo(np.float64).eps:
        return 0.0
    return float(np.clip(np.dot(left,right)/denominator,-1.0,1.0))


def _uniform_informativeness(probability):
    probability=np.asarray(probability,dtype=np.float64)
    probability=probability/probability.sum()
    return .5*float(np.abs(probability-1.0/len(probability)).sum())


def layer_association_score(first,second,intensity_log_scale=0.25,
                            informativeness_scale=0.05):
    """Score two layers without geometry or source labels.

    Probability overlap is the main evidence.  Intensity agreement is a soft
    factor because residual layers are effective contributions and therefore
    need not equal one generating source strength exactly.
    """
    if first.view_id==second.view_id:
        return 0.0
    if (not math.isfinite(informativeness_scale) or
            informativeness_scale<=0 or informativeness_scale>1):
        raise ValueError('informativeness_scale must be in (0,1]')
    overlap=bhattacharyya_overlap(first.probability,second.probability)
    shape=max(0.0,centered_shape_similarity(first.probability,second.probability))
    # Centered cosine measures direction but not magnitude: two almost-uniform
    # fields with the same numerical ripple can otherwise score nearly one.
    information=min(_uniform_informativeness(first.probability),
                    _uniform_informativeness(second.probability))
    reliability=min(1.0,information/informativeness_scale)
    if intensity_log_scale is None:
        return overlap*shape*reliability
    if not math.isfinite(intensity_log_scale) or intensity_log_scale<=0:
        raise ValueError('intensity_log_scale must be positive or None')
    distance=abs(math.log(first.intensity/second.intensity))
    return overlap*shape*reliability*math.exp(-distance/intensity_log_scale)


def associate_layers(fields,minimum_score=0.55,intensity_log_scale=0.25,
                     exact_intensity_minimum_score=0.15,
                     intensity_relative_tolerance=1e-6,
                     informativeness_scale=0.05):
    """Return conservative cross-view tracks using constrained agglomeration.

    A track may contain at most one layer from each view.  Edges are processed
    from strongest to weakest.  This is a deterministic first-stage proposal,
    not a proof that the linked layers have the same physical source.
    """
    fields=list(fields)
    if not fields:
        return []
    if not math.isfinite(minimum_score) or minimum_score<0 or minimum_score>1:
        raise ValueError('minimum_score must be in [0,1]')
    if (not math.isfinite(exact_intensity_minimum_score) or
            exact_intensity_minimum_score<0 or
            exact_intensity_minimum_score>1):
        raise ValueError('exact_intensity_minimum_score must be in [0,1]')
    exact_intensity_minimum_score=min(exact_intensity_minimum_score,minimum_score)
    if (not math.isfinite(intensity_relative_tolerance) or
            intensity_relative_tolerance<0):
        raise ValueError('intensity_relative_tolerance must be nonnegative')
    width=len(fields[0].probability)
    if any(len(field.probability)!=width for field in fields):
        raise ValueError('all fields must use the same candidate grid')
    parent=list(range(len(fields)))
    members={index:{index} for index in range(len(fields))}
    views={index:{fields[index].view_id} for index in range(len(fields))}

    def root(index):
        while parent[index]!=index:
            parent[index]=parent[parent[index]]
            index=parent[index]
        return index

    def qualifies(first,second,score):
        scale=max(fields[first].intensity,fields[second].intensity)
        exact=abs(fields[first].intensity-fields[second].intensity)<=intensity_relative_tolerance*scale
        return score>=(exact_intensity_minimum_score if exact else minimum_score)

    edges=[]
    for first in range(len(fields)):
        for second in range(first+1,len(fields)):
            score=layer_association_score(
                fields[first],fields[second],intensity_log_scale,informativeness_scale)
            if qualifies(first,second,score):
                edges.append((-score,first,second))
    for negative_score,first,second in sorted(edges):
        left,right=root(first),root(second)
        if left==right or views[left]&views[right]:
            continue
        # Strong pairwise evidence must connect the two components directly;
        # transitive chains alone do not justify merging unrelated endpoints.
        cross=[(a,b,layer_association_score(
                    fields[a],fields[b],intensity_log_scale,informativeness_scale))
               for a in members[left] for b in members[right]]
        if not all(qualifies(a,b,score) for a,b,score in cross):
            continue
        if len(members[left])<len(members[right]):
            left,right=right,left
        parent[right]=left
        members[left]|=members.pop(right)
        views[left]|=views.pop(right)
    tracks=[]
    for indices in members.values():
        tracks.append(sorted(indices,key=lambda index:(fields[index].view_id,
                                                        fields[index].layer_id)))
    return sorted(tracks,key=lambda track:(fields[track[0]].view_id,
                                           fields[track[0]].layer_id))


def fuse_track(fields,track,weights=None,prior=None,epsilon=1e-12,
               redundancy_discount=0.5):
    """Fuse one associated contribution track by a calibrated product rule."""
    fields=list(fields);track=list(track)
    if not track:
        raise ValueError('track must contain at least one layer')
    if len({fields[index].view_id for index in track})!=len(track):
        raise ValueError('a fused track may contain at most one layer per view')
    probabilities=np.stack([fields[index].probability for index in track])
    if weights is None:
        if (not math.isfinite(redundancy_discount) or redundancy_discount<0 or
                redundancy_discount>1):
            raise ValueError('redundancy_discount must be in [0,1]')
        # Effective evidence grows from one view toward the full independent
        # product.  A discount below one protects against overlapping windows
        # and the shared unknown geometry being counted repeatedly.
        total=1.0+redundancy_discount*(len(track)-1)
        weights=np.full(len(track),total/len(track),dtype=np.float64)
    weights=np.asarray(weights,dtype=np.float64)
    if (weights.shape!=(len(track),) or not np.isfinite(weights).all() or
            np.any(weights<0) or weights.sum()<=0):
        raise ValueError('weights must be finite nonnegative values with positive sum')
    log_evidence=(weights[:,None]*np.log(np.maximum(probabilities,epsilon))).sum(0)
    if prior is not None:
        prior=np.asarray(prior,dtype=np.float64)
        prior_total=float(prior.sum())
        if (prior.shape!=log_evidence.shape or not np.isfinite(prior).all() or
                np.any(prior<=0) or
                not np.isclose(prior_total,1.0,rtol=0,atol=1e-5)):
            raise ValueError('prior must be one strictly positive normalized vector')
        prior=prior/prior_total
        log_evidence-=(weights.sum()-1)*np.log(prior)
    log_evidence-=log_evidence.max()
    probability=np.exp(log_evidence)
    return probability/probability.sum()


def fuse_associated_layers(fields,minimum_score=0.55,intensity_log_scale=0.25,
                           prior=None,redundancy_discount=0.5,
                           exact_intensity_minimum_score=0.15,
                           intensity_relative_tolerance=1e-6,
                           informativeness_scale=0.05):
    tracks=associate_layers(
        fields,minimum_score,intensity_log_scale,exact_intensity_minimum_score,
        intensity_relative_tolerance,informativeness_scale)
    return [{'indices':track,'probability':fuse_track(
                fields,track,prior=prior,redundancy_discount=redundancy_discount)}
            for track in tracks]
