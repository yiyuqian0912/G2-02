# Legacy four-side diagnostic/report reproduction only; not the current dataset contract.
"""Exact centre-set distances, plus clearly labelled constructive nested chains."""
import numpy as np
from experimental_boundary_guided import pair_region
from analyze_boundary_ab import common_center_regions


def minimum_box_distance(point,boxes):
    if not boxes:return None,None
    candidates=[];p=np.asarray(point,dtype=int)
    for box in boxes:
        q=np.clip(p,box.lower,box.upper).astype(int)
        candidates.append((int(((q-p)**2).sum()),tuple(map(int,q))))
    squared,q=min(candidates)
    return float(np.sqrt(squared)),q


def actual_anchor_metrics(origin32,pair,source,resolution=1024):
    origin=np.asarray(origin32,dtype=int);center=origin+16
    common=common_center_regions(pair,source,resolution)
    shift,closest=minimum_box_distance(center,common)
    independent_scale_minima={};chain=[];previous=origin.copy();previous_side=32
    for side in (32,64,128,256):
        # These minima contain the actual side-32 window, but are independent
        # across larger scales; not falsely called a joint global optimum.
        options=[]
        for box in pair_region(pair,source,side,resolution):
            lo=np.maximum(box.lower,origin+32-side);hi=np.minimum(box.upper,origin)
            if np.any(lo>hi):continue
            q=np.clip(center-side//2,lo,hi).astype(int)
            options.append((int(((q+side//2-center)**2).sum()),tuple(map(int,q))))
        independent_scale_minima[str(side)]=float(np.sqrt(min(options)[0])) if options else None
        if side==32:chain=[{'side':32,'origin':origin.tolist(),'center_displacement':0.}];continue
        nested=[]
        for box in pair_region(pair,source,side,resolution):
            lo=np.maximum(box.lower,previous+previous_side-side);hi=np.minimum(box.upper,previous)
            if np.any(lo>hi):continue
            q=np.clip(center-side//2,lo,hi).astype(int)
            nested.append((int(((q+side//2-center)**2).sum()),tuple(map(int,q))))
        if not nested:break
        distance,q=min(nested);chain.append({'side':side,'origin':list(q),'center_displacement':float(np.sqrt(distance))})
        previous=np.asarray(q);previous_side=side
    return {'any_same_center_chain_exists':bool(common),'actual_anchor_exactly_aligned':shift==0 if shift is not None else False,
            'minimum_shift_to_an_all_scale_common_center_voxels':shift,'closest_common_center':list(closest) if closest else None,
            'independent_scale_minimum_center_shifts':independent_scale_minima,
            'anchored_nested_chain_constructed':len(chain)==4,'anchored_greedy_chain':chain,
            'note':'common-centre shift is exact; per-scale minima are independent; greedy chain is feasible, not claimed globally minimal'}
