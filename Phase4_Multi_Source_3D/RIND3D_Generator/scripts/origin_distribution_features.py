# Legacy four-side diagnostic; archived results/source snapshots are unchanged.
"""Recorded-window diagnostics, never used to accept/filter a window.

Composition is explicitly a held-out stratified estimator unless matching stored
truth exists. Metadata AABBs are descriptors, not approximated solid geometry.
"""
from fractions import Fraction
import hashlib,json,time
from pathlib import Path
import numpy as np
from rind.contracts import Leaf,CodecID
from rind.geometry import Clearance
from rind3d.geometry import Budget,Aborted,NumericalUnresolved
from rind3d.legacy.windows import Window
from rind3d.pack import RINDPack
from experimental_boundary_guided import pair_id
from boundary_experiment_support import plain


def stable_rng(values):
    token=json.dumps(plain(values),sort_keys=True).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:16],'little'))


def primitive_descriptors(bundle):
    result=[];counts={}
    for leaf in bundle.references:
        if not isinstance(leaf,Leaf):continue
        name=leaf.codec.name;counts[name]=counts.get(name,0)+1
        if leaf.codec not in (CodecID.AFFINE_BOX,CodecID.AFFINE_BALL):continue
        matrix=leaf.field('matrix').as_array();center=leaf.field('center').as_array()
        if not np.array_equal(matrix,np.diag(np.diag(matrix))):continue
        c=[Fraction(float(x)) for x in center];r=[abs(Fraction(float(x))) for x in np.diag(matrix)]
        result.append({'codec':name,'lower':tuple(a-b for a,b in zip(c,r)),
                       'upper':tuple(a+b for a,b in zip(c,r))})
    return result,counts


def spatial_features(window,source,resolution):
    origin=np.asarray(window.origin,dtype=float);side=window.size
    lo=origin/resolution;hi=(origin+side)/resolution;center=(lo+hi)/2;s=np.asarray(source)
    source_box=float(np.linalg.norm(np.maximum(np.maximum(lo-s,s-hi),0)))
    return {'origin':list(window.origin),'side':side,'center_world':center.tolist(),
            'center_world_boundary_distance':float(np.minimum(center,1-center).min()),
            'window_world_boundary_clearance':float(np.minimum(lo,1-hi).min()),
            'source_center_distance':float(np.linalg.norm(center-s)),'source_window_distance':source_box,
            'center_bucket128':np.minimum(7,((origin+side/2)//128).astype(int)).tolist()}


def witness_features(window,resolution):
    coords=np.asarray(window.witnesses,dtype=float);delta=coords[0]-coords[1]
    points=coords+.5;midpoint=points.mean(axis=0)
    rel=(points-window.origin)/window.size;mid_rel=rel.mean(axis=0)
    if not np.all((rel>=0)&(rel<1)):raise AssertionError('Witness outside accepted window')
    return {'pair_id':pair_id(*window.witnesses),'witnesses':plain(window.witnesses),
            'pair_distance_voxels':float(np.linalg.norm(delta)),'pair_max_axis_voxels':float(np.abs(delta).max()),
            'pair_distance_world':float(np.linalg.norm(delta))/resolution,
            'pair_distance_relative':float(np.linalg.norm(delta))/window.size,
            'pair_max_axis_relative':float(np.abs(delta).max())/window.size,
            'witness_midpoint_world':(midpoint/resolution).tolist(),
            'witness_bucket128':np.minimum(7,(midpoint//128).astype(int)).tolist(),
            'positive_relative':rel[0].tolist(),'negative_relative':rel[1].tolist(),
            'midpoint_relative':mid_rel.tolist(),
            'midpoint_face_min_relative':float(np.minimum(mid_rel,1-mid_rel).min()),
            'witness_face_min_relative':float(np.minimum(rel,1-rel).min())}


def metadata_geometry_features(window,resolution,descriptors,counts):
    lo=[Fraction(int(x),resolution) for x in window.origin]
    hi=[Fraction(int(x+window.size),resolution) for x in window.origin]
    touched=[d for d in descriptors if all(max(a,b)<=min(c,e) for a,b,c,e in zip(lo,d['lower'],hi,d['upper']))]
    return {'scene_leaf_counts':counts,'finite_leaf_aabb_touch_count':len(touched),
            'finite_leaf_aabb_touch_density':len(touched)/(window.size/resolution)**3,
            'structure_descriptor_semantics':'exact canonical diagonal-leaf AABB intersects window closure; not solid-intersection count',
            'nearest_geometry_exact_distance':None,
            'nearest_geometry_distance_note':'No distance approximation; certified fixed-radius clearance recorded separately'}


def diagnostic_coordinates(window,strata,seed_key):
    rng=stable_rng(['held-out-stratified-composition-v1',seed_key])
    shape=np.asarray(strata,dtype=int)
    if np.any(shape<=0) or np.any(window.size%shape):raise ValueError('Equal-volume integer strata required')
    width=window.size//shape;coordinates=[]
    for index in np.ndindex(*shape):
        offset=np.asarray(index)*width+rng.integers(0,width,size=3)
        coordinates.append((np.asarray(window.origin)+offset).tolist())
    return np.asarray(coordinates,dtype=int)


def measure_composition(window,source,oracle,settings,seed_key,stored=None):
    if stored:
        with RINDPack(stored['response'],verify_file=True) as y,RINDPack(stored['solid'],verify_file=True) as m:
            response=y.read_window(*window.origin,window.size);solid=m.read_window(*window.origin,window.size)
        free=~solid;n=int(free.sum());lit=int((response&free).sum())
        return {'composition_kind':'exact_stored','composition_status':'COMPLETE','solid_fraction':float(solid.mean()),
                'free_fraction':float(free.mean()),'lit_fraction':lit/n if n else None,'shadow_fraction':1-lit/n if n else None,
                'composition_n':window.size**3,'free_n':n,'lit_free_n':lit},None
    coordinates=diagnostic_coordinates(window,settings['composition_strata'],seed_key)
    budget=Budget(settings['composition_seconds_per_window'],50_000_000);solid=[];response=[];error=None
    try:
        for point in (coordinates+.5)/1024:
            solid.append(bool(oracle.contains(point[None],budget=budget)[0]))
            response.append(bool(oracle.response(source,point[None],budget=budget)[0]))
    except (Aborted,NumericalUnresolved) as exc:error=type(exc).__name__+':'+str(exc)
    probes={'coordinates':coordinates.tolist(),'solid':solid,'response':response,'error':error}
    if error is not None:
        return {'composition_kind':'diagnostic_estimator','composition_status':'UNAVAILABLE','composition_error':error,
                'solid_fraction':None,'free_fraction':None,'lit_fraction':None,'shadow_fraction':None,
                'composition_n':len(response),'free_n':None,'lit_free_n':None},probes
    solid=np.asarray(solid,bool);response=np.asarray(response,bool)
    if np.any(solid&response):raise AssertionError('Solid receiver lit in diagnostic queries')
    free=~solid;n=int(free.sum());lit=int((response&free).sum())
    return {'composition_kind':'diagnostic_estimator','composition_status':'COMPLETE','solid_fraction':float(solid.mean()),
            'free_fraction':float(free.mean()),'lit_fraction':lit/n if n else None,'shadow_fraction':1-lit/n if n else None,
            'composition_n':len(response),'free_n':n,'lit_free_n':lit,
            'composition_note':'32 equal-volume strata, one exact random voxel centre each; not exact full-window fractions; no witness forced into estimate'},probes


def measure_center_clearance(window,oracle,radius):
    center=(np.asarray(window.origin)+window.size/2)/1024
    result={'clearance_radius_world':radius,'center_solid':None,'center_near_solid_r32':None,'clearance_reason':None}
    try:
        result['center_solid']=bool(oracle.contains(center[None],budget=Budget(120))[0])
        # Read-only original canonical predicate. It can legitimately be unproved.
        query=oracle.reference.clearance_decide(center[None],radius)
        valid=bool(query.valid[0]);value=int(query.value[0])
        result.update(clearance_query_status=int(query.status[0]),clearance_query_value=value,
                      clearance_query_valid=valid,clearance_reason=str(query.reason_code[0]))
        if valid and value in (int(Clearance.LE),int(Clearance.GT)):
            result['center_near_solid_r32']=value==int(Clearance.LE)
    except (RuntimeError,AttributeError) as exc:result['clearance_reason']=type(exc).__name__+':'+str(exc)
    return result


def audit_windows(windows,source,cfg,scene_meta,source_rank,method,oracle,settings,output,stored=None):
    output=Path(output);descriptors,counts=primitive_descriptors(oracle.bundle);rows=[]
    with (output/'audit-probes.jsonl').open('w') as probes:
        for i,w in enumerate(windows):
            row={'method':method,'seed':scene_meta['seed'],'scene':scene_meta['scene'],'families':scene_meta['families'],
                 'source_rank':source_rank,'source':plain(source),'window_id':i,
                 **spatial_features(w,source,1024),**witness_features(w,1024),
                 **metadata_geometry_features(w,1024,descriptors,counts)}
            key=[scene_meta['seed'],scene_meta['scene'],source_rank,w.size,row['pair_id']]
            with oracle.timed('diagnostic_composition'):
                composition,raw=measure_composition(w,source,oracle,settings,key,stored)
            row.update(composition)
            with oracle.timed('diagnostic_clearance'):
                row.update(measure_center_clearance(w,oracle,settings['clearance_radius_voxels']/1024))
            if raw is not None:probes.write(json.dumps(plain({'window_id':i,**raw}))+'\n');probes.flush()
            rows.append(row)
            (output/'features.jsonl').write_text(''.join(json.dumps(plain(r))+'\n' for r in rows))
    return rows
