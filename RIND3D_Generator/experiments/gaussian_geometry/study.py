"""Small resumable geometry/distribution study and matched query benchmarks."""
from collections import Counter
from fractions import Fraction as Q
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.stats import qmc

from .generators import DEFAULT,generate,grid_points
from .models import KernelScene,SpectralScene,ConvexScene,record,save
from .queries import FieldOracle,ConvexOracle,Limits,make_oracle,Decision,rational,point
from .reference import Reference
from .render import small_field


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def quality(scene,n):
    began=time.perf_counter();points=grid_points(n);values=scene.approx_values(points)
    mask=(values>=0).reshape((n,)*3)
    labels,count=ndimage.label(mask,structure=ndimage.generate_binary_structure(3,1))
    sizes=np.bincount(labels.ravel())[1:];positions=points[mask.ravel()]
    boundary=mask & ~ndimage.binary_erosion(mask)
    if len(positions):
        extent=(positions.max(0)-positions.min(0)+1/n).tolist()
        octants=len(np.unique((positions>=.5).astype(int) @ [1,2,4]))
        spatial=len(np.unique(np.minimum(3,(positions*4).astype(int)) @ [1,4,16]))/64
        distance=float(np.min(np.minimum(positions,1-positions))-.5/n)
        if len(positions)>3:
            eig,vec=np.linalg.eigh(np.cov(positions.T));aspect=float(np.sqrt(eig[-1]/eig[0])) if eig[0]>0 else None
            direction=np.abs(vec[:,-1]).tolist()
        else:aspect=None;direction=None
        thickness=2*ndimage.distance_transform_edt(mask)[mask]/n
    else:extent=None;octants=0;spatial=0;distance=None;aspect=None;direction=None;thickness=np.array([])
    surface=sum(np.count_nonzero(np.diff(mask.astype(np.int8),axis=a)) for a in range(3))/n**2
    components=[]
    if isinstance(scene,KernelScene):components=[k.matrix for k in scene.kernels]
    elif isinstance(scene,ConvexScene):components=[o['matrix'] for o in scene.objects]
    singular=[np.linalg.svd(m,compute_uv=False) for m in components]
    axes=[np.linalg.svd(m)[0][:,0] for m in components]
    if axes:
        moment=sum(np.outer(a,a) for a in axes)/len(axes);ev=np.maximum(0,np.linalg.eigvalsh(moment))
        orientation_entropy=float(-sum(e*np.log(e) for e in ev if e>0)/np.log(3))
    else:orientation_entropy=None
    result={'scope':'ordinary-float sampled geometry estimates; NOT certified topology',
            'resolution':n,'occupied_ratio_estimate':float(mask.mean()),'components_6_connected_estimate':int(count),
            'component_voxels_estimate':sorted(map(int,sizes),reverse=True),
            'largest_component_fraction_estimate':float(sizes.max()/sizes.sum()) if len(sizes) else None,
            'tiny_components_at_most_8_voxels':int(np.sum(sizes<=8)),
            'spatial_octants_estimate':octants,'spatial_4x4x4_coverage_estimate':spatial,
            'occupied_extent_xyz_estimate':extent,'PCA_aspect_estimate':aspect,
            'PCA_major_axis_abs_xyz_estimate':direction,'kernel_orientation_entropy_estimate':orientation_entropy,
            'surface_cubical_area_estimate':float(surface),'boundary_distance_estimate':distance,
            'voxel_thickness_below_0_04_fraction_estimate':float(np.mean(thickness<.04)) if len(thickness) else None,
            'minimum_kernel_axis':float(min(v[-1] for v in singular)) if singular else None,
            'parameter_anisotropy_above_20_count':sum(bool(v[0]/v[-1]>20) for v in singular),
            'approx_peak_xyz':points[int(np.argmax(values))].tolist(),'seconds':time.perf_counter()-began}
    return result


def sample_source(oracle,seed):
    rng=np.random.default_rng(seed);reject=Counter()
    for attempt in range(128):
        p=rng.random(3);m=oracle.membership(p)
        if m.status!='FREE':reject['membership_'+m.status]+=1;continue
        clearance=oracle.source_clearance(p)
        if clearance.status!='FREE':reject['clearance_'+clearance.status]+=1;continue
        return p,{'proposals':attempt+1,'rejections':dict(reject),'clearance_radius':2**-26,
                  'policy':'independent uniform open-world proposals with strict FREE clearance proof'}
    return None,{'proposals':128,'rejections':dict(reject),'status':'PROPOSAL_BUDGET_EXHAUSTED'}


def query_benchmark(scene,oracle,source,ordinal,cfg):
    rng=np.random.default_rng(np.random.SeedSequence([cfg['master_seed'],ordinal,3209]))
    points=qmc.Sobol(3,scramble=True,seed=int(rng.integers(2**32))).random_base2(7)
    begun=time.perf_counter();cpu=time.process_time();membership=[]
    for p in points[:cfg['point_queries_per_scene']]:membership.append(oracle.membership(p).status)
    point_time=time.perf_counter()-begun;free=[]
    for p in points:
        if oracle.membership(p).status=='FREE':free.append(p)
        if len(free)>=cfg['rays_per_scene']:break
    began=time.perf_counter();rays=[oracle.segment(source,p) for p in free]
    ray_time=time.perf_counter()-began
    result={'membership_counts':dict(Counter(membership)),'point_queries':len(membership),'point_seconds':point_time,
            'free_receiver_rays':len(rays),'ray_counts':dict(Counter(d.status for d in rays)),
            'ray_seconds':ray_time,'query_cpu_seconds':time.process_time()-cpu,
            'max_subdivision_depth':max([d.depth for d in rays],default=0),'max_ray_nodes':max([d.nodes for d in rays],default=0),
            'shadow_fraction_resolved_free_receivers':sum(d.status=='HIT' for d in rays)/max(1,sum(d.certified for d in rays)),
            'ray_work':dict(oracle.work)}
    began=time.perf_counter();agreed=disagreed=unresolved=0;reference_counts=Counter()
    if isinstance(oracle,ConvexOracle):
        a=oracle.context.bind(source);b=oracle.context.bind(source,independent=True)
        targets=np.array(free[:cfg['independent_queries_per_scene']])
        aa=a.response(source,targets);bb=b.response(source,targets)
        agreed=int(np.count_nonzero(aa==bb));disagreed=int(np.count_nonzero(aa!=bb))
        result['independent_path']='unchanged native A clipping and native B rational shadow projection'
    else:
        reference=Reference(scene,seconds=cfg['query_seconds']*4)
        for p,d in zip(free[:cfg['independent_queries_per_scene']],rays):
            rr=reference.segment(source,p);reference_counts[rr.status]+=1
            mm=reference.membership(p);primary=oracle.membership(p)
            if mm.certified and primary.certified:
                if mm.status!=primary.status:disagreed+=1
                else:agreed+=1
            else:unresolved+=1
            if rr.certified and d.certified:
                if rr.status!=d.status:disagreed+=1
                else:agreed+=1
            else:unresolved+=1
        result['independent_path']='separate Fraction cofactor / natural hull + directed MPFR'
    result.update(independent_seconds=time.perf_counter()-began,independent_agreed=agreed,
                  independent_disagreed=disagreed,independent_unresolved=unresolved,independent_counts=dict(reference_counts))
    if disagreed:raise AssertionError('Physical independent disagreement')
    return result


def convex_fraction_segment(context,start,stop):
    """Attribution at rational window-entry endpoints (no float64 rounding)."""
    for body in context.bodies:
        lo,hi=Q(0),Q(1)
        for row in body:
            base=sum((a*x for a,x in zip(row[:3],start)),Q(row[3]))
            slope=sum((a*(y-x) for a,x,y in zip(row[:3],start,stop)),Q(0))
            if not slope:
                if base>0:break
            elif slope>0:hi=min(hi,-base/slope)
            else:lo=max(lo,-base/slope)
            if lo>hi:break
        else:
            if lo<=hi:return Decision('HIT','exact_rational_convex_clip_for_attribution')
    return Decision('MISS','exact_rational_convex_clip_for_attribution')


def window_diagnostic(oracle,source,seed,seconds=8.,*,sides=(64,128,256)):
    if any(side not in (64,128,256) for side in sides):raise ValueError('Diagnostic window sides: 64, 128, 256 only')
    rng=np.random.default_rng(seed);began=time.perf_counter();rows=[];external=Counter()
    for side in sides:
        accepted=0;proposed=0;probe_counts=Counter();attribution_counts=Counter()
        # Count actual shadow attribution queries, independently of proposal
        # and probe metadata. Metadata must never consume another side's cap.
        shadow_checks=0
        for attempt in range(8):
            if time.perf_counter()-began>seconds:break
            origin=rng.integers(0,1025-side,3);lo=[Q(int(v),1024) for v in origin];hi=[Q(int(v+side),1024) for v in origin]
            sq=[Q(float(v)) for v in source]
            if all(a<=p<b for a,p,b in zip(lo,sq,hi)):
                external[f'side_{side}_source_excluded_proposals']+=1;continue
            proposed+=1;found={}
            for offset in rng.integers(0,side,(16,3)):
                voxel=origin+offset;p=(voxel+.5)/1024
                membership=oracle.membership(p).status
                external[f'side_{side}_membership_{membership}']+=1
                if membership!='FREE':continue
                d=oracle.segment(source,p);probe_counts[d.status]+=1
                if not d.certified:continue
                found.setdefault(d.status,list(map(int,voxel)))
                if d.status=='HIT' and shadow_checks<32:
                    shadow_checks+=1
                    pq=[Q(float(v)) for v in p];entry=Q(0)
                    for s,x,l,h in zip(sq,pq,lo,hi):
                        delta=x-s
                        if delta>0:entry=max(entry,(l-s)/delta)
                        elif delta<0:entry=max(entry,(h-s)/delta)
                    e=[s+(x-s)*entry for s,x in zip(sq,pq)]
                    if isinstance(oracle,ConvexOracle):suffix=convex_fraction_segment(oracle.context,e,pq)
                    else:
                        eq=tuple(rational(v.numerator)/rational(v.denominator) for v in e)
                        suffix=oracle.segment(eq,p)
                    if suffix.status=='MISS':key='external_blocking_proved_inside_segment_free'
                    elif suffix.status=='HIT':key='inside_segment_blocks_first_location_unknown'
                    else:key='unresolved'
                    external[key]+=1;attribution_counts[key]+=1
            if 'HIT' in found and 'MISS' in found:accepted+=1
            rows.append({'side':side,'origin':origin.tolist(),'witnesses_xyz':found,'informative_witness_proved':len(found)==2})
        external[f'side_{side}_proposed']=proposed
        external[f'side_{side}_informative']=accepted
        for key,value in probe_counts.items():external[f'side_{side}_probe_{key}']=value
        external[f'side_{side}_shadow_attribution_queries']=shadow_checks
        for key,value in attribution_counts.items():external[f'side_{side}_attribution_{key}']=value
    return {'scope':'bounded actual 1024-grid point witnesses, not complete window responses or production quotas',
            'wall_seconds':time.perf_counter()-began,'counts':dict(external),'windows':rows,
            'soft_time_budget_seconds':seconds,'time_budget_reached':time.perf_counter()-began>seconds,
            'attribution_policy':'at most 32 shadow suffix checks per exact side, independent of proposal/probe counters',
            'external_semantics':'full closed segment HIT and exact window-entry-to-free-receiver suffix MISS; no first-hit-distance inference'}


def field_benchmark(scene,oracle,source,casepath,cfg):
    result=[]
    for n in cfg['field_resolutions']:
        value=small_field(oracle,source,n,cfg['field_seconds'])
        data={k:value.pop(k) for k in ('response','solid','valid')}
        path=casepath/f'field-{n}.npz';np.savez_compressed(path,**data)
        value.update(resolution=n,storage_bytes=path.stat().st_size)
        value['preprocessing']='same canonical scene and source prepared by preceding query diagnostics; initialization reported separately'
        if value['complete']:
            # Dense independent recheck is bounded separately and never
            # represented as completed if its own budget expires.
            reference=None if isinstance(oracle,ConvexOracle) else Reference(scene,seconds=2.)
            check=small_field(oracle,source,n,cfg['field_seconds'],independent=True if reference is None else reference)
            common=data['valid']&check['valid']
            mismatch=int(np.count_nonzero((data['response']!=check['response'])&common)+np.count_nonzero((data['solid']!=check['solid'])&common))
            value['independent_check']={k:v for k,v in check.items() if k not in ('response','solid','valid')}
            value['independent_check'].update(compared_voxels=int(common.sum()),mismatched_response_or_solid_bits=mismatch)
            if mismatch:raise AssertionError('Complete-field independent disagreement')
            if not check['complete']:result.append(value);break
        result.append(value)
        if not value['complete']:break
    return result


def run_case(model,ordinal,cfg,output,fields):
    case=f'{model}-{ordinal:03d}';path=output/case;path.mkdir(exist_ok=True)
    regimes=list(cfg['occupancy_targets']);regime=regimes[(ordinal//len(cfg['styles']))%len(regimes)]
    target=cfg['occupancy_targets'][regime];t=time.perf_counter();cpu=time.process_time()
    result={'case':case,'model':model,'ordinal':ordinal,'regime':regime,'target_occupied_ratio':target,
            'status':'INCOMPLETE','ready_capable':False,'publication':'EXPERIMENTAL_ONLY'}
    try:
        scene=generate(model,cfg['master_seed'],ordinal,target,cfg);result['generation_seconds']=time.perf_counter()-t
        save(path/'geometry.json',scene);result['geometry_id']=record(scene)['geometry_id']
        result['geometry_json_bytes']=(path/'geometry.json').stat().st_size
        result['quality']=quality(scene,cfg['quality_resolution'])
        result['density_target_deficit']=max(0.,target-result['quality']['occupied_ratio_estimate'])
        t2=time.perf_counter();limits=Limits(max_nodes=cfg['query_node_budget'],max_depth=cfg['query_depth_limit'],seconds=cfg['query_seconds'])
        oracle=make_oracle(scene,limits=limits);result['oracle_initialization_seconds']=time.perf_counter()-t2
        result['containment']=oracle.containment(cfg['world_margin'])
        if result['containment']['status']!='CERTIFIED':
            result['status']='CONTAINMENT_UNRESOLVED';return result
        if not isinstance(oracle,ConvexOracle):
            peak=point(result['quality']['approx_peak_xyz']);eps=rational(2**-20)
            interior=oracle.classify_box(tuple(x-eps for x in peak),tuple(x+eps for x in peak))
            result['nonempty_interior']={'status':interior.status,'cube_half_side':2**-20,'center':list(map(float,peak)),
                                         'proof':'entire positive-width closed cube is SOLID; not a global topology proof'}
            if interior.status!='SOLID':
                result['status']='INTERIOR_UNRESOLVED';return result
        source,source_report=sample_source(oracle,cfg['master_seed']+ordinal);result['source_sampling']=source_report
        if source is None:result['status']='SOURCE_BUDGET_INCOMPLETE';return result
        result['source_xyz_synthetic']=source.tolist()
        result['queries']=query_benchmark(scene,oracle,source,ordinal,cfg)
        if ordinal in cfg['field_case_ordinals']:
            result['windows']=window_diagnostic(oracle,source,cfg['master_seed']+ordinal*31)
            if fields:result['fields']=field_benchmark(scene,oracle,source,path,cfg)
        result['status']='RESEARCH_QUERIES_COMPLETE'
        return result
    finally:
        result['total_seconds']=time.perf_counter()-t;result['cpu_seconds']=time.process_time()-cpu
        result['process_peak_RSS_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        atomic_json(path/'result.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='experiments/gaussian_geometry/configs/study-v1.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scenes-per-model',type=int)
    parser.add_argument('--models',nargs='+')
    parser.add_argument('--fields',action='store_true')
    parser.add_argument('--case-ordinals',nargs='+',type=int)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text())
    if args.scenes_per_model is not None:cfg['scenes_per_model']=args.scenes_per_model
    if args.models:cfg['models']=args.models
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    identity={'config':cfg,'source_sha256':source,'fields_enabled':args.fields,'ordinals':args.case_ordinals,
              'base_git_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'versions':{n:importlib.metadata.version(n) for n in ('numpy','python-flint','gmpy2','scipy')},
              'python':sys.version,'source_precision':'exact binary64 canonical inputs','public_name':'RIND3D'}
    args.output.mkdir(parents=True,exist_ok=True);manifest=args.output/'run-manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=identity:raise ValueError('Run identity changed; choose a new output directory')
    else:
        atomic_json(manifest,identity);snapshot=args.output/'source-snapshot';snapshot.mkdir()
        for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    all_results=[]
    for model in cfg['models']:
        for ordinal in args.case_ordinals or range(cfg['scenes_per_model']):
            path=args.output/f'{model}-{ordinal:03d}'/'result.json'
            if path.exists():r=json.loads(path.read_text())
            else:
                try:r=run_case(model,ordinal,cfg,args.output,args.fields)
                except Exception as e:
                    r=json.loads(path.read_text()) if path.exists() else {}
                    r.update(case=f'{model}-{ordinal:03d}',status='EXCEPTION_INCOMPLETE',exception=type(e).__name__,message=str(e));atomic_json(path,r)
                    if isinstance(e,AssertionError):raise
            all_results.append(r)
            print(json.dumps({'case':r.get('case'),'status':r['status'],'seconds':r.get('total_seconds'),'occupancy':r.get('quality',{}).get('occupied_ratio_estimate')}),flush=True)
    atomic_json(args.output/'summary.json',{'cases':all_results,'mass_production':False,'new_READY_scenes':0})


if __name__=='__main__':main()
