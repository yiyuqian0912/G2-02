"""B2-only joint parameter proposals. No clipping or production registration."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..generators import DEFAULT,layout,grid_points
from ..models import Kernel,KernelScene,Threshold,record,save
from ..study import atomic_json,quality,sample_source,window_diagnostic
from ..queries import FieldOracle
from .api import NativeOracle
from .containment import certify,independently_recheck
from .build import sources


def proposals(seed,ordinal,target,cfg):
    base,metadata=layout(seed,ordinal)
    points=grid_points(cfg['calibration_resolution'])
    for contraction in cfg['center_contractions']:
        centers=[.5+contraction*(k.center-.5) for k in base]
        gauge=np.full(len(points),np.inf)
        for c,k in zip(centers,base):
            q=(points-c)@k.approx_inverse.T;gauge=np.minimum(gauge,np.linalg.norm(q,axis=1))
        desired=float(np.quantile(gauge,target))
        for factor in cfg['width_factors']:
            scale=desired*factor
            ks=[Kernel(c,k.matrix*scale,k.weight,k.cluster,k.scale_class) for c,k in zip(centers,base)]
            meta=dict(metadata,distribution_version=cfg['version'],center_contraction=contraction,
                      kernel_width_factor=factor,actual_global_matrix_scale=scale,
                      target_occupied_ratio=target,approximate_calibration_resolution=cfg['calibration_resolution'],
                      no_world_clipping=True)
            scene=KernelScene('gaussian_sum',ks,Threshold('binary64',1.),meta)
            score=scene.approx_values(points)+1.
            tau=max(float(np.quantile(score,1-target)),np.nextafter(0.,1.))
            scene.threshold=Threshold('binary64',tau)
            yield scene


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('experiments/gaussian_geometry/configs/b2-generation-v1.json'))
    p.add_argument('--cases',type=int);args=p.parse_args();cfg=json.loads(args.config.read_text())
    if args.cases is not None:cfg['cases']=args.cases
    identity={'config':cfg,'native_sources':sources(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'containment_sha256':hashlib.sha256(Path(__file__).with_name('containment.py').read_bytes()).hexdigest()}
    args.output.mkdir(parents=True,exist_ok=True);manifest=args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Generation study identity changed')
    atomic_json(manifest,identity);rows=[]
    for ordinal in range(cfg['cases']):
        path=args.output/f'case-{ordinal:03d}';path.mkdir(exist_ok=True);checkpoint=path/'result.json'
        if checkpoint.exists():rows.append(json.loads(checkpoint.read_text()));continue
        target=cfg['targets'][(ordinal//6)%len(cfg['targets'])];began=time.perf_counter();cpu=time.process_time()
        row={'ordinal':ordinal,'target':target,'status':'INCOMPLETE','proposals':[],'ready_capable':False}
        for attempt,scene in enumerate(proposals(cfg['seed'],ordinal,target,cfg)):
            t=time.perf_counter();engine=NativeOracle(scene);reference=NativeOracle(scene,independent=True)
            proof=certify(scene,engine,cfg['world_margin'],max_nodes=cfg['containment_nodes'],max_depth=cfg['containment_depth'],seconds=cfg['containment_seconds'])
            report={'attempt':attempt,'center_contraction':scene.metadata['center_contraction'],'width_factor':scene.metadata['kernel_width_factor'],
                    'geometry_id':record(scene)['geometry_id'],'containment_status':proof['status'],'containment_kind':proof['kind'],
                    'seconds':time.perf_counter()-t}
            row['proposals'].append(report)
            atomic_json(path/f'containment-{attempt:02d}.json',proof)
            if proof['status']=='UNKNOWN':
                # Budget is not a physical rejection or permission to select
                # another easier geometry. Keep the same case incomplete.
                row['status']='INCOMPLETE_CONTAINMENT_BUDGET';break
            if proof['status']!='CERTIFIED':
                if proof['status']=='OUTSIDE_SOLID_PROVED':
                    witness=reference.membership(proof['witness_xyz']);report['independent_outside_witness']=witness.status
                    if witness.status!='SOLID':row['status']='INCOMPLETE_INDEPENDENT_OUTSIDE_WITNESS';break
                continue
            checked=independently_recheck(scene,proof,reference);report['independent_containment']=checked
            if checked['status']!='CERTIFIED':row['status']='INCOMPLETE_INDEPENDENT_CONTAINMENT';break
            q=quality(scene,cfg['quality_resolution']);peak=np.array(q['approx_peak_xyz']);radius=2**-20
            # Shape diagnostic only: how much sampled solid requires SUM
            # contributions rather than any individual g_i >= shared tau.
            points=grid_points(cfg['quality_resolution']);total=np.zeros(len(points));largest=np.zeros(len(points))
            for kernel in scene.kernels:
                local=(points-kernel.center)@kernel.approx_inverse.T
                contribution=kernel.weight*np.exp(-np.einsum('ij,ij->i',local,local)/2)
                total+=contribution;largest=np.maximum(largest,contribution)
            solid=total>=scene.threshold.value
            q['fusion_only_solid_fraction_estimate']=float(np.mean(largest[solid]<scene.threshold.value)) if solid.any() else None
            interior=engine.classify_box(np.nextafter(peak-radius,-np.inf),np.nextafter(peak+radius,np.inf))
            if interior.status!='SOLID':row['status']='INCOMPLETE_INTERIOR_PROOF';break
            source,sreport=sample_source(engine,cfg['seed']+ordinal)
            if source is None:row['status']='INCOMPLETE_SOURCE_BUDGET';break
            save(path/'geometry.json',scene)
            row.update(status='RESEARCH_GEOMETRY_QUERIES_ACCEPTED',geometry_id=record(scene)['geometry_id'],quality=q,
                       target_deficit=max(0.,target-q['occupied_ratio_estimate']),source_xyz_synthetic=source.tolist(),source_sampling=sreport,
                       positive_width_interior_half_side=radius,selected_attempt=attempt)
            # Rational window-entry attribution intentionally retains the old
            # Arb oracle. Native endpoint APIs reject nonbinary rationals.
            if ordinal in (0,7,14,21):row['windows']=window_diagnostic(FieldOracle(scene),source,cfg['seed']+ordinal*31,seconds=12.)
            break
        else:row['status']='ALL_DECLARED_PROPOSALS_PROVED_OUTSIDE_MARGIN'
        row.update(total_seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu)
        atomic_json(checkpoint,row);rows.append(row)
        print(json.dumps({'ordinal':ordinal,'status':row['status'],'proposals':len(row['proposals']),
                          'occupancy':row.get('quality',{}).get('occupied_ratio_estimate'),'target':target}),flush=True)
    atomic_json(args.output/'summary.json',{'cases':rows,'identity':identity,'new_READY_scenes':0,
              'selection_policy':'first proved-contained proposal; UNKNOWN stops its case, never a speed-based replacement'})


if __name__=='__main__':main()
