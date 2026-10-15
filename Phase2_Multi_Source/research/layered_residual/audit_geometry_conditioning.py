"""Verify that a trained layer student receives and uses full-scene geometry."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve();PACKAGE_ROOT=HERE.parents[3]
PHASE1_ROOT=PACKAGE_ROOT/'Phase1_Single_Source_Single_Intensity'
PHASE2_ROOT=PACKAGE_ROOT/'Phase2_Multi_Source'
for item in (PHASE1_ROOT/'src',PHASE2_ROOT/'src',PHASE2_ROOT/'vendor/rind-dataset',HERE.parent):
    if str(item) not in sys.path:sys.path.insert(0,str(item))

from train_layers import LayerRecords,context,load_verified_index,read_config


def _forward(model,batch,chunk):
    from rind_phase1.model import student_distribution
    encoded=model.encode_observation(
        batch['response'],batch['obstacle'],batch['window'],
        batch['geometry_types'],batch['geometry_params'],batch['geometry_mask'])
    import torch
    energy=torch.cat([
        model.score_encoded(encoded,batch['candidate_xy'][:,start:start+chunk])
        for start in range(0,batch['candidate_xy'].shape[1],chunk)],dim=1)
    return student_distribution(energy.float(),batch['support'])


def _select(entries,maximum):
    by_size=defaultdict(list);seen=defaultdict(set)
    for row in entries:
        size=int(row['size']);scene=int(row['scene_id'])
        if scene in seen[size]:continue
        seen[size].add(scene);by_size[size].append(row)
    sizes=sorted(by_size);quota=max(2,maximum//max(1,len(sizes)))
    selected=[]
    for size in sizes:selected.extend(by_size[size][:quota])
    return selected[:maximum]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='cuda')
    parser.add_argument('--max-layers',type=int,default=128)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--candidate-chunk',type=int,default=4096)
    args=parser.parse_args(argv)
    if args.max_layers<2 or args.batch_size<2 or args.candidate_chunk<1:
        parser.error('max-layers and batch-size must be at least 2; chunk must be positive')
    import torch
    from rind_phase1.model import SourceEnergyField
    from rind_phase1.train import collate_records,load_model
    config=read_config(args.config);dataset,teacher_root,output=context(config)
    index=load_verified_index(config,teacher_root,output)
    checkpoint_path=output/'models'/f'seed_{args.seed}'/'best.pt'
    model,checkpoint=load_model(checkpoint_path)
    if not checkpoint['model_config'].get('use_scene_geometry',False) or not model.use_scene_geometry:
        raise ValueError('checkpoint is not a full-scene geometry model')
    geometry_keys=[name for name in checkpoint['model']
                   if name.startswith(('geometry_encoder.','geometry_attention.',
                                       'geometry_correction_head.'))]
    if not geometry_keys:raise ValueError('checkpoint contains no geometry parameters')
    torch.manual_seed(int(checkpoint.get('seed',args.seed)))
    initial=SourceEnergyField(**checkpoint['model_config']).state_dict()
    squared_delta=sum(float((checkpoint['model'][name]-initial[name]).double().square().sum())
                      for name in geometry_keys)
    device_name=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device_name=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    device=torch.device(device_name);model.to(device).eval()
    selected=_select(index['records']['test'],args.max_layers)
    rows=[];correct_total=shuffled_total=tv_total=count=0
    grouped=defaultdict(list)
    for row in selected:grouped[int(row['size'])].append(row)
    with torch.inference_mode():
        for size,items in sorted(grouped.items()):
            for start in range(0,len(items),args.batch_size):
                subset=items[start:start+args.batch_size]
                if len(subset)<2:continue
                records=LayerRecords(subset,dataset,teacher_root,
                                     config['training']['normalization_support'],True)
                batch=collate_records([records[i] for i in range(len(records))])
                batch={key:value.to(device) for key,value in batch.items()}
                correct_log,correct_p=_forward(model,batch,args.candidate_chunk)
                shuffled=dict(batch)
                for key in ('geometry_types','geometry_params','geometry_mask'):
                    shuffled[key]=torch.roll(batch[key],shifts=1,dims=0)
                shuffled_log,shuffled_p=_forward(model,shuffled,args.candidate_chunk)
                target=batch['teacher_prob']
                correct_ce=-(target*correct_log.masked_fill(~batch['support'],0)).sum(-1)
                shuffled_ce=-(target*shuffled_log.masked_fill(~batch['support'],0)).sum(-1)
                tv=.5*(correct_p-shuffled_p).abs().sum(-1)
                for i,row in enumerate(subset):
                    item={'scene_id':int(row['scene_id']),'view_id':int(row['view_id']),
                          'window_size':size,'step':int(row['step']),
                          'correct_geometry_ce':float(correct_ce[i]),
                          'shuffled_geometry_ce':float(shuffled_ce[i]),
                          'shuffled_minus_correct_ce':float(shuffled_ce[i]-correct_ce[i]),
                          'prediction_tv_after_shuffle':float(tv[i])}
                    rows.append(item)
                correct_total+=float(correct_ce.sum());shuffled_total+=float(shuffled_ce.sum())
                tv_total+=float(tv.sum());count+=len(subset)
    if not count:raise ValueError('not enough distinct-scene same-size layers for the audit')
    differences=np.asarray([row['shuffled_minus_correct_ce'] for row in rows],dtype=np.float64)
    standard_error=(float(differences.std(ddof=1)/math.sqrt(len(differences)))
                    if len(differences)>1 else 0.0)
    by_size={}
    for size in sorted({row['window_size'] for row in rows}):
        group=[row for row in rows if row['window_size']==size]
        by_size[str(size)]={
            'layers':len(group),
            'shuffled_minus_correct_cross_entropy':float(np.mean(
                [row['shuffled_minus_correct_ce'] for row in group])),
            'mean_prediction_total_variation_after_geometry_shuffle':float(np.mean(
                [row['prediction_tv_after_shuffle'] for row in group])),
        }
    report={
        'schema':'full-scene-geometry-conditioning-audit-v1',
        'checkpoint':str(checkpoint_path.resolve()),'checkpoint_epoch':int(checkpoint['epoch']),
        'model_config':checkpoint['model_config'],'geometry_parameter_tensors':len(geometry_keys),
        'geometry_parameter_l2_change_from_initialization':float(np.sqrt(squared_delta)),
        'layers_evaluated':count,'correct_geometry_cross_entropy':correct_total/count,
        'shuffled_geometry_cross_entropy':shuffled_total/count,
        'shuffled_minus_correct_cross_entropy':(shuffled_total-correct_total)/count,
        'mean_prediction_total_variation_after_geometry_shuffle':tv_total/count,
        'paired_difference_standard_error':standard_error,
        'paired_difference_approximate_95pct_interval':[
            float(differences.mean()-1.96*standard_error),
            float(differences.mean()+1.96*standard_error)],
        'fraction_where_correct_geometry_is_better':float(np.mean(differences>0)),
        'by_window_size':by_size,
        'interpretation':{
            'feed_verified':'model configuration and checkpoint contain the geometry branch; every audited batch supplies exact scene obstacle tokens',
            'usage_test':'positive shuffled_minus_correct_cross_entropy means correct scene geometry helps relative to wrong-scene geometry',
            'sensitivity_test':'total variation measures whether predictions react to geometry; useful conditioning additionally requires shuffled geometry to worsen CE consistently'},
        'rows':rows}
    destination=output/'evaluation'/'geometry_conditioning_audit.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='rows'},indent=2))
    print('Saved:',destination)


if __name__=='__main__':main()
