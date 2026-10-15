"""Compare the complete v4 local-only and v5 full-geometry evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path):
    path=Path(path)
    if path.is_dir():
        path=path/'evaluation'/'comprehensive_metrics.json'
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text()),path


def _get(report,path):
    value=report
    for key in path:
        value=value[key]
    if isinstance(value,dict) and 'mean' in value:
        value=value['mean']
    return float(value)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=Path('outputs/layer_student_v4'))
    parser.add_argument('--geometry',type=Path,default=Path('outputs/layer_student_v5_geometry'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    baseline,baseline_path=_load(args.baseline)
    geometry,geometry_path=_load(args.geometry)
    metrics={
        'Test cross entropy':(('training_quality','test_cross_entropy'),'lower'),
        'Forward KL':(('distribution_metrics','combined_max_field','forward_kl'),'lower'),
        'Information recovery':(('training_quality','test_normalized_information_recovery'),'higher'),
        'Dice@90':(('distribution_metrics','combined_max_field','support_dice_90'),'higher'),
        'Recall@90':(('distribution_metrics','combined_max_field','support_recall_90'),'higher'),
        'Precision@90':(('distribution_metrics','combined_max_field','support_precision_90'),'higher'),
        'Invalid-candidate mass':(('distribution_metrics','combined_max_field','student_mass_on_teacher_invalid_candidates'),'lower'),
        'Student mean source distance':(('coordinate_metrics','overall','student_mean_matched_distance'),'lower'),
        'Student Recall@32':(('coordinate_metrics','overall','student_recall_at_32'),'higher'),
    }
    lines=['# Full-scene geometry ablation','',
           f'- Baseline: `{baseline_path}`',f'- Geometry: `{geometry_path}`','',
           '| Metric | v4 local only | v5 full geometry | Change | Direction |','|---|---:|---:|---:|---|']
    for name,(path,direction) in metrics.items():
        old=_get(baseline,path);new=_get(geometry,path);change=new-old
        lines.append(f'| {name} | {old:.6f} | {new:.6f} | {change:+.6f} | {direction} |')
    lines += ['','Positive change is desirable only for rows marked `higher`; negative change is desirable for `lower`.','']
    text='\n'.join(lines)
    print(text)
    destination=args.output or args.geometry/'evaluation'/'v4_vs_v5_geometry.md'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(text)
    print('Saved:',destination)


if __name__=='__main__':
    main()
