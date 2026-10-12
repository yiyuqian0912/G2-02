"""Summarize one trained combined-teacher experiment with distribution metrics."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve();REPO=HERE.parents[3]
P1=REPO/'Phase1_Single_Source_Single_Intensity';P2=REPO/'Phase2_Multi_Source'
for item in (P1/'src',P2/'src',P2/'vendor/rind-dataset',HERE.parent):
    if str(item) not in sys.path:sys.path.insert(0,str(item))

from train_combined import CombinedRecords,context,read_config


def metrics(teacher,student,candidate_xy,support):
    q=np.asarray(teacher,dtype=np.float64);p=np.asarray(student,dtype=np.float64)
    active=np.asarray(support,dtype=bool)
    if q.shape!=p.shape or active.shape!=q.shape or not np.isclose(q.sum(),1) or not np.isclose(p.sum(),1):
        raise ValueError('misaligned or unnormalized teacher/student distributions')
    positive=q>0
    cross_entropy=float(-np.sum(q[positive]*np.log(p[positive])))
    entropy=float(-np.sum(q[positive]*np.log(q[positive])))
    mixture=.5*(q+p);q_active=q>0;p_active=p>0
    js=.5*float(np.sum(q[q_active]*np.log(q[q_active]/mixture[q_active])))
    js+=.5*float(np.sum(p[p_active]*np.log(p[p_active]/mixture[p_active])))
    tv=.5*float(np.abs(q-p).sum())
    teacher_top=int(np.argmax(q));student_top=int(np.argmax(p))
    top_count=max(1,int(np.ceil(.01*np.count_nonzero(active))))
    teacher_top_set=np.argpartition(q,-top_count)[-top_count:]
    return {'cross_entropy':cross_entropy,'teacher_entropy':entropy,
            'kl_teacher_student':cross_entropy-entropy,'js_divergence':js,
            'total_variation':tv,'distribution_overlap':1-tv,
            'uniform_cross_entropy':float(np.log(np.count_nonzero(active))),
            'gain_over_uniform_ce':float(np.log(np.count_nonzero(active))-cross_entropy),
            'top1_distance_world':float(np.linalg.norm(candidate_xy[teacher_top]-candidate_xy[student_top])),
            'teacher_peak_mass':float(q[teacher_top]),'student_peak_mass':float(p[student_top]),
            'student_mass_on_teacher_top_1pct':float(p[teacher_top_set].sum())}


def aggregate(rows):
    keys=[key for key,value in rows[0].items() if isinstance(value,(int,float))
          and key not in ('index','scene_id','view_id','size','layers')]
    return {key:{'mean':float(np.mean([row[key] for row in rows])),
                 'median':float(np.median([row[key] for row in rows])),
                 'min':float(np.min([row[key] for row in rows])),
                 'max':float(np.max([row[key] for row in rows]))} for key in keys}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    config=read_config(args.config);dataset,root=context(config)
    manifest=json.loads((root/'records.json').read_text())
    records=CombinedRecords(manifest['records']['test'],dataset,root,
        config['training'].get('normalization_support','geometry'),config['teacher'].get('tolerance',1e-12))
    from rind_phase1.train import load_model
    model,checkpoint=load_model(root/'models'/f'seed_{args.seed}'/'best.pt');model.eval()
    import torch
    rows=[]
    with torch.inference_mode():
        for index in range(len(records)):
            sample=records[index];entry=manifest['records']['test'][index]
            prediction=model.predict(
                torch.as_tensor(sample['response'][None],dtype=torch.float32),
                torch.as_tensor(sample['obstacle'][None],dtype=torch.bool),
                torch.as_tensor(sample['window'][None],dtype=torch.float32),
                torch.as_tensor(sample['candidate_xy'][None],dtype=torch.float32),
                torch.as_tensor(sample['support'][None],dtype=torch.bool))['student_prob'][0].numpy()
            row={'index':index,'scene_id':int(entry['scene_id']),'view_id':int(entry['view_id']),
                 'size':int(entry['size']),'layers':int(entry['layers_recovered'])}
            row.update(metrics(sample['teacher_prob'],prediction,sample['candidate_xy'],sample['support']))
            rows.append(row)
    groups={}
    for field in ('size','layers'):
        buckets=defaultdict(list)
        for row in rows:buckets[row[field]].append(row)
        groups[field]={str(key):{'count':len(values),'metrics':aggregate(values)}
                       for key,values in sorted(buckets.items())}
    training=json.loads((root/'models'/f'seed_{args.seed}'/'training.json').read_text())
    summary={'experiment_root':str(root.resolve()),'seed':args.seed,'best_epoch':training['best_epoch'],
             'best_validation_cross_entropy':training['best_validation_cross_entropy'],
             'reported_test_cross_entropy':training['test_cross_entropy'],
             'record_counts':manifest['counts'],'skip_counts':manifest['skip_counts'],
             'test_count':len(rows),'overall':aggregate(rows),'groups':groups}
    output=args.output or root/'models'/f'seed_{args.seed}'/'results_summary';output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (output/'test_records.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plot(output,training,rows)
    overall=summary['overall']
    lines=['# Phase2 训练结果','',f'- 最佳 epoch：{summary["best_epoch"]}',
           f'- validation cross-entropy：{summary["best_validation_cross_entropy"]:.6f}',
           f'- test cross-entropy：{summary["reported_test_cross_entropy"]:.6f}',
           f'- 教师记录 train/validation/test：{manifest["counts"]["train"]}/{manifest["counts"]["validation"]}/{manifest["counts"]["test"]}',
           f'- 测试集平均 KL(teacher||student)：{overall["kl_teacher_student"]["mean"]:.6f}',
           f'- 测试集平均 JS divergence：{overall["js_divergence"]["mean"]:.6f}',
           f'- 测试集平均 total variation：{overall["total_variation"]["mean"]:.6f}',
           f'- 测试集平均分布重叠量：{overall["distribution_overlap"]["mean"]:.6f}',
           f'- 相对同支持均匀分布的平均 CE 改善：{overall["gain_over_uniform_ce"]["mean"]:.6f}',
           '', '详细逐视窗数值见 `test_records.csv`，按窗口大小和恢复层数的分组见 `summary.json`。']
    (output/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n')
    print(output)


def plot(output,training,rows):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    history=training['history'];epochs=[x['epoch'] for x in history]
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    axes[0].plot(epochs,[x['train_cross_entropy'] for x in history],label='train')
    axes[0].plot(epochs,[x['validation_cross_entropy'] for x in history],label='validation')
    axes[0].axvline(training['best_epoch'],color='black',linestyle='--',label='best epoch')
    axes[0].set(title='Training curve',xlabel='Epoch',ylabel='Cross-entropy');axes[0].legend()
    axes[1].hist([x['js_divergence'] for x in rows],bins=min(15,len(rows)),color='#6a4c93')
    axes[1].set(title='Test JS divergence',xlabel='JS divergence',ylabel='Windows')
    sizes=sorted(set(x['size'] for x in rows))
    axes[2].boxplot([[x['total_variation'] for x in rows if x['size']==size] for size in sizes],
                    tick_labels=[str(x) for x in sizes])
    axes[2].set(title='Total variation by window size',xlabel='Window size',ylabel='TV distance')
    fig.savefig(output/'training_and_test_metrics.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
