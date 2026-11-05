"""Retain the completed previous run and compare two primary metrics, paired."""
import gzip,hashlib,json
from pathlib import Path
import torch.distributed as dist
from train_decomposition import resolve,write_json,digest
from evaluation_snapshot import create_snapshot
from mask_validation import validate_mask_cache
from failure_diagnostics import rate_summary


def final_reference(config,output,index,seed,rank=0,world=1):
    """Never substitute the earlier warm-start snapshot for the final old best."""
    message=[None]
    if rank==0:
        try:
            source=resolve(config.get('comparison_reference_root',config['cache_source_root'])).resolve()
            if source==output.resolve():raise ValueError('reference experiment must differ from current output')
            run=source/'models'/f'seed_{seed}'
            marker=run/'training.json'
            info={'status':'pending','source_root':str(source),'seed':seed,
                  'reason':'previous_training_not_completed',
                  'retry_command':'./run_implicit_masks_tmux.sh start evaluate --gpus auto --auto-batch'}
            new_marker=output/'models'/f'seed_{seed}'/'training.json'
            new_done=json.loads(new_marker.read_text()) if new_marker.is_file() else {}
            current_complete=all(type(new_done.get(k)) is int for k in ('best_epoch','trained_through_epoch'))
            if not current_complete:info['reason']='current_training_not_completed'
            if current_complete and marker.is_file():
                raw=marker.read_bytes();completed=json.loads(raw)
                # Interim epoch files contain history, but not both final keys.
                if all(type(completed.get(k)) is int for k in ('best_epoch','trained_through_epoch')):
                    old_index=json.loads((source/'mask_records.json').read_text())
                    validate_mask_cache(old_index,source/'mask_cache',old_index['max_slots'])
                    if old_index['corpus_sha256']!=index['corpus_sha256'] or old_index['records']!=index['records']:
                        raise ValueError('reference cache/cohorts differ; refusing to intersect or filter evaluation windows')
                    if (source/'mask_cache').resolve()!=(output/'mask_cache').resolve():
                        raise ValueError('reference/current cohorts must share the verified cache bytes')
                    directory=output/'reference_weights'/f'seed_{seed}'
                    best=create_snapshot(run/'best.pt',directory)
                    last=create_snapshot(run/'last.pt',directory)
                    if (best['epoch']!=completed['best_epoch'] or
                            last['epoch']!=completed['trained_through_epoch'] or marker.read_bytes()!=raw):
                        info['reason']='previous_training_marker_or_weights_changed; retry after completion'
                    else:
                        info={'status':'ready','source_root':str(source),'seed':seed,'best':best,'last':last,
                              'completion':{'best_epoch':completed['best_epoch'],
                                            'trained_through_epoch':completed['trained_through_epoch'],
                                            'training_json_sha256':hashlib.sha256(raw).hexdigest()},
                              'source_index_sha256':digest(old_index),'corpus_sha256':index['corpus_sha256'],
                              'original_best_last_untouched':True,'retained_snapshots_include_optimizer':False}
                        write_json(output/'evaluation'/f'seed_{seed}'/'preserved_reference_weights.json',info)
            message[0]={'reference':info}
        except Exception as error:message[0]={'error':f'{type(error).__name__}: {error}'}
    if world>1:dist.broadcast_object_list(message,src=0)
    if 'error' in message[0]:raise RuntimeError('reference preparation failed: '+message[0]['error'])
    return message[0]['reference']


def read_rows(path):
    with gzip.open(path,'rt') as stream:return [json.loads(line) for line in stream]


def compare_reports(current,reference,current_rows,reference_rows,retention,destination,stem,repetitions,seed):
    if current['split']!=reference['split']:raise ValueError('cannot compare different evaluation splits')
    key=lambda r:(r['scene_id'],r['view_id'])
    now={key(r):r for r in current_rows};old={key(r):r for r in reference_rows}
    if len(now)!=len(current_rows) or len(old)!=len(reference_rows) or set(now)!=set(old):
        raise ValueError('paired comparison requires exactly the same unique windows')
    rows=[]
    for identity in sorted(now):
        n,o=now[identity],old[identity]
        if (n['size'],n['layers'],n['layer_count'])!=(o['size'],o['layers'],o['layer_count']):
            raise ValueError('paired teacher labels/window size differ')
        rows.append({'scene_id':n['scene_id'],'view_id':n['view_id'],'layers':n['layers'],
                     'count_delta':int(n['count_correct'])-int(o['count_correct']),
                     'shape_delta':n['shape_correct']-o['shape_correct']})
    rates=rate_summary(rows,{'count_accuracy':[r['count_delta'] for r in rows],
                            'layer_accuracy':[r['shape_delta'] for r in rows]},
                           {'count_accuracy':[1]*len(rows),'layer_accuracy':[r['layers'] for r in rows]},repetitions,seed)
    metrics={}
    for name in ('count_accuracy','layer_accuracy'):
        expected=current['metrics'][name]['window_weighted']-reference['metrics'][name]['window_weighted']
        if abs(rates[name]['window_weighted']-expected)>1e-12:
            raise ValueError('metric reports do not agree with the paired raw counts')
        metrics[name]={'previous':reference['metrics'][name],'current':current['metrics'][name],
                       'paired_change':rates[name]}
    groups={}
    for name,new in current['by_group'].items():
        before=reference['by_group'][name]
        if (new['windows'],new['layers'])!=(before['windows'],before['layers']):
            raise ValueError('group denominators differ')
        if new['windows']:
            groups[name]={'windows':new['windows'],'layers':new['layers'],
                **{k:{'previous':before[k],'current':new[k],'change':new[k]-before[k]}
                   for k in ('count_accuracy','layer_accuracy')}}
    outcomes={}
    for metric,column in (('count_accuracy','count_delta'),('layer_accuracy','shape_delta')):
        outcomes[metric]={'improved_windows':sum(r[column]>0 for r in rows),
                          'tied_windows':sum(r[column]==0 for r in rows),
                          'worse_windows':sum(r[column]<0 for r in rows)}
    report={'schema':'rind-paired-primary-comparison-v1','split':current['split'],'windows':len(rows),
            'scenes':len({r['scene_id'] for r in rows}),
            'window_list_sha256':digest([(key(now[k]),now[k]['size'],now[k]['layers']) for k in sorted(now)]),
            'teacher_corpus_sha256':retention['corpus_sha256'],'reference_retention':retention,
            'evaluation_code_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('model_comparison.py','evaluate_sequential.py','sequential_objective.py')},
            'previous_checkpoint':reference['checkpoint'],'current_checkpoint':current['checkpoint'],
            'metrics':metrics,'by_group':groups,'window_outcomes':outcomes,
            'matching':'actual active masks, threshold .5, activation-first shape-only Hungarian; missing layers fail',
            'no_constraint_or_confidence_filtering':True,
            'checkpoint_selection_note':'Each run retains its own validation-selected best; this compares complete pipelines, not an isolated architecture change.'}
    write_json(destination/(stem+'.json'),report)
    lines=['# 同一视窗：旧模型与隐式解耦模型对比','',
           f"split={current['split']}；视窗={len(rows)}；scene_id={report['scenes']}。",
           f"旧模型最终 best epoch={reference['epoch']}；新模型 best epoch={current['epoch']}。",'',
           '旧模型来自原训练完成后的best，不是新模型启动时的热启动快照。旧best/last不覆盖，保留副本与SHA见preserved_reference_weights.json。',
           '','| 指标 | 旧模型 | 新模型 | 变化（百分点） | 配对场景bootstrap95%区间（百分点） |',
           '|---|---:|---:|---:|---|']
    for k,label in (('count_accuracy','层数准确率'),('layer_accuracy','单层形状准确率 IoU≥0.90')):
        v=metrics[k];d=v['paired_change'];ci=d['window_cluster_95ci']
        interval=f"[{100*ci['lower']:+.2f}, {100*ci['upper']:+.2f}]" if ci else 'NA'
        lines.append(f"| {label} | {v['previous']['window_weighted']:.2%} | {v['current']['window_weighted']:.2%} | {100*d['window_weighted']:+.2f} | {interval} |")
    lines+=['','| 层数组 | 视窗 | 旧层数准确率 | 新层数准确率 | 旧单层准确率 | 新单层准确率 |',
            '|---|---:|---:|---:|---:|---:|']
    for name,g in groups.items():
        a,b=g['count_accuracy'],g['layer_accuracy']
        lines.append(f"| {name} | {g['windows']} | {a['previous']:.2%} | {a['current']:.2%} | {b['previous']:.2%} | {b['current']:.2%} |")
    lines+=['','两模型在相同响应、教师答案、掩码阈值和实际启用层匹配规则下重新推理。不复用旧报告中不同口径的均值。',
            '未通过新模型守恒验收的窗口仍计入全部分母。区间按scene_id配对重采样，不把每层当独立样本。',
            '两个best沿用各自训练时的验证集选择策略，因此这是完整流程对比，不能把变化全部归因于某一个结构改动。']
    (destination/(stem+'.zh-CN.md')).write_text('\n'.join(lines)+'\n')
    with gzip.open(destination/(stem+'_paired_windows.jsonl.gz'),'wt') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    return report
