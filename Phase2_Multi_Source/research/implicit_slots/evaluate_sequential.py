"""Autonomous sequential evaluation: count + actual single-layer accuracy."""
import gzip,json,os,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from train_decomposition import make_model,resolve,write_json,digest,semantic_config
from mask_only_data import MaskRecords
from mask_validation import validate_mask_cache
from decomposition_data import decomposed_batches
from sequential_objective import activation_assignment,primary_counts,summarize_primary
from failure_diagnostics import rate_summary


def _evaluate_one(config,output,index,seed=0,repetitions=1000,*,split='test',checkpoint_kind='best',reference=None):
    if split not in ('validation','test'):raise ValueError('invalid evaluation split')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    cuda=config['training']['device']=='cuda';device=torch.device(f'cuda:{local}' if cuda else 'cpu')
    if cuda:
        torch.cuda.set_device(device);torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=True
    torch.set_num_threads(max(1,config['training'].get('num_threads',8)//world))
    if world>1 and not dist.is_initialized():dist.init_process_group('nccl' if cuda else 'gloo')
    validate_mask_cache(index,output/'mask_cache',config['model']['max_slots'])
    from evaluation_snapshot import load_snapshot
    if reference is None:
        saved,metadata=load_snapshot(output/'models'/f'seed_{seed}'/(checkpoint_kind+'.pt'),output/'diagnostics',rank,world)
        signature_index=index
        metadata.update(checkpoint_role='current_'+checkpoint_kind)
    else:
        from evaluation_snapshot import sha256_file
        metadata={**reference['best'],'checkpoint_role':'previous_final_best'}
        if sha256_file(metadata['snapshot_path'])!=metadata['snapshot_sha256']:
            raise ValueError('retained reference weights changed')
        saved=torch.load(metadata['snapshot_path'],map_location='cpu',weights_only=True)
        signature_index=json.loads((Path(reference['source_root'])/'mask_records.json').read_text())
        if digest(signature_index)!=reference['source_index_sha256']:
            raise ValueError('reference index changed after weights were retained')
    signed={k:v for k,v in signature_index.items() if k!='statistics'}
    if saved['signature']!=digest({'config':semantic_config(saved['config']),'index':signed}):
        raise ValueError('checkpoint/cache/config signature mismatch')
    if reference is None and saved['config']['model']!=config['model']:raise ValueError('checkpoint/model mismatch')
    metadata['architecture']=saved['config']['model'].get('decomposition_architecture','legacy')
    metadata['model_selection']=saved['config']['training'].get('model_selection')
    model=make_model(saved['config'],index['global_size'],device,False)
    model.load_state_dict(saved['model'],strict=True);model.eval()
    if cuda:
        from sequential_objective import verify_effective_assignment
        verify_effective_assignment(device)
    rows_in=index['records'][split];records=MaskRecords(rows_in,output/'mask_cache',split,config['model']['max_slots'])
    if cuda and config['training'].get('gpu_inputs', 'off') != 'off':
        from gpu_mask_inputs import enable_gpu_inputs
        enable_gpu_inputs(records, device, config['training'])
    counts=torch.zeros(33,4,device=device,dtype=torch.int64);rows=[];started=time.perf_counter()
    with torch.inference_mode():
        for number,batch in enumerate(decomposed_batches(records,config['training']['batch_size'],
                prefetch=config['training'].get('prefetch_batches',2),rank=rank,world_size=world,training=False),1):
            batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=cuda and config['training']['amp_dtype']=='bfloat16'):
                out=model(batch['response'],batch['obstacle'],batch['window'])
            if 'active_slots' not in out:
                from decomposition_model import slot_activation
                out['active_slots']=slot_activation(out)
            out['predicted_count']=out['active_slots'].sum(1)
            assignment=activation_assignment(out,batch)
            counts+=primary_counts(out,batch,assignment=assignment)
            m=out['mask_probability'].gather(1,assignment[...,None,None].expand_as(out['mask_probability']))>=.5
            m &= out['active_slots'].gather(1,assignment)[...,None,None]
            t=batch['teacher_mask'].bool();free=(~batch['obstacle'])[:,None]
            inter=(m&t&free).sum((-1,-2));union=((m|t)&free).sum((-1,-2)).clamp_min(1)
            passed=(((inter/union)>=.9)&batch['layer_mask']).sum(1)
            has_constraints='constraint_accepted' in out
            if has_constraints:
                summed=(out['mask_probability'].double()*out['intensity'][...,None,None]).sum(1)
                closure=(summed-batch['response'].double()).abs().amax((-1,-2))
                if bool((out['residual'].amin((-1,-2))<0).any()):raise RuntimeError('negative residual violates decoder invariant')
                if bool((out['constraint_accepted'] & (closure>out['constraint_tolerance']*1.01)).any()):
                    raise RuntimeError('accepted decomposition violates conservation')
                audit=torch.stack((out['constraint_accepted'],closure,out['constraint_tolerance'],out['executed_layers']),1).double().cpu().tolist()
            else:audit=[[None,None,None,None] for _ in range(len(passed))]
            packed=torch.stack((batch['scene_id'],batch['view_id'],batch['window'][:,2],batch['layer_mask'].sum(1),
                out['predicted_count'],passed),1).double().cpu().tolist()
            for values,checks in zip(packed,audit):
                scene,view,size,r,pr,good=values;accepted,error,tol,executed=checks
                rows.append({'scene_id':int(scene),'view_id':int(view),'size':int(size),'layer_count':int(r),
                    'predicted_count':int(pr),'count_correct':r==pr,'shape_correct':int(good),'layers':int(r),
                    'constraint_accepted':bool(accepted) if accepted is not None else None,'max_abs_conservation_error':error,
                    'roundoff_tolerance':tol,'executed_layers':int(executed) if executed is not None else None})
            if rank==0 and (number==1 or number%20==0):print(f'sequential evaluation {split}: batches={number} local windows={len(rows)}',flush=True)
    if world>1:
        dist.all_reduce(counts);collected=[None]*world;dist.all_gather_object(collected,rows)
        rows=[r for part in collected for r in part]
    if rank:return None
    if len(rows)!=len(records) or len({(r['scene_id'],r['view_id']) for r in rows})!=len(rows):
        raise ValueError('missing/duplicate evaluation windows')
    rows.sort(key=lambda r:(r['scene_id'],r['view_id']))
    rates=rate_summary(rows,{'count_accuracy':[r['count_correct'] for r in rows],
                            'layer_accuracy':[r['shape_correct'] for r in rows]},
                           {'count_accuracy':[1]*len(rows),'layer_accuracy':[r['layers'] for r in rows]},repetitions,seed)
    summary=summarize_primary(counts.cpu())
    report={'schema':'rind-primary-model-evaluation-v1','checkpoint':metadata,'epoch':saved['epoch'],'split':split,
        'matching_protocol':'enabled-shape-only-empty-padding-compacted-v2',
        'windows':len(rows),'scenes':len({r['scene_id'] for r in rows}),'candidate_renders':0,
        'inference_teacher_inputs':False,'metrics':rates,'by_R':summary['primary_by_R'],
        'by_group':summary['primary_by_group'],'by_size':{},
        'constraint_audit':({'available':True,'accepted':sum(r['constraint_accepted'] for r in rows),'total':len(rows),
                            'negative_residuals':0,'rejected_windows_remain_in_metrics':True,
                            'tolerance':'roundoff_ulps * float32_eps * window_max; per pixel, not 5%'} if has_constraints else
                            {'available':False,'reason':'previous architecture has no hard conservation decoder'}),
        'elapsed_seconds':time.perf_counter()-started}
    for size in sorted({r['size'] for r in rows}):
        selected=[r for r in rows if r['size']==size]
        report['by_size'][str(size)]={'windows':len(selected),
            'count_accuracy':sum(r['count_correct'] for r in selected)/len(selected),
            'layer_accuracy':sum(r['shape_correct'] for r in selected)/sum(r['layers'] for r in selected)}
    destination=output/'evaluation'/f'seed_{seed}';destination.mkdir(parents=True,exist_ok=True)
    if reference is not None:destination=destination/'reference_final';destination.mkdir(parents=True,exist_ok=True)
    elif checkpoint_kind=='last':destination=destination/f'last_epoch_{saved["epoch"]}';destination.mkdir(parents=True,exist_ok=True)
    stem='primary_metrics' if split=='test' else 'validation_primary_metrics'
    report['rows_file']=str(destination/(stem+'_windows.jsonl.gz'))
    write_json(destination/(stem+'.json'),report)
    with gzip.open(destination/(stem+'_windows.jsonl.gz'),'wt') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    lines=['# 分层模型：两个核心指标','',f'checkpoint={checkpoint_kind}, epoch={saved["epoch"]}, split={split}, windows={len(rows)}。',
           '','推断只输入原始响应与窗口位置；没有教师中间残差、教师层数、教师强度输入。',
           '','| 指标 | 总体 | 场景等权 | 场景聚类95%区间 |','|---|---:|---:|---|']
    for k,label in (('count_accuracy','层数预测准确率'),('layer_accuracy','单层形状准确率 IoU≥0.90')):
        v=rates[k];ci=v['window_cluster_95ci'];interval=(f"[{ci['lower']:.2%}, {ci['upper']:.2%}]" if ci else 'NA')
        lines.append(f"| {label} | {v['window_weighted']:.2%} | {v['scene_balanced']:.2%} | {interval} |")
    lines+=['','| 层数组 | 视窗 | 层数准确率 | 单层形状准确率 |','|---|---:|---:|---:|']
    for k,v in report['by_group'].items():
        if v['windows']:lines.append(f"| {k} | {v['windows']} | {v['count_accuracy']:.2%} | {v['layer_accuracy']:.2%} |")
    if has_constraints:
        lines+=['',f"约束验收：{report['constraint_audit']['accepted']}/{len(rows)}；负残差=0。未通过验收的窗口仍计入上述两个指标。",
                '验收失败表示未完成。浮点缓存只允许舍入级误差，不添加任意最后一层。']
    else:lines+=['','旧模型无新守恒解码器，仅按同口径两个主要指标重新评估。']
    (destination/(stem+'.zh-CN.md')).write_text('\n'.join(lines)+'\n')
    print(f'Sequential primary report: {destination/(stem+".json")}',flush=True)
    return report


def evaluate_with_reference(config,output,index,seed=0,repetitions=1000,*,split='test',checkpoint_kind='best'):
    """After training, re-evaluate the previous FINAL best under identical rules."""
    current=_evaluate_one(config,output,index,seed,repetitions,split=split,checkpoint_kind=checkpoint_kind)
    if checkpoint_kind!='best' or not config.get('cache_source_root') or not config['training'].get('compare_previous_checkpoint',True):
        return current
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    from model_comparison import final_reference,read_rows,compare_reports
    reference=final_reference(config,output,index,seed,rank,world)
    destination=output/'evaluation'/f'seed_{seed}'
    stem='reference_comparison' if split=='test' else 'validation_reference_comparison'
    pending=destination/(stem+'_pending.json')
    if reference['status']!='ready':
        if rank==0:
            write_json(pending,reference)
            print(f"Model comparison pending: {reference['reason']}; {reference['retry_command']}",flush=True)
        return current
    previous=_evaluate_one(config,output,index,seed,repetitions,split=split,reference=reference)
    if rank==0:
        compare_reports(current,previous,read_rows(current['rows_file']),read_rows(previous['rows_file']),
                        reference,destination,stem,repetitions,seed)
        pending.unlink(missing_ok=True)
        print(f'Paired model comparison: {destination/(stem+".zh-CN.md")}',flush=True)
    return current
