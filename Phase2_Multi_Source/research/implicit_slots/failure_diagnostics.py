"""GPU window/layer diagnostics; original strict retained beside new analyses.

Unconstrained matching can select an inactive duplicate/better slot. We report
that old rule AND activation-first matching of actual predictions. Teacher
substitutions are evaluation-only counterfactuals, never deployment outputs.
"""
import csv,gzip,hashlib,io,json,math,tarfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from pairwise_ops import masked_pairwise_dot
from decomposition_model import exact_batched_assignment,slot_activation


PROTOCOLS={'strict':{'mask_iou':.9,'relative_strength':.1,'relative_reconstruction':.05},
           'relaxed':{'mask_iou':.85,'relative_strength':.15,'relative_reconstruction':.08}}
FACTORS=('count','slot_selection','shape','strength','reconstruction')
VIEWS=('reported','deployed')


def evaluation_pair_cost(output,batch):
    """Self-contained fixed-v5 pair cost, usable by already-running workers."""
    from torch.nn import functional as F
    z=output['mask_logits'].float().flatten(2);m=output['mask_probability'].float().flatten(2)
    target=batch['teacher_mask'].float().flatten(2);free=(~batch['obstacle']).float().flatten(1)
    pixels=free.sum(1).clamp_min(1)
    bce=((F.softplus(z)*free[:,None]).sum(2)[:,None]-masked_pairwise_dot(target, z, free))/pixels[:,None,None]
    inter=masked_pairwise_dot(target, m, free)
    dice=1-(2*inter+1e-6)/((target*free[:,None]).sum(2)[:,:,None]+(m*free[:,None]).sum(2)[:,None]+1e-6)
    scale=batch['response'].float().amax((-1,-2)).clamp_min(1e-12)
    pa=torch.log((output['intensity'].float()/scale[:,None]).clamp_min(1e-12))
    ta=torch.log((batch['teacher_intensity'].float()/scale[:,None]).clamp_min(1e-12))
    strength=F.smooth_l1_loss(pa[:,None].expand_as(bce),ta[:,:,None].expand_as(bce),beta=.1,reduction='none')
    return torch.where(batch['layer_mask'][:,:,None],bce+dice+.1*strength,torch.zeros_like(bce))


@torch.no_grad()
def diagnostic_rows(result,batch,assignment,*,fixed_matching=True):
    masks=result['mask_probability'].float();target=batch['teacher_mask'].bool()
    real=batch['layer_mask'].bool();active=slot_activation(result)
    strengths=result['intensity'].float();truth=batch['teacher_intensity'].float()
    free=(~batch['obstacle']).bool();y=batch['response'].float();b,s,h,w=masks.shape
    cost=evaluation_pair_cost(result,batch)
    # Prioritize real<->active and dummy<->inactive. On unequal counts the
    # minimum mismatch is unavoidable; missing predictions are explicitly zero.
    penalty=(cost.amax((1,2))-cost.amin((1,2)))*s+1
    mismatch=real[:,:,None]!=active[:,None,:]
    deployed=exact_batched_assignment(cost+penalty[:,None,None]*mismatch)
    fixed_assignment=assignment if fixed_matching else exact_batched_assignment(cost)
    assignments={'reported':assignment,'deployed':deployed,'oracle_pool':fixed_assignment}
    denominator=(y*free).sum((-1,-2)).clamp_min(1e-12)
    max_y=y.amax((-1,-2)).clamp_min(1e-12)
    pixels=free.sum((-1,-2)).clamp_min(1)
    relative=lambda value:((value-y).abs()*free).sum((-1,-2))/denominator
    deploy_error=relative(result['reconstruction'])
    count=real.sum(1);pred_count=active.sum(1);count_correct=count==pred_count
    packed={};layers={};gates={};summaries={}
    for view,a in assignments.items():
        pm=masks.gather(1,a[:,:,None,None].expand_as(masks))
        pa=strengths.gather(1,a);on=active.gather(1,a)
        if view=='deployed':
            pm=pm*on[:,:,None,None];pa=pa*on
        hard=pm>=.5;valid=free[:,None]
        inter=(hard&target&valid).sum((-1,-2)).double()
        area_p=(hard&valid).sum((-1,-2)).double();area_t=(target&valid).sum((-1,-2)).double()
        iou=inter/(area_p+area_t-inter).clamp_min(1)
        hard_dice=2*inter/(area_p+area_t).clamp_min(1)
        soft_inter=(pm*target*valid).sum((-1,-2)).double()
        soft_dice=(2*soft_inter+1e-6)/((pm*valid).sum((-1,-2)).double()+area_t+1e-6)
        rel=(pa-truth).abs()/truth.clamp_min(1e-12)
        inside_probability=soft_inter/area_t.clamp_min(1)
        outside_probability=(pm*(~target)*valid).sum((-1,-2)).double()/(free.sum((-1,-2))[:,None]-area_t).clamp_min(1)
        effective_error=(pa*inside_probability-truth).abs()/truth.clamp_min(1e-12)
        layer_data=torch.stack((a.double(),on.double(),iou,hard_dice,soft_dice,pa.double(),truth.double(),
                               rel.double(),area_t,area_p,area_t/(h*w),(truth/max_y[:,None]).double(),
                               inside_probability,outside_probability,effective_error),-1).cpu().numpy()
        layers[view]=layer_data
        present_exact=(on==real).all(1)
        reconstruction=(pm*(pa*real)[:,:,None,None]).sum(1)
        reconstruction_error=relative(reconstruction)
        # Reported/deployed gates always use the ACTUAL output reconstruction.
        eval_error=reconstruction_error if view=='oracle_pool' else deploy_error
        summaries[view]=torch.stack((torch.where(real,iou,torch.inf).amin(1),
            torch.where(real,rel,torch.zeros_like(rel)).amax(1).double(),
            present_exact.double(),reconstruction_error.double()),1).cpu().numpy()
        for protocol,t in PROTOCOLS.items():
            all_shape=((iou>=t['mask_iou'])|~real).all(1)
            all_strength=((rel<=t['relative_strength'])|~real).all(1)
            response_pass=eval_error<=t['relative_reconstruction']
            selection=present_exact if view!='oracle_pool' else torch.ones_like(present_exact)
            success=(selection&all_shape&all_strength&response_pass)
            if view!='oracle_pool':success=success&count_correct
            count_gate=count_correct if view!='oracle_pool' else torch.ones_like(count_correct)
            gates[(view,protocol)]=torch.stack((count_gate,selection,all_shape,all_strength,response_pass,success),1).cpu().numpy()
        if view=='deployed':
            # These substitutions are interpreted ONLY when counts agree;
            # otherwise missing/extra slots make them confounded by cardinality.
            teacher_a=(pm*(truth*real)[:,:,None,None]).sum(1)
            teacher_m=(target*(pa*real)[:,:,None,None]).sum(1)
            packed['teacher_strength_relative_l1']=relative(teacher_a).cpu().numpy()
            packed['teacher_mask_relative_l1']=relative(teacher_m).cpu().numpy()
            packed['hard_mask_relative_l1']=relative((hard*(pa*real)[:,:,None,None]).sum(1)).cpu().numpy()
    teacher_reconstruction=(target*(truth*real)[:,:,None,None]).sum(1)
    scalars=torch.stack((deploy_error,((result['reconstruction']-y).abs()*free).sum((-1,-2))/pixels/max_y,
                         relative(teacher_reconstruction)),1).cpu().numpy()
    scenes=batch['scene_id'].cpu().tolist();views=batch['view_id'].cpu().tolist();windows=batch['window'].cpu().tolist()
    counts=count.cpu().tolist();pred_counts=pred_count.cpu().tolist();real_np=real.cpu().numpy()
    output=[]
    for i in range(b):
        row={'scene_id':int(scenes[i]),'view_id':int(views[i]),'size':int(windows[i][2]),
             'layer_count':counts[i],'predicted_count':pred_counts[i],'count_correct':counts[i]==pred_counts[i],
             'deployment_relative_l1':float(scalars[i,0]),'deployment_peak_normalized_mae':float(scalars[i,1]),
             'teacher_label_closure_relative_l1':float(scalars[i,2]),'views':{},'layers':[]}
        for view in assignments:
            mini,maxa,exact,oracle_error=map(float,summaries[view][i])
            row['views'][view]={'min_layer_iou':mini,'max_layer_relative_strength_error':maxa,
                               'slot_exact':bool(exact),'teacher_matched_reconstruction_relative_l1':oracle_error,
                               'protocols':{}}
            for protocol in PROTOCOLS:
                values=list(map(bool,gates[(view,protocol)][i]))
                flags=dict(zip(('count','slot_exact','shape','strength','reconstruction','success'),values))
                # Slot selection is assessed conditional on the correct count;
                # it is NA, not "passed", when the counts differ.
                flags['slot_selection_given_correct_count']=flags['slot_exact'] if row['count_correct'] else None
                failed=[name for name in FACTORS if
                    (not flags['count'] if name=='count' else
                     row['count_correct'] and not flags['slot_exact'] if name=='slot_selection' else not flags[name])]
                flags['failed_factors']=failed
                if view!='oracle_pool' and (not failed)!=flags['success']:
                    raise ValueError('failure pattern and success disagree')
                row['views'][view]['protocols'][protocol]=flags
            for t in np.flatnonzero(real_np[i]):
                values=layers[view][i,t]
                layer=dict(zip(('slot','active','iou','hard_dice','soft_dice','predicted_strength','teacher_strength',
                                'relative_strength_error','teacher_area','predicted_area','teacher_area_fraction','relative_brightness',
                                'mean_probability_inside_teacher','mean_probability_outside_teacher','effective_inside_strength_error'),map(float,values)))
                layer.update(matching=view,teacher_layer=int(t),slot=int(layer['slot']),active=bool(layer['active']))
                row['layers'].append(layer)
        for key,value in packed.items():row[key]=float(value[i]) if row['count_correct'] and 'active_slots' in result else None
        output.append(row)
    return output


def rate_summary(rows,numerators,denominators,repetitions,seed):
    """Ratios plus scene-ID cluster bootstrap, with explicit denominators."""
    names=list(numerators);n=len(rows)
    x=np.column_stack([numerators[k] for k in names]).astype(float)
    d=np.column_stack([denominators[k] for k in names]).astype(float)
    scenes,inv=np.unique([r['scene_id'] for r in rows],return_inverse=True)
    sx=np.zeros((len(scenes),len(names)));sd=np.zeros_like(sx)
    np.add.at(sx,inv,x);np.add.at(sd,inv,d)
    scene_rates=np.divide(sx,sd,out=np.zeros_like(sx),where=sd>0);eligible=sd>0
    total_x=x.sum(0);total_d=d.sum(0);es=eligible.sum(0)
    window=np.divide(total_x,total_d,out=np.full(len(names),np.nan),where=total_d>0)
    balanced=np.divide(scene_rates.sum(0),es,out=np.full(len(names),np.nan),where=es>0)
    samples_window=[];samples_scene=[]
    if len(scenes)>1 and repetitions>0:
        rng=np.random.default_rng(seed)
        for start in range(0,repetitions,32):
            m=min(32,repetitions-start);draw=rng.integers(0,len(scenes),size=(m,len(scenes)))
            weights=np.zeros((m,len(scenes)))
            np.add.at(weights,(np.arange(m)[:,None],draw),1)
            bx,bd=weights@sx,weights@sd;se=weights@eligible
            samples_window.append(np.divide(bx,bd,out=np.full_like(bx,np.nan),where=bd>0))
            samples_scene.append(np.divide(weights@scene_rates,se,out=np.full_like(bx,np.nan),where=se>0))
    def ci(samples,j):
        if not samples:return None
        values=np.concatenate(samples)[:,j];values=values[np.isfinite(values)]
        if not len(values):return None
        lo,hi=np.quantile(values,[.025,.975]);return {'lower':float(lo),'upper':float(hi),'repetitions':repetitions,'scene_clusters':len(scenes)}
    return {k:{'numerator':int(total_x[j]),'denominator':int(total_d[j]),
               'window_weighted':float(window[j]) if np.isfinite(window[j]) else None,
               'scene_balanced':float(balanced[j]) if np.isfinite(balanced[j]) else None,
               'eligible_scenes':int(es[j]),'window_cluster_95ci':ci(samples_window,j),'scene_cluster_95ci':ci(samples_scene,j)}
            for j,k in enumerate(names)}


def summarize_protocol(rows,view,protocol,repetitions=0,seed=0):
    flags=[r['views'][view]['protocols'][protocol] for r in rows];n=len(rows)
    all_rows=np.ones(n,bool);count=np.array([f['count'] for f in flags]);exact=np.array([f['slot_exact'] for f in flags])
    passed=np.array([f['success'] for f in flags])
    numerators={k:np.array([f[k] for f in flags]) for k in ('count','slot_exact','shape','strength','reconstruction','success')}
    denominators={k:all_rows for k in numerators}
    numerators.update(selection_given_correct_count=count&exact,success_given_correct_count=count&passed,
                      success_given_exact_slots=exact&passed)
    denominators.update(selection_given_correct_count=count,success_given_correct_count=count,success_given_exact_slots=exact)
    failure=Counter();only=Counter();first=Counter();overlap=np.zeros((5,5),int)
    for f in flags:
        bad=f['failed_factors'];key='+'.join(bad) if bad else 'pass';failure[key]+=1
        if len(bad)==1:only[bad[0]]+=1
        first[bad[0] if bad else 'pass']+=1
        for a in bad:
            for b in bad:overlap[FACTORS.index(a),FACTORS.index(b)]+=1
    if sum(failure.values())!=n or sum(first.values())!=n:raise ValueError('failure categories do not exhaust windows')
    for name in FACTORS:
        numerators['failure_'+name]=np.array([name in f['failed_factors'] for f in flags])
        denominators['failure_'+name]=count if name=='slot_selection' else all_rows
    for name in ('shape','strength','reconstruction'):
        good=np.array([f[name] for f in flags])
        numerators[name+'_given_correct_count']=good&count
        denominators[name+'_given_correct_count']=count
        numerators[name+'_given_exact_slots']=good&exact
        denominators[name+'_given_exact_slots']=exact
    rates=rate_summary(rows,numerators,denominators,repetitions,seed)
    def stats(values):
        a=np.asarray(values,float);return {'mean':float(a.mean()),'median':float(np.median(a)),
                                        'p90':float(np.quantile(a,.9)),'p95':float(np.quantile(a,.95)),'max':float(a.max())}
    layers=[a for r in rows for a in r['layers'] if a['matching']==view]
    layer_fail=Counter()
    for layer in layers:
        bad=[]
        if layer['iou']<PROTOCOLS[protocol]['mask_iou']:bad.append('shape')
        if layer['relative_strength_error']>PROTOCOLS[protocol]['relative_strength']:bad.append('strength')
        if bad:
            layer_fail['failed_layers']+=1
            if layer['teacher_area_fraction']<=.01:layer_fail['failed_layers_area_le_1pct']+=1
            if layer['relative_brightness']<=.1:layer_fail['failed_layers_brightness_le_10pct']+=1
    near=[a for a in layers if a['iou']>=PROTOCOLS[protocol]['mask_iou'] and
          a['relative_strength_error']>PROTOCOLS[protocol]['relative_strength']]
    compensated=sum(a['effective_inside_strength_error']<=PROTOCOLS[protocol]['relative_strength'] for a in near)
    strata={'area_fraction':{},'relative_brightness':{}}
    for field,bins in (('area_fraction',(.01,.05,.2)),('relative_brightness',(.1,.3,.6))):
        key='teacher_area_fraction' if field=='area_fraction' else 'relative_brightness'
        buckets=defaultdict(list)
        for layer in layers:
            value=layer[key];label=next(('le_'+str(x) for x in bins if value<=x),'gt_'+str(bins[-1]))
            buckets[label].append(layer)
        for label,items in buckets.items():
            m=sum(a['iou']<PROTOCOLS[protocol]['mask_iou'] for a in items)
            a=sum(a['relative_strength_error']>PROTOCOLS[protocol]['relative_strength'] for a in items)
            strata[field][label]={'layers':len(items),'shape_failures':m,'strength_failures':a,
                                 'shape_failure_rate':m/len(items),'strength_failure_rate':a/len(items)}
    return {'count_partition':{'wrong_count':int((~count).sum()),
                             'correct_count_but_incomplete':int((count&~passed).sum()),'complete':int(passed.sum()),
                             'fraction_of_failures_with_correct_count':float((count&~passed).sum()/(~passed).sum()) if (~passed).sum() else None},
            'softmask_strength_compensation':{'shape_pass_strength_fail_layers':len(near),
                'effective_inside_error_pass_layers':compensated,
                'fraction':compensated/len(near) if near else None,
                'definition':'raw strength fails but mean(mask probability on teacher support)*strength passes; diagnostic hypothesis'},
            'layer_strata':strata,'windows':n,'scenes':len({r['scene_id'] for r in rows}),'rates':rates,
            'exact_failure_patterns':dict(sorted(failure.items(),key=lambda kv:(-kv[1],kv[0]))),
            'only_one_failure':{k:only[k] for k in FACTORS},'multiple_failures':sum(len(f['failed_factors'])>=2 for f in flags),
            'sequential_first_failure':dict(first),'sequential_order':list(FACTORS),'pairwise_failure_overlap':overlap.tolist(),
            'overlap_factor_order':list(FACTORS),'minimum_layer_iou':stats([r['views'][view]['min_layer_iou'] for r in rows]),
            'maximum_layer_strength_relative_error':stats([r['views'][view]['max_layer_relative_strength_error'] for r in rows]),
            'layer_diagnostics':{'layers':len(layers),**layer_fail}}


def aggregate(rows,repetitions=1000,seed=0,metadata=None):
    if not rows or len({(r['scene_id'],r['view_id']) for r in rows})!=len(rows):raise ValueError('empty/duplicate diagnostic windows')
    result={'schema':'rind-decomposition-failure-audit-v1','metadata':metadata or {},'protocols':PROTOCOLS,
            'windows':len(rows),'scenes':len({r['scene_id'] for r in rows}),
            'bootstrap_cluster':'scene_id; repeated geometries across scene IDs are not independently verified',
            'notes':['Failure overlap is observational, not unique causal attribution.',
                     'Slot-selection failure is assessed only when counts agree; otherwise NA.',
                     'Reported matching preserves the original rule; deployed matching prioritizes active slots.',
                     'Teacher substitutions/oracle-pool results are evaluation-only and never deployed.'],
            'overall':{},'by_R':{},'by_R_group':{},'by_size':{},'matching_sensitivity':{},'counterfactuals':{}}
    groups={'by_R':lambda r:str(r['layer_count']),'by_R_group':lambda r:'R>=5' if r['layer_count']>=5 else 'R='+str(r['layer_count']),
            'by_size':lambda r:str(r['size'])}
    for view in VIEWS:
        result['overall'][view]={}
        for protocol in PROTOCOLS:
            print(f'Failure audit: {view}/{protocol} | {len(rows)} windows | scene bootstrap={repetitions}',flush=True)
            result['overall'][view][protocol]=summarize_protocol(rows,view,protocol,repetitions,seed)
        for name,key in groups.items():
            buckets=defaultdict(list)
            for row in rows:buckets[key(row)].append(row)
            result[name][view]={group:{p:summarize_protocol(values,view,p,0,seed) for p in PROTOCOLS}
                                for group,values in sorted(buckets.items())}
    for protocol in PROTOCOLS:
        before=np.array([r['views']['reported']['protocols'][protocol]['success'] for r in rows])
        after=np.array([r['views']['deployed']['protocols'][protocol]['success'] for r in rows])
        oracle=np.array([r['views']['oracle_pool']['protocols'][protocol]['success'] for r in rows])
        result['matching_sensitivity'][protocol]={'reported_fail_deployed_pass':int((~before&after).sum()),
            'reported_pass_deployed_fail':int((before&~after).sum()),'oracle_pool_pass_deployed_fail':int((oracle&~after).sum()),
            'oracle_pool_success':int(oracle.sum()),'denominator':len(rows)}
    eligible=[r for r in rows if r['count_correct'] and r['teacher_strength_relative_l1'] is not None]
    def describe(values):
        return {'mean':float(np.mean(values)),'median':float(np.median(values)),'p90':float(np.quantile(values,.9))} if values else None
    result['counterfactuals']={'condition':'count-correct binary-active-gate windows only; paired same windows','windows':len(eligible),
        **{key:describe([r[key] for r in eligible]) for key in
           ('deployment_relative_l1','teacher_strength_relative_l1','teacher_mask_relative_l1','hard_mask_relative_l1')},
        'teacher_label_closure_relative_l1':describe([r['teacher_label_closure_relative_l1'] for r in rows]),
        'teacher_closure_above_1e_5':sum(r['teacher_label_closure_relative_l1']>1e-5 for r in rows)}
    return result


def write_artifacts(directory,rows,report):
    """Small portable statistics only; never copy observations or weights."""
    from train_decomposition import write_json
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'failure_analysis.json',report)
    with gzip.open(directory/'diagnostic_windows.jsonl.gz','wt',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    window_rows=[];layer_rows=[]
    for row in rows:
        record={k:v for k,v in row.items() if k not in ('views','layers')}
        for view in VIEWS:
            for protocol in PROTOCOLS:
                f=row['views'][view]['protocols'][protocol]
                for key in ('count','slot_exact','slot_selection_given_correct_count','shape','strength','reconstruction','success'):
                    record[f'{view}_{protocol}_{key}']=f[key]
                record[f'{view}_{protocol}_failures']='+'.join(f['failed_factors']) or 'pass'
            record[view+'_minimum_iou']=row['views'][view]['min_layer_iou']
            record[view+'_maximum_strength_error']=row['views'][view]['max_layer_relative_strength_error']
        window_rows.append(record)
        layer_rows.extend({'scene_id':row['scene_id'],'view_id':row['view_id'],'size':row['size'],
                           'layer_count':row['layer_count'],**layer} for layer in row['layers'])
    for name,values in (('windows.csv.gz',window_rows),('layers.csv.gz',layer_rows)):
        with gzip.open(directory/name,'wt',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]) if values else [])
            writer.writeheader();writer.writerows(values)
    lines=['# 分层解耦失败诊断','',f"窗口数：{report['windows']}；scene_id数：{report['scenes']}。",'']
    meta=report['metadata'];lines.extend([f"Checkpoint：`{meta.get('checkpoint_kind')}`；epoch={meta.get('epoch')}；split=`{meta.get('split')}`。",
        f"源checkpoint SHA256：`{meta.get('source_checkpoint_sha256')}`。",'',
        'reported保留原来的全槽匹配；deployed只优先匹配实际启用的槽。两者并行报告，不覆盖旧strict。',
        '计数错误时，槽选择条件项记为NA；shape/strength等可以同时失败。以下不是唯一因果归因。',''] )
    for view in VIEWS:
        for protocol in PROTOCOLS:
            summary=report['overall'][view][protocol];rates=summary['rates']
            lines.extend([f'## {view} / {protocol}','',
                '| 项目 | 通过数/分母 | 窗口等权 | 场景等权 | 窗口率场景聚类95%CI |','|---|---:|---:|---:|---|'])
            for key in ('count','slot_exact','selection_given_correct_count','shape','strength','reconstruction','success',
                        'success_given_correct_count','success_given_exact_slots','shape_given_correct_count',
                        'strength_given_correct_count','reconstruction_given_correct_count'):
                r=rates[key];pct=lambda x:'NA' if x is None else f'{100*x:.2f}%'
                ci=r['window_cluster_95ci'];text='NA' if ci is None else f"[{100*ci['lower']:.2f}%, {100*ci['upper']:.2f}%]"
                lines.append(f"| {key} | {r['numerator']}/{r['denominator']} | {pct(r['window_weighted'])} | {pct(r['scene_balanced'])} | {text} |")
            lines.extend(['','按层数是否正确拆分：',json.dumps(summary['count_partition'],ensure_ascii=False),'',
                          '形状过关但强度失败时的软掩码补偿候选：',json.dumps(summary['softmask_strength_compensation'],ensure_ascii=False),'',
                          '单因素失败计数（slot_selection只在层数正确时判断）：',''])
            for k in FACTORS:lines.append(f"- 仅{k}失败：{summary['only_one_failure'][k]}")
            lines.extend([f"- 多项同时失败：{summary['multiple_failures']}",'',
                          '失败组合（互斥，含pass，计数之和等于全部窗口）：',''])
            lines.extend(f'- {k}: {n}' for k,n in summary['exact_failure_patterns'].items())
            lines.extend(['','按count→slot_selection→shape→strength→reconstruction顺序筛查的首次失败量（依赖顺序）：',
                          '`'+json.dumps(summary['sequential_first_failure'])+'`',''])
    lines.extend(['## 按有效层数R分组（实际启用集合）','','| R组 | 窗口数 | strict | relaxed |','|---|---:|---:|---:|'])
    for group,values in report['by_R_group']['deployed'].items():
        a,b=[values[k]['rates']['success']['window_weighted'] for k in PROTOCOLS]
        lines.append(f"| {group} | {values['strict']['windows']} | {100*a:.2f}% | {100*b:.2f}% |")
    lines.extend(['','## 匹配口径敏感性','',json.dumps(report['matching_sensitivity'],ensure_ascii=False,indent=2),'',
                  'reported失败而deployed通过表示指标受匹配方式影响，不能直接计为模型修复。',
                  'oracle_pool通过而deployed失败表示原始槽池存在一组较好的教师匹配，但实际启用集合未达标；这使用教师信息，仅用于诊断。','',
                  '## 只在层数正确窗口上的替换敏感性','',json.dumps(report['counterfactuals'],ensure_ascii=False,indent=2),'',
                  '教师强度/掩码替换不是部署能力，也不保证误差单调下降；形状与强度误差存在交互。',
                  '置信区间以scene_id聚类；不同scene_id的几何独立性未在此缓存诊断中额外验证。',
                  'strict=(IoU≥0.90, 强度误差≤10%, 重建相对L1≤5%)；relaxed=(0.85,15%,8%)。层数与启用集合匹配均要求完全正确。'])
    (directory/'failure_analysis.zh-CN.md').write_text('\n'.join(lines)+'\n')
    plot_report(directory/'failure_breakdown.png',report)


def plot_report(path,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    keys=('count','selection_given_correct_count','shape','strength','reconstruction','success')
    for offset,(view,protocol,label) in enumerate((('reported','strict','Original strict'),('deployed','strict','Active-set strict'),('deployed','relaxed','Active-set relaxed'))):
        values=[report['overall'][view][protocol]['rates'][k]['window_weighted'] for k in keys]
        axes[0,0].bar(np.arange(len(keys))+(offset-1)*.25,[np.nan if v is None else v for v in values],width=.25,label=label)
    axes[0,0].set_xticks(np.arange(len(keys)),['Count','Selection\n(count correct)','All masks','All strengths','Response','Complete'])
    axes[0,0].set_ylim(0,1.05);axes[0,0].set_title('Individual pass rates');axes[0,0].legend(fontsize=8)
    summary=report['overall']['deployed']['strict'];patterns=list(summary['exact_failure_patterns'].items())[:10]
    short={'count':'C','slot_selection':'S','shape':'M','strength':'A','reconstruction':'R','pass':'pass'}
    axes[0,1].barh([ '+'.join(short[x] for x in k.split('+')) for k,n in patterns][::-1],
                   [n/report['windows'] for k,n in patterns][::-1])
    axes[0,1].set_title('Joint patterns: C=count, S=selection, M=mask, A=strength, R=response')
    axes[0,1].set_xlabel('Fraction of all windows')
    groups=list(report['by_R_group']['deployed'])
    for j,protocol in enumerate(PROTOCOLS):
        values=[report['by_R_group']['deployed'][g][protocol]['rates']['success']['window_weighted'] for g in groups]
        axes[1,0].bar(np.arange(len(groups))+(j-.5)*.35,values,width=.35,label=protocol)
    axes[1,0].set_xticks(np.arange(len(groups)),groups);axes[1,0].set_ylim(0,1.05);axes[1,0].set_title('Active-set success by effective R');axes[1,0].legend()
    counter=report['counterfactuals'];names=('deployment_relative_l1','teacher_strength_relative_l1','teacher_mask_relative_l1','hard_mask_relative_l1')
    if counter['windows']:
        axes[1,1].bar(['Actual output','Teacher strength','Teacher mask','Hard mask'],[counter[k]['mean'] for k in names])
        axes[1,1].set_ylabel('Mean response relative L1')
    else:axes[1,1].text(.5,.5,'No count-correct windows',ha='center')
    axes[1,1].set_title(f"Evaluation-only substitutions | count-correct n={counter['windows']}")
    fig.savefig(path,dpi=150);plt.close(fig)


def portable_bundle(directory,root,metadata):
    directory,root=Path(directory),Path(root)
    allowed=('failure_analysis.json','failure_analysis.zh-CN.md','failure_breakdown.png',
             'diagnostic_windows.jsonl.gz','windows.csv.gz','layers.csv.gz','checkpoint_metadata.json')
    allowed+=tuple(name for name in ('validation_metrics.json','validation_metrics.zh-CN.md','decomposition_metrics.json','decomposition_metrics.zh-CN.md') if (directory/name).is_file())
    manifest={'schema':'rind-diagnostic-bundle-v1','metadata':metadata,'weights_included':False,'responses_included':False,
              'files_sha256':{name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in allowed}}
    (directory/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    target=root/'latest_diagnostics.tar.gz';temporary=target.with_suffix('.tmp')
    with tarfile.open(temporary,'w:gz') as archive:
        for name in (*allowed,'BUNDLE_MANIFEST.json'):
            archive.add(directory/name,arcname='decomposition_diagnostics/'+name,recursive=False)
    temporary.replace(target)
    from train_decomposition import write_json
    write_json(root/'latest.json',{'directory':str(directory),'bundle':str(target),'metadata':metadata})
    return target
