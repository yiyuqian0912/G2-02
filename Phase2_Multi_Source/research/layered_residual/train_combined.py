"""Prepare equal-layer-mixture teachers and train the Phase-I energy field."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np

HERE=Path(__file__).resolve(); REPO=HERE.parents[3]
PHASE1=REPO/'Phase1_Single_Source_Single_Intensity'; PHASE2=REPO/'Phase2_Multi_Source'
for item in (PHASE1/'src',PHASE2/'src',PHASE2/'vendor/rind-dataset',HERE.parent):
    if str(item) not in sys.path: sys.path.insert(0,str(item))

from layered_residual import recover,save_run
from rind_phase2.data import Phase2Dataset


def _json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _atomic_checkpoint(path,value):
    import torch
    path=Path(path);temp=path.with_suffix('.tmp.pt')
    torch.save(value,temp);temp.replace(path)


def _progress(root,event,message,**fields):
    """Print one readable status line and append its structured form on disk."""
    stamp=time.strftime('%Y-%m-%d %H:%M:%S')
    row={'time':stamp,'event':event,**fields}
    print(f'[{stamp}] {message}',flush=True)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n')


def read_config(path):
    c=json.loads(Path(path).read_text()); required={'data_root','output_root','split_seed','scene_counts','record_counts','window_sizes','teacher','training'}
    if not required <= set(c): raise ValueError(f'missing config fields: {sorted(required-set(c))}')
    if set(c['scene_counts'])!={'train','validation','test'} or any(type(x) is not int or x<1 for x in c['scene_counts'].values()):
        raise ValueError('scene_counts requires positive train/validation/test counts')
    if (set(c['record_counts'])!={'train','validation','test'} or
            any(not ((type(x) is int and x>=1) or x=='all') for x in c['record_counts'].values())):
        raise ValueError('record_counts requires positive integers or "all" for train/validation/test')
    if not c['window_sizes'] or len(set(c['window_sizes']))!=len(c['window_sizes']) or any(x not in (16,32,64,128,256,512) for x in c['window_sizes']):
        raise ValueError('invalid window_sizes')
    if type(c['split_seed']) is not int or c['split_seed']<0: raise ValueError('split_seed must be nonnegative')
    t=c['training']
    if c['teacher'].get('combined_rule') != 'equal_layer_mixture':
        raise ValueError('teacher.combined_rule must be equal_layer_mixture')
    if t.get('normalization_support','geometry') not in ('geometry','world'):
        raise ValueError('normalization_support must be geometry or world')
    for key in ('epochs','patience','batch_size','hidden','num_threads','candidate_chunk'):
        if type(t[key]) is not int or t[key]<1: raise ValueError(f'{key} must be positive integer')
    if type(t.get('progress_every_batches',10)) is not int or t.get('progress_every_batches',10)<1:
        raise ValueError('progress_every_batches must be positive integer')
    if type(t['fourier_frequencies']) is not int or t['fourier_frequencies']<0:
        raise ValueError('fourier_frequencies must be nonnegative integer')
    for key in ('learning_rate','grad_clip'):
        if not np.isfinite(t[key]) or t[key]<=0: raise ValueError(f'{key} must be positive')
    if not np.isfinite(t['weight_decay']) or t['weight_decay']<0: raise ValueError('weight_decay must be nonnegative')
    if t.get('device') not in ('cpu','cuda'): raise ValueError('device must be cpu or cuda')
    if any(type(t.get(key)) is not bool for key in ('use_obstacle','pixel_coordinates')):
        raise ValueError('use_obstacle and pixel_coordinates must be explicit booleans')
    if not t['seeds'] or len(set(t['seeds']))!=len(t['seeds']) or any(type(x) is not int or x<0 for x in t['seeds']):
        raise ValueError('training seeds must be distinct nonnegative integers')
    teacher=c['teacher']
    if type(teacher.get('shadow_screening')) is not bool:
        raise ValueError('teacher.shadow_screening must be an explicit boolean')
    if type(teacher.get('save_plots',True)) is not bool:
        raise ValueError('teacher.save_plots must be an explicit boolean when provided')
    if type(teacher.get('max_layers',32)) is not int or teacher.get('max_layers',32)<1:
        raise ValueError('teacher.max_layers must be a positive integer')
    for key in ('spacing','temperature','boundary_sigma'):
        if not np.isfinite(teacher[key]) or teacher[key]<=0: raise ValueError(f'{key} must be positive')
    for key in ('alpha','beta','boundary_lambda'):
        if not np.isfinite(teacher[key]) or teacher[key]<0: raise ValueError(f'{key} must be nonnegative')
    return c


def context(c):
    data=Path(c['data_root']); output=Path(c['output_root'])
    if not data.is_absolute(): data=PHASE2/data
    if not output.is_absolute(): output=PHASE2/output
    dataset=Phase2Dataset(data)
    world=int(dataset.manifest['global_size']);spacing=c['teacher']['spacing']
    if not float(spacing).is_integer() or world%int(spacing):
        raise ValueError('teacher spacing must be an integer divisor of world size')
    from rind_phase1.sampling import require_resolved_sampling
    require_resolved_sampling(world,spacing,c['training']['fourier_frequencies'])
    return dataset,output


def _window_class(dataset,scene,view,tolerance):
    o=dataset.get_observation(scene,view);x,y,L=map(int,o['window'])
    response=np.asarray(dataset.get_region(scene,x,y,L)['response'],dtype=np.float64)
    free=response[~o['obstacle']]
    if not len(free): return 'no_free_pixels'
    if np.all(free>tolerance): return 'all_lit'
    if not np.any(free>tolerance): return 'all_dark'
    return 'eligible'


def _protocol(c,dataset,output):
    total=sum(c['scene_counts'].values())
    if total>dataset.num_scenes: raise ValueError('scene counts exceed dataset')
    rng=np.random.default_rng(c['split_seed']);order=rng.permutation(dataset.num_scenes)
    splits={};offset=0
    for name in ('train','validation','test'):
        count=c['scene_counts'][name];splits[name]=order[offset:offset+count].astype(int).tolist();offset+=count
    source_files=(HERE.parent/'layered_residual.py',HERE.parent/'train_combined.py',
                  PHASE1/'src/rind_phase1/model.py',PHASE1/'src/rind_phase1/train.py',
                  PHASE2/'src/rind_phase2/regions.py',PHASE2/'src/rind_phase2/physics.py')
    hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    protocol={'config':c,'config_sha256':_digest(c),'scene_splits':splits,
              'selection':'scene split first; deterministically shuffle real windows within each requested size and scan sizes round-robin; integer record_count stops at that many completed teachers, while "all" exhausts every real window in the split; filtered and failed windows do not count',
              'teacher_target':'one equal-weight mixture of all recovered layer distributions',
              'teacher_execution':'decompose and peel every residual layer first; after complete reconstruction, traverse one shared uniform full-field candidate grid once and compare each rendered visibility mask with all layers',
              'physical_sampling_count':'one uniform candidate visibility pass per completed observation; layer count changes score comparisons but not physical render passes',
              'student_input':'original response, obstacle option, window and candidate coordinate; no residual/layer inputs',
              'source_sha256':hashes}
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('output_root contains another protocol; choose a new output_root')
    _json(path,protocol);return protocol


def prepare(c,dataset,output):
    started=time.perf_counter();protocol=_protocol(c,dataset,output);rng=np.random.default_rng(c['split_seed'])
    teacher=c['teacher'];tol=float(teacher.get('tolerance',1e-12))
    records={name:[] for name in ('train','validation','test')};audit=[]
    scanned=teacher_attempts=0
    desired=dict(c['record_counts'])
    _progress(output,'prepare_start',
        f'prepare start | required completed records train={desired["train"]}, validation={desired["validation"]}, test={desired["test"]} | filtered or failed windows do not count | spacing={teacher["spacing"]}, temperature={teacher["temperature"]}',
        requested=desired,spacing=teacher['spacing'],temperature=teacher['temperature'])
    for split,scenes in protocol['scene_splits'].items():
        split_started=time.perf_counter()
        _progress(output,'prepare_split_start',f'prepare {split} | {len(scenes)} independent scenes, required completed records={desired[split]}',
                  split=split,scenes=len(scenes),requested=desired[split])
        # Build one deterministic pool per size, then interleave the pools. This
        # avoids selecting only the first easy size while still allowing more
        # than one real view from a scene when needed to reach the exact count.
        pools={int(size):[] for size in c['window_sizes']}
        for scene in scenes:
            views=dataset._base.local_views[scene,:int(dataset._base.view_counts[scene])]
            for size in pools:
                pools[size].extend((int(scene),int(view),size)
                                   for view in np.flatnonzero(views[:,2]==size))
        for pool in pools.values():rng.shuffle(pool)
        offsets={size:0 for size in pools}
        exhaustive=desired[split]=='all'
        while exhaustive or len(records[split])<desired[split]:
            progressed=False
            for size,pool in pools.items():
                if not exhaustive and len(records[split])>=desired[split]:break
                offset=offsets[size]
                if offset>=len(pool):continue
                progressed=True;scene,view,size=pool[offset];offsets[size]=offset+1
                scanned+=1;kind=_window_class(dataset,scene,view,tol)
                if kind!='eligible':
                    audit.append({'split':split,'scene_id':scene,'view_id':view,'size':size,'status':'skipped_'+kind})
                    if scanned%100==0:
                        counts={k:sum(x['status']=='skipped_'+k for x in audit)
                                for k in ('all_lit','all_dark','no_free_pixels')}
                        _progress(output,'prepare_scan',
                            f'prepare scan | checked={scanned}, eligible teacher windows processed={teacher_attempts}, selected={sum(map(len,records.values()))}, skips={counts}',
                            checked=scanned,teacher_attempts=teacher_attempts,selected=sum(map(len,records.values())),skips=counts)
                    continue
                directory=output/'teachers'/split/f'scene{scene}-view{view}'
                existing=directory/'report.json'
                if existing.exists() and (directory/'combined.npz').exists():
                    report=json.loads(existing.read_text())
                    _progress(output,'teacher_reused',
                        f'teacher reused | {split} scene={scene} view={view} size={size} status={report["status"]} layers={report["layers_recovered"]}',
                        split=split,scene_id=scene,view_id=view,size=size,status=report['status'],layers=report['layers_recovered'])
                else:
                    teacher_attempts+=1;teacher_started=time.perf_counter()
                    _progress(output,'teacher_start',
                        f'teacher window {teacher_attempts} | {split} scene={scene} view={view} size={size} | completed {len(records[split])}/{desired[split]}',
                        attempt=teacher_attempts,split=split,scene_id=scene,view_id=view,size=size,
                        selected=len(records[split]),requested=desired[split])
                    result=recover(dataset,scene,view,spacing=teacher['spacing'],temperature=teacher['temperature'],
                        tolerance=tol,alpha=teacher.get('alpha',1.),beta=teacher.get('beta',0.),
                        boundary_lambda=teacher.get('boundary_lambda',0.),boundary_sigma=teacher.get('boundary_sigma',1.),
                        max_layers=teacher.get('max_layers',32),backend=teacher.get('backend','auto'),
                        shadow_screening=teacher['shadow_screening'])
                    report=dict(result[0]);report['window']=dataset.get_observation(scene,view)['window'].tolist()
                    report['world_size']=int(dataset.manifest['global_size'])
                    save_run(directory,report,*result[1:],plots=teacher.get('save_plots',True))
                    _progress(output,'teacher_done',
                        f'teacher done | status={report["status"]}, layers={report["layers_recovered"]}/{report["layers_attempted"]}, MAE={report["final_mae"]:.6g}, {time.perf_counter()-teacher_started:.1f}s'
                        +(f', reason={report.get("failure_reason")}' if report['status']!='complete' else ''),
                        attempt=teacher_attempts,split=split,scene_id=scene,view_id=view,size=size,
                        status=report['status'],layers_recovered=report['layers_recovered'],
                        layers_attempted=report['layers_attempted'],final_mae=report['final_mae'],
                        failure_reason=report.get('failure_reason'),seconds=time.perf_counter()-teacher_started)
                row={'split':split,'scene_id':scene,'view_id':view,'size':size,'status':report['status'],
                     'failure_reason':report.get('failure_reason'),'layers_recovered':report['layers_recovered'],
                     'directory':str(directory.relative_to(output))}
                audit.append(row)
                if report['status']=='complete' and report['layers_recovered']>0:
                    records[split].append(row)
            if not progressed:break
        if not exhaustive and len(records[split])<desired[split]:
            audit.append({'split':split,'status':'record_quota_unfilled','selected':len(records[split]),
                          'requested':desired[split],'candidate_windows':sum(map(len,pools.values()))})
        _progress(output,'prepare_split_done',
            f'prepare {split} done | selected={len(records[split])}/{desired[split]}, elapsed={time.perf_counter()-split_started:.1f}s',
            split=split,selected=len(records[split]),requested=desired[split],seconds=time.perf_counter()-split_started)
    manifest={'target':'equal_layer_mixture','records':records,'audit':audit,
              'requested_counts':desired,
              'counts':{k:len(v) for k,v in records.items()},
              'skip_counts':{kind:sum(x['status']==kind for x in audit) for kind in
                             ('skipped_all_lit','skipped_all_dark','skipped_no_free_pixels')}}
    _json(output/'records.json',manifest)
    failures=sum(x.get('status')=='incomplete' for x in audit)
    _progress(output,'prepare_done',
        f'prepare done | records={manifest["counts"]}, skips={manifest["skip_counts"]}, incomplete teachers={failures}, elapsed={time.perf_counter()-started:.1f}s',
        counts=manifest['counts'],skip_counts=manifest['skip_counts'],incomplete_teachers=failures,
        checked_windows=scanned,teacher_attempts=teacher_attempts,seconds=time.perf_counter()-started)
    short={split:{'selected':len(records[split]),'requested':desired[split]}
           for split in records if desired[split]!='all' and len(records[split])!=desired[split]}
    if short:
        raise ValueError(f'completed record quota not reached: {short}; enlarge scene_counts or adjust window_sizes')
    return manifest


class CombinedRecords:
    def __init__(self,entries,dataset,root,normalization_support='geometry',response_tolerance=1e-12):
        self.entries=list(entries);self.dataset=dataset;self.root=Path(root)
        self.normalization_support=normalization_support
        self.response_tolerance=float(response_tolerance)
        self.window_sizes=[int(x['size']) for x in self.entries]
        self.scene_ids={int(x['scene_id']) for x in self.entries}
    def __len__(self): return len(self.entries)
    def __getitem__(self,index):
        row=self.entries[index];directory=self.root/row['directory']
        report=json.loads((directory/'report.json').read_text())
        if report['status']!='complete' or report.get('partial_combined_distribution'):
            raise ValueError('training requires a complete teacher book')
        observation=self.dataset.get_observation(row['scene_id'],row['view_id'])
        kind=_window_class(self.dataset,row['scene_id'],row['view_id'],self.response_tolerance)
        if kind!='eligible':
            raise ValueError(f'{kind.replace("_","-")} window leaked into training records')
        with np.load(directory/'combined.npz',allow_pickle=False) as z:
            probability=np.asarray(z['probability'],dtype=np.float64)
            candidate=np.asarray(z['candidate_xy'],dtype=np.float64)
            if 'valid' in z and len(z['valid']): valid=np.asarray(z['valid'],dtype=bool)
            else:
                with np.load(directory/'layer-01.npz',allow_pickle=False) as layer:
                    valid=np.asarray(layer['valid'],dtype=bool)
        if probability.shape!=(len(candidate),) or valid.shape!=probability.shape or not np.isclose(probability.sum(),1,atol=1e-12):
            raise ValueError('malformed combined teacher target')
        support=valid if self.normalization_support=='geometry' else np.ones_like(valid)
        return {'response':observation['response'],'obstacle':observation['obstacle'],'window':observation['window'],
                'candidate_xy':candidate,'valid':valid,'support':support,'teacher_prob':probability,
                'scene_id':row['scene_id'],'view_id':row['view_id']}


def train(c,dataset,output):
    import torch
    from rind_phase1.interfaces import INTERFACE_VERSION
    from rind_phase1.model import SourceEnergyField
    from rind_phase1.train import run_epoch,load_model
    manifest=json.loads((output/'records.json').read_text());t=c['training']
    actual={name:int(manifest.get('counts',{}).get(name,-1))
            for name in ('train','validation','test')}
    required=dict(c['record_counts'])
    mismatched={name:{'required':required[name],'found':actual[name]} for name in actual
                if ((required[name]=='all' and actual[name]<1) or
                    (required[name]!='all' and actual[name]!=required[name]))}
    if mismatched:
        raise ValueError(f'training record requirements are not satisfied: {mismatched}')
    torch.set_num_threads(t['num_threads'])
    support=t.get('normalization_support','geometry')
    response_tolerance=float(c['teacher'].get('tolerance',1e-12))
    splits={name:CombinedRecords(manifest['records'][name],dataset,output,support,response_tolerance)
            for name in ('train','validation','test')}
    if splits['train'].scene_ids & splits['validation'].scene_ids or splits['train'].scene_ids & splits['test'].scene_ids or splits['validation'].scene_ids & splits['test'].scene_ids:
        raise ValueError('scene leakage between splits')
    device=torch.device(t.get('device','cpu'))
    if device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    all_reports={}
    for seed in t['seeds']:
        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(seed)
        model_config={'hidden':t['hidden'],'fourier_frequencies':t['fourier_frequencies'],
                      'coordinate_scale':float(dataset.manifest['global_size']),
                      'use_obstacle':t.get('use_obstacle',True),'pixel_coordinates':t.get('pixel_coordinates',False),
                      'normalization_support':support}
        model=SourceEnergyField(**model_config).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'])
        directory=output/'models'/f'seed_{seed}';directory.mkdir(parents=True,exist_ok=True)
        protocol=json.loads((output/'protocol.json').read_text())
        signature=_digest({'protocol':_digest(protocol),'records':_digest(manifest),'seed':seed})
        history=[];best=float('inf');bad=0;start=0;steps=0;started=time.perf_counter()
        if (directory/'last.pt').exists():
            checkpoint=torch.load(directory/'last.pt',map_location='cpu',weights_only=True)
            if checkpoint.get('run_signature')!=signature:
                raise ValueError('resume checkpoint protocol/targets/seed mismatch')
            model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
            torch.set_rng_state(checkpoint['torch_rng']);rng.bit_generator.state=checkpoint['numpy_rng']
            history=checkpoint['history'];best=checkpoint['best_validation_ce'];bad=checkpoint['bad_epochs']
            start=checkpoint['epoch'];steps=checkpoint['global_step']
        epoch_steps=sum(int(np.ceil(sum(size==s for size in splits['train'].window_sizes)/t['batch_size']))
                        for s in sorted(set(splits['train'].window_sizes)))
        validation_steps=sum(int(np.ceil(sum(size==s for size in splits['validation'].window_sizes)/t['batch_size']))
                             for s in sorted(set(splits['validation'].window_sizes)))
        test_steps=sum(int(np.ceil(sum(size==s for size in splits['test'].window_sizes)/t['batch_size']))
                       for s in sorted(set(splits['test'].window_sizes)))
        parameters=sum(p.numel() for p in model.parameters())
        _progress(output,'train_start',
            f'train seed={seed} | records train/validation/test={len(splits["train"])}/{len(splits["validation"])}/{len(splits["test"])} | model parameters={parameters:,} | resume epoch={start}',
            seed=seed,train_records=len(splits['train']),validation_records=len(splits['validation']),
            test_records=len(splits['test']),model_parameters=parameters,resume_epoch=start,
            epochs=t['epochs'],batch_size=t['batch_size'],learning_rate=t['learning_rate'],weight_decay=t['weight_decay'])
        every=t.get('progress_every_batches',10)
        def batch_progress(phase,epoch,total_batches):
            def report(batch_index,batch_records,running_ce):
                if batch_index==1 or batch_index%every==0 or batch_index==total_batches:
                    _progress(output,'batch_progress',
                        f'{phase} | seed={seed}, epoch={epoch}/{t["epochs"]}, batch={batch_index}/{total_batches}, batch records={batch_records}, running CE={running_ce:.6f}',
                        phase=phase,seed=seed,epoch=epoch,total_epochs=t['epochs'],batch=batch_index,
                        total_batches=total_batches,batch_records=batch_records,running_cross_entropy=running_ce)
            return report
        for epoch in range(start+1,t['epochs']+1):
            if bad>=t['patience']:break
            tick=time.perf_counter()
            _progress(output,'epoch_start',f'epoch {epoch}/{t["epochs"]} start | seed={seed}',seed=seed,epoch=epoch,total_epochs=t['epochs'])
            train_ce=run_epoch(model,splits['train'],t['batch_size'],optimizer,rng,t['grad_clip'],t['candidate_chunk'],
                               batch_progress('train',epoch,epoch_steps))
            val_ce=run_epoch(model,splits['validation'],t['batch_size'],candidate_chunk=t['candidate_chunk'],
                             progress_callback=batch_progress('validation',epoch,validation_steps))
            steps+=epoch_steps;improved=val_ce<best
            best=min(best,val_ce);bad=0 if improved else bad+1
            history.append({'epoch':epoch,'train_cross_entropy':train_ce,'validation_cross_entropy':val_ce,
                            'seconds':time.perf_counter()-tick})
            common={'interface_version':INTERFACE_VERSION,'model_config':model_config,'model':model.state_dict(),
                    'epoch':epoch,'seed':seed,'records_manifest':str((output/'records.json').resolve()),
                    'target_semantics':'equal-weight mixture across complete recovered layer distributions',
                    'scene_splits':{k:sorted(v.scene_ids) for k,v in splits.items()},
                    'optimizer':optimizer.state_dict(),'history':history,'best_validation_ce':best,
                    'bad_epochs':bad,'global_step':steps,'run_signature':signature,
                    'torch_rng':torch.get_rng_state(),'numpy_rng':rng.bit_generator.state,
                    'software':{'torch':str(torch.__version__),'numpy':np.__version__,'device':str(device),
                                'num_threads':t['num_threads']}}
            if improved:_atomic_checkpoint(directory/'best.pt',common)
            _atomic_checkpoint(directory/'last.pt',common)
            _json(directory/'training.json',{'seed':seed,'history':history,'best_validation_cross_entropy':best,
                  'selection':'minimum validation cross entropy; test evaluated after training'})
            _progress(output,'epoch_done',
                f'epoch {epoch}/{t["epochs"]} done | train CE={train_ce:.6f}, validation CE={val_ce:.6f}, best={best:.6f}, no-improvement={bad}/{t["patience"]}, {history[-1]["seconds"]:.1f}s',
                seed=seed,epoch=epoch,total_epochs=t['epochs'],train_cross_entropy=train_ce,
                validation_cross_entropy=val_ce,best_validation_cross_entropy=best,bad_epochs=bad,
                patience=t['patience'],seconds=history[-1]['seconds'])
        if not (directory/'best.pt').exists(): raise ValueError('training produced no checkpoint')
        best_model,checkpoint=load_model(directory/'best.pt');best_model.to(device)
        _progress(output,'test_start',f'test start | seed={seed}, best epoch={checkpoint["epoch"]}',seed=seed,best_epoch=checkpoint['epoch'])
        test_ce=run_epoch(best_model,splits['test'],t['batch_size'],candidate_chunk=t['candidate_chunk'],
                          progress_callback=batch_progress('test',checkpoint['epoch'],test_steps))
        report={'seed':seed,'history':history,'best_epoch':checkpoint['epoch'],'best_validation_cross_entropy':checkpoint['best_validation_ce'],
                'test_cross_entropy':test_ce,'elapsed_seconds':time.perf_counter()-started,
                'records':manifest['counts'],'all_lit_windows_skipped':manifest['skip_counts']['skipped_all_lit']}
        _json(directory/'training.json',report);all_reports[str(seed)]=report
        _progress(output,'train_done',
            f'train done | seed={seed}, best epoch={checkpoint["epoch"]}, best validation CE={checkpoint["best_validation_ce"]:.6f}, test CE={test_ce:.6f}, elapsed={report["elapsed_seconds"]:.1f}s',
            seed=seed,best_epoch=checkpoint['epoch'],best_validation_cross_entropy=checkpoint['best_validation_ce'],
            test_cross_entropy=test_ce,seconds=report['elapsed_seconds'])
    _json(output/'training_summary.json',all_reports);return all_reports


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','train','all'])
    p.add_argument('--config',type=Path,required=True);a=p.parse_args(argv)
    c=read_config(a.config);dataset,output=context(c);output.mkdir(parents=True,exist_ok=True)
    _progress(output,'run_start',f'command={a.command} | config={a.config.resolve()} | output={output}',
              command=a.command,config=str(a.config.resolve()),output=str(output))
    try:
        if a.command in ('prepare','all'):
            m=prepare(c,dataset,output);print('Prepared:',m['counts'],'skips:',m['skip_counts'],flush=True)
        if a.command in ('train','all'):
            train(c,dataset,output);print('Ready:',output/'training_summary.json')
    except Exception as error:
        _progress(output,'run_failed',f'FAILED | {type(error).__name__}: {error}',
                  error_type=type(error).__name__,error=str(error))
        raise
    _progress(output,'run_done',f'command={a.command} completed | output={output}',command=a.command,output=str(output))

if __name__=='__main__':main()
