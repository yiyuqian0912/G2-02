"""Runtime GPU eligibility and no-update batch calibration for mask training.

Adaptive sharing is explicit policy: retained contexts are allowed if repeated
utilization samples are low and enough VRAM is free. No process is terminated.
"""
import argparse,json,math,os,subprocess,sys,time
from pathlib import Path

MIB=1024**2
GIB=1024**3


def read_gpus():
    data=subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    rows={}
    for line in data.splitlines():
        parts=[v.strip() for v in line.split(',')]
        if len(parts)!=6:raise ValueError('unexpected nvidia-smi GPU inventory: '+line)
        i,uuid,total,used,free,util=parts
        try:row={'id':str(int(i)),'uuid':uuid,'total_mib':int(total),'used_mib':int(used),
                 'free_mib':int(free),'utilization':int(util)}
        except ValueError:
            print('[GPU] Skipping unavailable telemetry: '+line,file=sys.stderr,flush=True);continue
        rows[row['id']]=row
    return rows


def select_gpus(*, count='auto', ids=None, policy='adaptive', min_free_mib=24576,
                max_used_mib=2048,max_utilization=10,samples=3,interval=.5):
    if policy not in ('adaptive','idle'):raise ValueError('GPU policy must be adaptive or idle')
    if count!='auto' and (not str(count).isdigit() or int(count)<1):raise ValueError('--gpus must be auto or positive integer')
    if any(type(x) is not int or x<0 for x in (min_free_mib,max_used_mib,max_utilization)) or max_utilization>100:
        raise ValueError('invalid GPU thresholds')
    if type(samples) is not int or samples<1 or not 0<=interval<=10:raise ValueError('invalid GPU sampling parameters')
    manual=None
    if ids is not None:
        manual=[i.strip() for i in ids.split(',')]
        if not manual or any(not x.isdigit() for x in manual) or len(set(manual))!=len(manual):
            raise ValueError('invalid or duplicate --gpu-ids')
        manual=[str(int(i)) for i in manual]
        if len(set(manual))!=len(manual):raise ValueError('duplicate --gpu-ids')
    snapshots=[]
    for n in range(samples):
        if n:time.sleep(interval)
        snapshots.append(read_gpus())
    occupied=set()
    if policy=='idle' and manual is None:
        text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                      '--format=csv,noheader,nounits'],text=True)
        occupied={line.split(',')[0].strip() for line in text.splitlines() if ',' in line}
    selected=[]
    for i,row in snapshots[-1].items():
        history=[s.get(i) for s in snapshots]
        reasons=[]
        if any(r is None or r['uuid']!=row['uuid'] for r in history):
            reasons.append('unstable GPU inventory');free=0;util=100
        else:
            free=min(r['free_mib'] for r in history);util=max(r['utilization'] for r in history)
            if free<min_free_mib:reasons.append(f'free VRAM < {min_free_mib} MiB')
            if manual is None:
                if util>max_utilization:reasons.append(f'recent utilization > {max_utilization}%')
                if policy=='idle' and (row['uuid'] in occupied or max(r['used_mib'] for r in history)>max_used_mib):
                    reasons.append('strict idle policy: process or occupied VRAM')
        include=(i in manual if manual is not None else not reasons)
        if include and reasons and manual is not None:
            raise RuntimeError(f'explicit GPU {i} has insufficient/unstable free VRAM: '+', '.join(reasons))
        if include:selected.append(i)
        decision='eligible' if include else ('not requested' if manual is not None else '; '.join(reasons))
        print(f'[GPU {i}] total={row["total_mib"]/1024:.1f} GiB used={row["used_mib"]/1024:.1f} GiB '
              f'min-free={free/1024:.1f} GiB max-recent-util={util}% | {decision}',file=sys.stderr,flush=True)
    if manual is not None:
        absent=set(manual)-set(selected)
        if absent:raise RuntimeError('requested GPU IDs not found: '+','.join(sorted(absent)))
        selected=manual
    elif count!='auto':selected=selected[:int(count)]
    if not selected:
        raise RuntimeError('No eligible GPU under the reported free-memory/utilization policy; '
                           'retained processes are allowed by --gpu-policy adaptive. No CPU fallback.')
    return selected


def batch_candidates(base,world,target,options=(8,16,24,32,48,64,96,128,192,256)):
    if any(type(x) is not int or x<1 for x in (base,world,target)):raise ValueError('invalid batch dimensions')
    effective=base*world*math.ceil(target/(base*world))
    choices=sorted(set([base,*options]))
    return [b for b in choices if type(b) is int and b>0 and
            b*world*math.ceil(target/(b*world))==effective]


def memory_budget(allocated,reserved,free,total,*,fraction=.9,reserve_bytes=6*GIB):
    if not 0<fraction<1 or reserve_bytes<0:raise ValueError('invalid calibration VRAM margin')
    # mem_get_info's free count excludes our existing allocations. Recover our
    # reserved bytes once, but never count another process's 44 GiB as available.
    accessible=min(total,free+reserved)
    return max(0,int(min(accessible*fraction,accessible-reserve_bytes)))


def choose_batch(measurements,world):
    feasible=[]
    for entry in measurements:
        ranks=entry['ranks']
        if len(ranks)!=world:raise ValueError('missing rank in batch calibration')
        if all(r.get('feasible',False) for r in ranks):
            slowest=max(r['step_seconds'] for r in ranks)
            if not math.isfinite(slowest) or slowest<=0:raise ValueError('invalid probe time')
            feasible.append((entry['batch']*world/slowest,entry['batch']))
    if not feasible:raise RuntimeError('No common batch fits selected GPUs; lower VRAM reserve or wait for more free VRAM.')
    # Do not fill VRAM for its own sake: choose measured throughput, not size.
    return max(feasible)[1]


def calibration_profiles(model,sequential):
    if not sequential:return [{'name':'native','horizon':None}]
    profiles=[{'name':'full_horizon','horizon':model.max_slots}]
    threshold=getattr(model,'adaptive_checkpoint_min_layers',0)
    if not getattr(model,'activation_checkpoint_decomposer',False) and 1<threshold<=model.max_slots:
        profiles.append({'name':'largest_uncheckpointed_horizon','horizon':threshold-1})
    return profiles


def _measure(model,batch,loss_config,device,amp,steps,*,horizon=None):
    import torch
    from decomposition_model import decomposition_training_loss
    durations=[];output=loss=None
    old_horizon=getattr(model,'calibration_horizon',None)
    had_horizon=hasattr(model,'calibration_horizon')
    old_full=getattr(model,'calibration_full_horizon',False)
    try:
        for step in range(steps+1):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device);started=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                # Probe the worst autonomous horizon, not just the cheaper
                # teacher-forced step. Restore the flag before real training.
                sequential=loss_config.get('objective')=='sequential_count_shape_v6'
                if sequential:
                    model.calibration_horizon=horizon if horizon is not None else model.max_slots
                try:
                    output=model(batch['response'],batch['obstacle'],batch['window'],None,None,compute_location=False,
                                 **({'verify_constraints':False} if sequential else {}))
                finally:
                    if sequential:
                        if had_horizon:model.calibration_horizon=old_horizon
                        elif hasattr(model,'calibration_horizon'):del model.calibration_horizon
                        model.calibration_full_horizon=old_full
            loss=decomposition_training_loss(output,batch,**loss_config)['loss']
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite calibration loss')
            loss.backward();torch.cuda.synchronize(device)
            if step:durations.append(time.perf_counter()-started)
            output=loss=None
        return sum(durations)/len(durations)
    finally:
        output=loss=None;model.zero_grad(set_to_none=True)


def calibrate_batch(model,records,config,device,rank,world,output,seed):
    """All ranks probe the SAME candidate sequence before DDP wrapping.

    Forward/loss/backward only; no optimizer step or teacher data creation.
    Always re-probe on resume, since other processes/free VRAM can change.
    """
    import torch
    import torch.distributed as dist
    if device.type!='cuda':raise ValueError('--auto-batch requires CUDA; CPU fallback is disabled')
    if config.get('task')!='decomposition_only':raise ValueError('runtime autobatch currently requires mask-only training')
    from decomposition_data import collate_decomposition_records
    from train_decomposition import write_json
    setting=config['training'];base=setting['batch_size']
    options=batch_candidates(base,world,setting['target_global_batch_size'])
    profiles=calibration_profiles(model,config['loss'].get('objective')=='sequential_count_shape_v6')
    # One largest-native training observation, copied/repeated on GPU. No bulk
    # CPU batch collation, no validation/test examples, no fit-weight updates.
    chosen=max(range(len(records)),key=lambda i:(records.entries[i]['size'],len(records.entries[i]['intensities'])))
    single=(records.device_batch([chosen]) if getattr(records,'device_batch',None) is not None else
            {k:v.to(device) for k,v in collate_decomposition_records([records[chosen]]).items()})
    training=model.training;cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state(device)
    model.train();measurements=[]
    reserve_gib=float(os.environ.get('RIND_GPU_RESERVE_GIB','6'))
    if not math.isfinite(reserve_gib) or reserve_gib<0:raise ValueError('GPU reserve must be finite and nonnegative')
    try:
        for micro in options:
            batch=None;profile_results=[];active_profile='batch_inputs'
            try:
                model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.synchronize(device)
                free,total=torch.cuda.mem_get_info(device)
                budget=memory_budget(torch.cuda.memory_allocated(device),torch.cuda.memory_reserved(device),free,total,
                                     reserve_bytes=int(reserve_gib*GIB))
                batch={k:v.expand(micro,*v.shape[1:]).contiguous() for k,v in single.items()}
                for profile in profiles:
                    active_profile=profile['name']
                    print(f'[GPU batch probe] rank={rank} micro={micro} '
                          f'profile={active_profile} horizon={profile["horizon"]}',flush=True)
                    model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    seconds=_measure(model,batch,config['loss'],device,setting['amp_dtype']=='bfloat16',steps=2,
                                     horizon=profile['horizon'])
                    profile_results.append({**profile,'step_seconds':seconds,
                        'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)})
                seconds=max(r['step_seconds'] for r in profile_results)
                peak=max(r['peak_reserved_bytes'] for r in profile_results)
                # Fresh Adam moments and DDP gradient buckets may not exist yet.
                extra=12*sum(p.numel() for p in model.parameters() if p.requires_grad)
                estimated=peak+extra
                local={'rank':rank,'gpu_name':torch.cuda.get_device_name(device),'feasible':estimated<=budget,'step_seconds':seconds,
                       'peak_reserved_bytes':peak,'estimated_peak_with_adam_ddp_bytes':estimated,
                       'budget_bytes':budget,'free_before_bytes':free,'total_bytes':total,
                       'profiles':profile_results}
            except torch.cuda.OutOfMemoryError:
                local={'rank':rank,'feasible':False,'out_of_memory':True,
                       'failed_profile':active_profile,'profiles_completed':profile_results}
            finally:
                batch=None;model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
            ranks=[local]
            if world>1:
                ranks=[None]*world;dist.all_gather_object(ranks,local)
            measurements.append({'batch':micro,'ranks':ranks})
            if rank==0:
                print(f'[GPU batch probe] micro={micro} | '+ ' | '.join(
                    f'rank{r["rank"]}: '+(f'{r["estimated_peak_with_adam_ddp_bytes"]/GIB:.1f} GiB / '
                     f'{r["budget_bytes"]/GIB:.1f} GiB budget; {r["step_seconds"]:.3f}s; fits={r["feasible"]}'
                     if 'step_seconds' in r else 'OOM '+r.get('failed_profile','')) for r in ranks),flush=True)
            # Always keep the common sequence: one rank OOM must never make
            # that rank leave while another enters a later collective.
        # Another retained worker can change its VRAM use during calibration.
        # Recheck the current budget before accepting an earlier measurement.
        free,total=torch.cuda.mem_get_info(device)
        latest={'rank':rank,'free_bytes':free,'budget_bytes':memory_budget(
            torch.cuda.memory_allocated(device),torch.cuda.memory_reserved(device),free,total,
            reserve_bytes=int(reserve_gib*GIB))}
        current_budgets=[latest]
        if world>1:
            current_budgets=[None]*world;dist.all_gather_object(current_budgets,latest)
        for entry in measurements:
            for r,current in zip(entry['ranks'],current_budgets):
                r['budget_at_selection_bytes']=current['budget_bytes']
                r['feasible']=r.get('feasible',False) and r['estimated_peak_with_adam_ddp_bytes']<=current['budget_bytes']
        selected=choose_batch(measurements,world)
        setting['batch_size']=selected
        report={'selected_micro_batch':selected,'baseline_micro_batch':base,'world_size':world,
                'physical_gpu_ids':os.environ.get('RIND_PHYSICAL_GPU_IDS'),
                'effective_global_batch':selected*world*math.ceil(setting['target_global_batch_size']/(selected*world)),
                'worst_training_size':int(single['response'].shape[-1]),'reserve_gib':reserve_gib,
                'calibration_profiles':profiles,
                'budgets_at_selection':current_budgets,'seed':seed,'amp_dtype':setting['amp_dtype'],'candidate_batches':options,'measurements':measurements,'weights_updated':False,
                'scope':'actual GPU forward/loss/backward; repeat one training observation; no disk/DDP timing'}
        if rank==0:
            output=Path(output);write_json(output/'gpu_batch_probe.json',report)
            write_json(output/'runtime/gpu_probes'/f'{time.time_ns()}-seed{seed}.json',report)
            print(f'[GPU batch probe] selected per-GPU batch={selected}; effective global batch={report["effective_global_batch"]}',flush=True)
        return selected
    finally:
        model.zero_grad(set_to_none=True);model.train(training)
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng,device);torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',default='auto');p.add_argument('--gpu-ids')
    p.add_argument('--policy',choices=('adaptive','idle'),default='adaptive')
    p.add_argument('--min-free-memory-mib',type=int,default=24576);p.add_argument('--max-used-memory-mib',type=int,default=2048)
    p.add_argument('--max-utilization',type=int,default=10);p.add_argument('--samples',type=int,default=3)
    a=p.parse_args();print(','.join(select_gpus(count=a.gpus,ids=a.gpu_ids,policy=a.policy,min_free_mib=a.min_free_memory_mib,
        max_used_mib=a.max_used_memory_mib,max_utilization=a.max_utilization,samples=a.samples)))


if __name__=='__main__':main()
