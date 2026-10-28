"""Training on canonical records, with native-size batches and scene-disjoint splits."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
import torch
from .interfaces import INTERFACE_VERSION, training_record
from .model import SourceEnergyField, student_distribution


DEFAULT_SHARPENED_OBJECTIVE = {
    'entropy_match_weight': 0.05,
    'ranking_loss_weight': 0.10,
    'ranking_positive_mass': 0.90,
    'ranking_negative_fraction': 0.50,
    'ranking_margin': 0.50,
    'ranking_softmin_temperature': 1.0,
}


def validate_student_objective(objective):
    """Validate an optional CE + entropy matching + ranking objective."""
    if objective is None:
        return None
    if not isinstance(objective,dict) or set(objective)!=set(DEFAULT_SHARPENED_OBJECTIVE):
        raise ValueError('student objective fields do not match the declared sharpened objective')
    result={key:float(objective[key]) for key in DEFAULT_SHARPENED_OBJECTIVE}
    if any(not np.isfinite(value) for value in result.values()):
        raise ValueError('student objective values must be finite')
    if result['entropy_match_weight']<0 or result['ranking_loss_weight']<0:
        raise ValueError('student objective weights must be nonnegative')
    for key in ('ranking_positive_mass','ranking_negative_fraction'):
        if not 0<result[key]<1:raise ValueError(f'{key} must lie strictly between zero and one')
    if result['ranking_margin']<0 or result['ranking_softmin_temperature']<=0:
        raise ValueError('ranking margin must be nonnegative and softmin temperature positive')
    return result


def sharpened_objective_loss(energy,log_probability,probability,teacher_probability,
                             support,objective):
    """Return per-record CE and a calibrated concentration/ranking objective.

    The positive set is the teacher's tie-inclusive highest-density region.
    The negative set is the bottom teacher-probability fraction.  Ranking uses
    a smooth hard-negative energy so a single false hotspot is not hidden by
    averaging thousands of easy negatives.
    """
    objective=validate_student_objective(objective)
    if objective is None:raise ValueError('sharpened objective settings are required')
    if (energy.shape!=log_probability.shape or energy.shape!=probability.shape or
            energy.shape!=teacher_probability.shape or energy.shape!=support.shape or
            energy.ndim!=2 or support.dtype!=torch.bool):
        raise ValueError('objective tensors must be aligned [B,N] with boolean support')
    masked_log=log_probability.masked_fill(~support,0)
    cross_entropy=-(teacher_probability.detach()*masked_log).sum(-1)
    q=teacher_probability.detach()
    q_log=torch.log(q.clamp_min(torch.finfo(q.dtype).tiny))
    teacher_entropy=-(q*q_log).sum(-1)
    student_entropy=-(probability*masked_log).sum(-1)
    entropy_loss=(student_entropy-teacher_entropy).square()

    descending=torch.where(support,q,torch.full_like(q,-torch.inf)).sort(dim=-1,descending=True).values
    cumulative=descending.clamp_min(0).cumsum(-1)
    boundary=(cumulative<objective['ranking_positive_mass']).sum(-1).clamp_max(q.shape[1]-1)
    positive_cutoff=descending.gather(1,boundary[:,None]).squeeze(1)
    positive=support & (q>=positive_cutoff[:,None])
    positive_weight=q*positive
    positive_energy=(positive_weight*energy).sum(-1)/positive_weight.sum(-1).clamp_min(
        torch.finfo(energy.dtype).tiny)

    ascending=torch.where(support,q,torch.full_like(q,torch.inf)).sort(dim=-1).values
    support_count=support.sum(-1)
    negative_count=torch.ceil(support_count*objective['ranking_negative_fraction']).long().clamp_min(1)
    negative_cutoff=ascending.gather(1,(negative_count-1)[:,None]).squeeze(1)
    negative=support & (q<=negative_cutoff[:,None]) & ~positive
    # Degenerate teachers can make the tie-inclusive sets overlap everywhere.
    # Fall back to all non-positive supported candidates; every ordinary Gibbs
    # teacher has at least one such candidate.
    fallback=support & ~positive
    no_negative=~negative.any(-1)
    negative=torch.where(no_negative[:,None],fallback,negative)
    still_empty=~negative.any(-1)
    if bool(still_empty.any()):
        # A perfectly uniform teacher makes its tie-inclusive positive HDR the
        # full support. Ranking is undefined for that record, so install one
        # numerically safe dummy and zero its ranking term below.
        dummy=support.to(torch.int64).argmax(-1)
        negative=negative.clone()
        negative[still_empty,dummy[still_empty]]=True
    temperature=objective['ranking_softmin_temperature']
    scaled=torch.where(negative,-energy/temperature,torch.full_like(energy,-torch.inf))
    log_count=negative.sum(-1).clamp_min(1).to(energy.dtype).log()
    negative_softmin=-temperature*(torch.logsumexp(scaled,dim=-1)-log_count)
    ranking_loss=torch.nn.functional.softplus(
        objective['ranking_margin']+positive_energy-negative_softmin)
    ranking_loss=torch.where(still_empty,torch.zeros_like(ranking_loss),ranking_loss)
    total=(cross_entropy+
           objective['entropy_match_weight']*entropy_loss+
           objective['ranking_loss_weight']*ranking_loss)
    return total,cross_entropy,entropy_loss,ranking_loss


def collate_records(records):
    if not records or len({np.asarray(r['response']).shape for r in records}) != 1:
        raise ValueError('each batch must have one native window size')
    batch = {k: torch.as_tensor(np.stack([r[k] for r in records]), dtype=dtype)
             for k, dtype in [('response', torch.float32), ('obstacle', torch.bool), ('window', torch.float32)]}
    b, n = len(records), max(len(r['candidate_xy']) for r in records)
    batch.update(candidate_xy=torch.zeros(b,n,2), valid=torch.zeros(b,n,dtype=torch.bool), support=torch.zeros(b,n,dtype=torch.bool), teacher_prob=torch.zeros(b,n))
    for i, r in enumerate(records):
        m = len(r['candidate_xy'])
        for key in ('candidate_xy','valid','teacher_prob'):
            value=np.asarray(r[key])
            # Uniform-grid coordinates reconstructed from the shared shard
            # cache are intentionally read-only.  Make the host copy explicit
            # before converting to Torch so PyTorch never receives a
            # non-writable NumPy view (the destination batch is writable).
            if not value.flags.writeable:value=value.copy()
            batch[key][i,:m] = torch.as_tensor(value)
        batch['support'][i,:m] = torch.as_tensor(r.get('support',r['valid']))
    geometry_keys=('geometry_types','geometry_params','geometry_mask')
    present=[all(key in record for key in geometry_keys) for record in records]
    if any(present) and not all(present):
        raise ValueError('scene geometry must be present for every record in a batch')
    if all(present):
        batch.update(
            geometry_types=torch.as_tensor(
                np.stack([r['geometry_types'] for r in records]),dtype=torch.long),
            geometry_params=torch.as_tensor(
                np.stack([r['geometry_params'] for r in records]),dtype=torch.float32),
            geometry_mask=torch.as_tensor(
                np.stack([r['geometry_mask'] for r in records]),dtype=torch.bool))
    return batch


def batches(records, batch_size, rng=None, prefetch=0, rank=0, world_size=1):
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if type(prefetch) is not int or prefetch<0:
        raise ValueError('prefetch must be a nonnegative integer')
    if (type(rank) is not int or type(world_size) is not int or world_size<1 or
            rank<0 or rank>=world_size):
        raise ValueError('rank must be in [0, world_size)')
    buckets = {}
    sizes=records.window_sizes if hasattr(records,'window_sizes') else [int(r['window'][2]) for r in records]
    counts=(records.candidate_counts if hasattr(records,'candidate_counts') else
            [len(r['candidate_xy']) for r in records])
    if len(sizes)!=len(counts):raise ValueError('record size/candidate metadata mismatch')
    for i,(size,count) in enumerate(zip(sizes,counts)):
        buckets.setdefault((int(size),int(count)), []).append(i)
    groups = []
    for bucket in buckets.values():
        if rng is not None:
            rng.shuffle(bucket)
        groups.extend(bucket[i:i+batch_size] for i in range(0,len(bucket),batch_size))
    if rng is not None:
        rng.shuffle(groups)
    if world_size>1:
        # Every DDP rank must execute the same number of optimizer steps with
        # the same local batch size.  Pad only the tail of each homogeneous
        # global batch (at most world_size-1 repeated records), then stride it
        # across ranks.  Candidate count and native window size therefore stay
        # identical within every local batch.
        distributed=[]
        for group in groups:
            remainder=len(group)%world_size
            if remainder:
                group=group+[group[i%len(group)] for i in range(world_size-remainder)]
            distributed.append(group[rank::world_size])
        groups=distributed
    def load(group):
        return collate_records([records[i] for i in group])
    if not prefetch:
        for group in groups:yield load(group)
        return
    # NumPy NPZ decompression and file reads release the GIL.  A small ordered
    # future queue overlaps them with the current CUDA batch without changing
    # sample order or loading the complete dataset into RAM.
    with ThreadPoolExecutor(max_workers=prefetch,thread_name_prefix='rind-batch') as pool:
        pending=deque()
        iterator=iter(groups)
        for _ in range(min(prefetch,len(groups))):
            pending.append(pool.submit(load,next(iterator)))
        while pending:
            future=pending.popleft()
            try:group=next(iterator)
            except StopIteration:group=None
            if group is not None:pending.append(pool.submit(load,group))
            yield future.result()


def run_epoch(model, records, batch_size=4, optimizer=None, rng=None, grad_clip=None,
              candidate_chunk=None, progress_callback=None, amp_dtype=None,
              prefetch_batches=0, distributed_rank=0, distributed_world_size=1,
              accumulation_steps=1,student_objective=None):
    model.train(optimizer is not None)
    total, count = 0., 0
    device = next(model.parameters()).device
    if amp_dtype not in (None,'none','bfloat16','float16'):
        raise ValueError('amp_dtype must be none, bfloat16 or float16')
    amp_enabled=device.type=='cuda' and amp_dtype not in (None,'none')
    amp_active=amp_enabled
    amp_torch_dtype=(torch.bfloat16 if amp_dtype=='bfloat16' else torch.float16)
    if type(accumulation_steps) is not int or accumulation_steps<1:
        raise ValueError('accumulation_steps must be positive')
    student_objective=validate_student_objective(student_objective)
    if optimizer is None and student_objective is not None:
        raise ValueError('student objective applies only to training; validation remains canonical CE')
    total_batches=batch_count(records,batch_size)
    with torch.set_grad_enabled(optimizer is not None):
        for batch_index,batch in enumerate(batches(
                records,batch_size,rng,prefetch_batches,
                distributed_rank,distributed_world_size),start=1):
            if device.type=='cuda':
                batch={k:v.pin_memory().to(device,non_blocking=True) for k,v in batch.items()}
            else:
                batch={k:v.to(device) for k,v in batch.items()}
            diagnose=(batch_index in (1,accumulation_steps) and device.type=='cuda')
            def milestone(name):
                if diagnose:
                    torch.cuda.synchronize(device)
                    print(f'[rank {distributed_rank}] first batch: {name}',flush=True)
            milestone('loaded on GPU')
            def forward(use_amp):
                with torch.autocast(device_type=device.type,dtype=amp_torch_dtype,enabled=use_amp):
                    geometry=tuple(batch.get(key) for key in
                                   ('geometry_types','geometry_params','geometry_mask'))
                    if candidate_chunk:
                        from torch.utils.checkpoint import checkpoint
                        encoded=model.encode_observation(
                            batch['response'],batch['obstacle'],batch['window'],*geometry)
                        parts=[]
                        for start in range(0,batch['candidate_xy'].shape[1],candidate_chunk):
                            xy=batch['candidate_xy'][:,start:start+candidate_chunk]
                            parts.append(checkpoint(model.score_encoded,encoded,xy,use_reentrant=False) if optimizer is not None else model.score_encoded(encoded,xy))
                        return torch.cat(parts,dim=1)
                    arguments={k:batch[k] for k in ('response','obstacle','window','candidate_xy')}
                    if geometry[0] is not None:
                        arguments.update(geometry_types=geometry[0],geometry_params=geometry[1],
                                         geometry_mask=geometry[2])
                    return model(**arguments)
            energy=forward(amp_active)
            milestone('forward complete')
            finite_energy=torch.isfinite(energy[batch['support']]).all()
            if not bool(finite_energy):
                if not amp_active:
                    raise ValueError('nonfinite valid energies in full-precision forward at batch {}'.format(batch_index))
                # A single unstable mixed-precision batch must not poison the
                # optimizer. Recompute it in FP32 and keep FP32 for the rest of
                # this epoch. The next epoch may try AMP again from finite weights.
                print('[stability] batch {} produced nonfinite {} energies; retrying in FP32 and disabling AMP for this epoch'.format(
                    batch_index,amp_dtype),flush=True)
                amp_active=False
                energy=forward(False)
                if not bool(torch.isfinite(energy[batch['support']]).all()):
                    raise ValueError('nonfinite valid energies remain after FP32 retry at batch {}'.format(batch_index))
            # Always normalize 16k-way candidate distributions in float32;
            # BF16 is used only for the expensive student network operations.
            log_p,probability = student_distribution(energy.float(),batch['support'])
            canonical_ce=-(batch['teacher_prob'].detach()*
                           log_p.masked_fill(~batch['support'],0)).sum(-1)
            if student_objective is None:
                loss=canonical_ce.mean()
            else:
                objective_loss,canonical_ce,_,_=sharpened_objective_loss(
                    energy.float(),log_p,probability,batch['teacher_prob'],
                    batch['support'],student_objective)
                loss=objective_loss.mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training/validation loss')
            if optimizer is not None:
                position=(batch_index-1)%accumulation_steps
                remaining=total_batches-(batch_index-position)+1
                window_steps=min(accumulation_steps,remaining)
                if position==0:optimizer.zero_grad(set_to_none=True)
                milestone('starting backward')
                (loss/window_steps).backward()
                milestone('backward complete')
                should_step=(position+1==window_steps)
                if should_step:
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                    optimizer.step()
                    milestone('optimizer step complete')
                    for name,parameter in model.named_parameters():
                        if not bool(torch.isfinite(parameter).all()):
                            raise ValueError('optimizer produced nonfinite parameter {} at batch {}'.format(
                                name,batch_index))
                else:
                    milestone('gradients accumulated')
            # Progress/history always report canonical CE so baseline and
            # sharpened experiments remain directly comparable.
            total += canonical_ce.detach().sum().item()
            count += len(energy)
            if progress_callback is not None:
                progress_callback(batch_index,len(energy),total/count)
    if not count:
        raise ValueError('empty record split')
    if distributed_world_size>1:
        import torch.distributed as dist
        totals=torch.tensor((total,float(count)),dtype=torch.float64,device=device)
        dist.all_reduce(totals,op=dist.ReduceOp.SUM)
        total,count=float(totals[0]),int(totals[1])
    return total/count


class TeacherRecords:
    """Lazy on-disk supervision: keep IDs/sizes in memory, decode only a batch."""
    def __init__(self, paths, dataset, normalization_support='geometry',cache_in_memory=False,
                 index_rows=None):
        if normalization_support not in ('world','geometry'):
            raise ValueError('unknown normalization support')
        self.paths,self.dataset,self.normalization_support=list(paths),dataset,normalization_support
        self.cache=[None]*len(self.paths) if cache_in_memory else None
        self.identities=[];self.window_sizes=[];self.candidate_counts=[];self.config_ids=set()
        if index_rows is not None:
            if len(index_rows)!=len(self.paths):raise ValueError('teacher index/path length mismatch')
            for row in index_rows:
                self.identities.append((int(row['scene_id']),int(row['view_id'])))
                self.window_sizes.append(int(row['size']))
                self.config_ids.add(str(row['config_id']))
                count=row.get('candidate_count')
                if count is None:
                    with np.load(self.paths[len(self.candidate_counts)],allow_pickle=False) as r:
                        count=len(r['candidate_xy'])
                self.candidate_counts.append(int(count))
        else:
            for path in self.paths:
                with np.load(path,allow_pickle=False) as r:
                    self.identities.append((int(r['scene_id']),int(r['view_id'])))
                    self.window_sizes.append(int(r['window'][2]))
                    self.candidate_counts.append(len(r['candidate_xy']))
                    self.config_ids.add(str(r['config_id'].item()))
        if len(set(self.identities)) != len(self.identities):
            raise ValueError('duplicate observation in split')
        self.scene_ids={s for s,v in self.identities}

    def __len__(self): return len(self.paths)

    def __getitem__(self,index):
        if self.cache is not None and self.cache[index] is not None:
            return self.cache[index]
        from .teacher import load_teacher_record
        record=load_teacher_record(self.paths[index])
        sample=training_record(record,self.dataset.get_observation(*self.identities[index]))
        sample['support']=np.ones_like(sample['valid']) if self.normalization_support=='world' else sample['valid'].copy()
        if self.cache is not None:self.cache[index]=sample
        return sample


class CompositeRecords:
    """Concatenate compatible training sources without mixing their batch grids."""
    def __init__(self, parts):
        self.parts=[part for part in parts if len(part)]
        if not self.parts:raise ValueError('at least one nonempty record source is required')
        self.offsets=np.cumsum([0]+[len(part) for part in self.parts])
        self.window_sizes=[size for part in self.parts for size in part.window_sizes]
        self.candidate_counts=[count for part in self.parts for count in part.candidate_counts]
        self.scene_ids=set().union(*(part.scene_ids for part in self.parts))
        self.config_ids=set().union(*(part.config_ids for part in self.parts))

    def __len__(self):return int(self.offsets[-1])

    def __getitem__(self,index):
        if index<0:index+=len(self)
        if index<0 or index>=len(self):raise IndexError(index)
        part=int(np.searchsorted(self.offsets,index,side='right')-1)
        return self.parts[part][index-int(self.offsets[part])]


def batch_count(records,batch_size):
    """Exact number of size/grid-homogeneous batches without decoding records."""
    if batch_size<=0:raise ValueError('batch_size must be positive')
    buckets={}
    sizes=(records.window_sizes if hasattr(records,'window_sizes') else
           [int(record['window'][2]) for record in records])
    counts=(records.candidate_counts if hasattr(records,'candidate_counts') else
            [len(record['candidate_xy']) for record in records])
    for key in zip(sizes,counts):
        buckets[key]=buckets.get(key,0)+1
    return sum(int(np.ceil(count/batch_size)) for count in buckets.values())


def load_splits(manifest, dataset, normalization_support='geometry',cache_in_memory=False):
    """Validate scene-disjoint teacher lists without retaining decoded observations."""
    path=Path(manifest)
    entries=json.loads(path.read_text())
    index_path=path.with_name('teachers.json')
    indexed={}
    if index_path.is_file():
        for row in json.loads(index_path.read_text()).get('records',[]):
            indexed.setdefault(row['split'],{})[row['path']]=row
    splits,seen={},set()
    for name in ('train','validation','test'):
        if not entries.get(name): raise ValueError(f'nonempty {name} teacher list required')
        use_index=(name in indexed and all(p in indexed[name] and 'config_id' in indexed[name][p]
                                           for p in entries[name]))
        index_rows=([indexed[name][p] for p in entries[name]] if use_index else None)
        records=TeacherRecords([path.parent/p for p in entries[name]],dataset,normalization_support,
                               cache_in_memory,index_rows)
        if seen & records.scene_ids: raise ValueError(f'scene leakage in {name}')
        seen |= records.scene_ids
        splits[name]=records
    return splits


def load_model(path):
    checkpoint = torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('interface_version') != INTERFACE_VERSION:
        raise ValueError('not a phase1-v1 checkpoint; use explicit legacy adapter')
    model = SourceEnergyField(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'])
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--seed',type=int,default=0)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error('epochs and batch-size must be positive')
    from .data import Phase1Dataset
    dataset = Phase1Dataset(args.data_root)
    splits = load_splits(args.records,dataset)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = dict(hidden=128,fourier_frequencies=6,coordinate_scale=float(dataset.manifest['global_size']))
    model = SourceEnergyField(**config)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
    args.output.mkdir(parents=True,exist_ok=True)
    history, best, steps = [], float('inf'), 0
    for epoch in range(args.epochs):
        train_ce = run_epoch(model,splits['train'],args.batch_size,optimizer,rng)
        steps += sum(1 for _ in batches(splits['train'],args.batch_size))
        val_ce = run_epoch(model,splits['validation'],args.batch_size)
        history.append(dict(epoch=epoch+1,train_cross_entropy=train_ce,validation_cross_entropy=val_ce))
        if val_ce < best:
            best = val_ce
            torch.save(dict(interface_version=INTERFACE_VERSION,model_config=config,model=model.state_dict(),
                epoch=epoch+1,global_step=steps,seed=args.seed,
                scene_splits={k:sorted({r['scene_id'] for r in v}) for k,v in splits.items()},
                teacher_configs=sorted({r['config_id'] for v in splits.values() for r in v}),
                records_manifest=str(args.records.resolve()),validation_cross_entropy=best),args.output/'best.pt')
    model, _ = load_model(args.output/'best.pt')
    report = dict(history=history,test_cross_entropy=run_epoch(model,splits['test'],args.batch_size))
    (args.output/'training.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
