"""Training on canonical records, with native-size batches and scene-disjoint splits."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .interfaces import INTERFACE_VERSION, training_record
from .model import SourceEnergyField, student_distribution


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
            batch[key][i,:m] = torch.as_tensor(r[key])
        batch['support'][i,:m] = torch.as_tensor(r.get('support',r['valid']))
    return batch


def batches(records, batch_size, rng=None):
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    buckets = {}
    sizes=records.window_sizes if hasattr(records,'window_sizes') else [int(r['window'][2]) for r in records]
    for i,size in enumerate(sizes):
        buckets.setdefault(int(size), []).append(i)
    groups = []
    for bucket in buckets.values():
        if rng is not None:
            rng.shuffle(bucket)
        groups.extend(bucket[i:i+batch_size] for i in range(0,len(bucket),batch_size))
    if rng is not None:
        rng.shuffle(groups)
    for group in groups:
        yield collate_records([records[i] for i in group])


def run_epoch(model, records, batch_size=4, optimizer=None, rng=None, grad_clip=None, candidate_chunk=None):
    model.train(optimizer is not None)
    total, count = 0., 0
    device = next(model.parameters()).device
    with torch.set_grad_enabled(optimizer is not None):
        for batch in batches(records,batch_size,rng):
            batch = {k:v.to(device) for k,v in batch.items()}
            if candidate_chunk:
                from torch.utils.checkpoint import checkpoint
                encoded=model.encode_observation(batch['response'],batch['obstacle'],batch['window'])
                parts=[]
                for start in range(0,batch['candidate_xy'].shape[1],candidate_chunk):
                    xy=batch['candidate_xy'][:,start:start+candidate_chunk]
                    parts.append(checkpoint(model.score_encoded,encoded,xy,use_reentrant=False) if optimizer is not None else model.score_encoded(encoded,xy))
                energy=torch.cat(parts,dim=1)
            else:
                energy = model(**{k:batch[k] for k in ('response','obstacle','window','candidate_xy')})
            log_p, _ = student_distribution(energy,batch['support'])
            loss = -(batch['teacher_prob'].detach() * log_p.masked_fill(~batch['support'],0)).sum(-1).mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training/validation loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
            total += loss.item()*len(energy)
            count += len(energy)
    if not count:
        raise ValueError('empty record split')
    return total/count


class TeacherRecords:
    """Lazy on-disk supervision: keep IDs/sizes in memory, decode only a batch."""
    def __init__(self, paths, dataset, normalization_support='geometry'):
        if normalization_support not in ('world','geometry'):
            raise ValueError('unknown normalization support')
        self.paths,self.dataset,self.normalization_support=list(paths),dataset,normalization_support
        self.identities=[];self.window_sizes=[];self.config_ids=set()
        for path in self.paths:
            with np.load(path,allow_pickle=False) as r:
                self.identities.append((int(r['scene_id']),int(r['view_id'])))
                self.window_sizes.append(int(r['window'][2]))
                self.config_ids.add(str(r['config_id'].item()))
        if len(set(self.identities)) != len(self.identities):
            raise ValueError('duplicate observation in split')
        self.scene_ids={s for s,v in self.identities}

    def __len__(self): return len(self.paths)

    def __getitem__(self,index):
        from .teacher import load_teacher_record
        record=load_teacher_record(self.paths[index])
        sample=training_record(record,self.dataset.get_observation(*self.identities[index]))
        sample['support']=np.ones_like(sample['valid']) if self.normalization_support=='world' else sample['valid'].copy()
        return sample


def load_splits(manifest, dataset, normalization_support='geometry'):
    """Validate scene-disjoint teacher lists without retaining decoded observations."""
    path=Path(manifest)
    entries=json.loads(path.read_text())
    splits,seen={},set()
    for name in ('train','validation','test'):
        if not entries.get(name): raise ValueError(f'nonempty {name} teacher list required')
        records=TeacherRecords([path.parent/p for p in entries[name]],dataset,normalization_support)
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
