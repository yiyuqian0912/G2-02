"""Train/evaluate a truth-blind PPO window policy with a frozen Phase1 locator."""
import argparse,copy,hashlib,json,math,os,random,re,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from .core import TileEnv,Window,ActorCritic,advantages,potential,belief_metrics
from .locator import FrozenLocator,Belief,policy_inputs
from .frontier import make_env,make_policy,policy_names,boundary_actions
from .prepare import prepare,resolve,sha,write_json,geometry_hash
from rind_phase1.data import Phase1Dataset,PROJECT_ROOT


def log(message):print(time.strftime('[%Y-%m-%d %H:%M:%S] ')+message,flush=True)


def atomic_save(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp.pt');torch.save(value,tmp);tmp.replace(path)


def load_config(args,base=None):
    c=copy.deepcopy(base) if base is not None else json.loads(resolve(args.config).read_text())
    for item in args.set:
        key,sep,value=item.partition('=')
        if not sep or key not in c:raise ValueError('expected known KEY=JSON: '+item)
        c[key]=json.loads(value)
    if args.phase1_run:c['phase1_run']=args.phase1_run
    if args.data_root:c['data_root']=args.data_root
    if args.scene_counts:c['scene_counts']=dict(zip(('train','validation','test'),args.scene_counts))
    for name in ('updates','num_envs','rollout_steps'):
        if getattr(args,name) is not None:c[name]=getattr(args,name)
    for k in ('seed','tile_size','initial_size','spacing','budget','max_steps','num_envs','rollout_steps','updates',
              'ppo_epochs','minibatch_size','hidden','eval_every','locator_batch_size','candidate_chunk','tile_cache_size','threads'):
        if type(c[k]) is not int or c[k]<(0 if k=='seed' else 1):raise ValueError('invalid '+k)
    if c.get('environment','window') not in ('window','frontier'):raise ValueError('unknown acquisition environment')
    if c.get('fusion_method','product') not in ('product','minimum'):raise ValueError('unknown fusion method')
    if c['spacing']>16 or c['tile_size']%c['spacing']:raise ValueError('spacing<=16 and tile divisible by spacing required')
    if c.get('environment')=='frontier':
        if c['initial_size']!=c['tile_size']:raise ValueError('frontier starts with exactly one tile')
        budgets=c.get('train_budgets',[c['budget']])+c['evaluation_budgets']+[c['budget']]
        if any(type(b) is not int or b<c['tile_size']**2 or b%(c['tile_size']**2) for b in budgets):
            raise ValueError('frontier budgets must contain whole tiles')
        if c['max_steps']<max(budgets)//c['tile_size']**2-1:raise ValueError('max_steps would truncate the largest frontier budget')
        if not c.get('train_budgets',[c['budget']]):raise ValueError('empty training budget list')
    elif c['initial_size']<2*c['tile_size']:raise ValueError('window initial>=2*tile required')
    for key in ('evaluation_envs','evaluation_curve_stride','validation_probe_scenes'):
        if key in c and (type(c[key]) is not int or c[key]<1):raise ValueError('invalid '+key)
    if any(type(v) is not int or v<1 for v in c['scene_counts'].values()):raise ValueError('positive scene counts required')
    if c['budget']<=c['initial_size']**2:raise ValueError('training budget must exceed the initial observation')
    if any(type(b) is not int or b<c['initial_size']**2 for b in c['evaluation_budgets']):raise ValueError('invalid evaluation budgets')
    if type(c['bootstrap_repetitions']) is not int or c['bootstrap_repetitions']<1:raise ValueError('positive bootstrap repetitions required')
    if c['fusion_discount']<=0 or c['fusion_discount']>1:raise ValueError('fusion discount in (0,1]')
    if not 0<c['gae_lambda']<=1 or not 0<c['clip_ratio']<1:raise ValueError('invalid PPO settings')
    return c


def new_context(scene,window,budget,c,locator):
    e=make_env(c,locator.world,Window(*window),budget)
    return SimpleNamespace(scene=int(scene),env=e,belief=Belief(locator,c['fusion_discount'],c.get('fusion_method','product')))


def acquire(contexts,tile_lists,locator):
    requests=[(c.scene,i) for c,tiles in zip(contexts,tile_lists) for i in tiles]
    answers=locator.query(requests)
    for c,tiles in zip(contexts,tile_lists):c.belief.update(c.scene,tiles,answers,finite_model_fields=True)


def truth(dataset,scene):
    # Private scoring channel: no actor/critic input construction calls this.
    return np.array(dataset._base.drivers[scene,0],copy=True)


def reset_many(contexts,indices,rows,protocol,c,locator,rng):
    selected=[]
    for i in indices:
        scene=int(rng.choice(rows));w=protocol['initial_windows'][str(scene)]
        if c.get('randomize_train_start',False) and c.get('initial_pixel') is None:
            side=c['initial_size'];limit=(locator.world-side)//c['tile_size']+1
            w=[int(rng.integers(limit))*c['tile_size'],int(rng.integers(limit))*c['tile_size'],side]
        budget=int(rng.choice(c.get('train_budgets',[c['budget']])))
        contexts[i]=new_context(scene,w,budget,c,locator);selected.append(contexts[i])
    acquire(selected,[x.env.initial_tiles for x in selected],locator)


def collect(policy,contexts,protocol,c,locator,rng,training_rows=None):
    device=locator.device;data={k:[] for k in ('image','scalar','actions','legal','action','logp','value','reward','done')};episodes=[]
    for step in range(c['rollout_steps']):
        inputs=policy_inputs(contexts,locator)
        with torch.no_grad():distribution,value=policy(*inputs);action=distribution.sample();lp=distribution.log_prob(action)
        before=score_contexts(contexts,locator)
        tiles=[x.env.step(int(a)) for x,a in zip(contexts,action.cpu().tolist())]
        acquire(contexts,tiles,locator)
        after=score_contexts(contexts,locator)
        done=torch.tensor([x.env.done for x in contexts],device=device)
        for k,v in zip(('image','scalar','actions','legal'),inputs):data[k].append(v.detach())
        for k,v in (('action',action),('logp',lp),('value',value),('reward',after-before),('done',done)):data[k].append(v.detach())
        finished=done.nonzero().flatten().cpu().tolist()
        if finished:
            for i,m in zip(finished,metric_rows([contexts[i] for i in finished],locator)):
                x=contexts[i];episodes.append(dict(scene_id=x.scene,spent=x.env.spent,steps=x.env.steps,**m))
        if finished:reset_many(contexts,finished,training_rows or protocol['scene_splits']['train'],protocol,c,locator,rng)
    with torch.no_grad():_,bootstrap=policy(*policy_inputs(contexts,locator))
    data={k:torch.stack(v) for k,v in data.items()}
    adv,returns=advantages(data['reward'],data['value'],data['done'],bootstrap,lam=c['gae_lambda'])
    data['advantage']=adv;data['returns']=returns
    return {k:v.flatten(0,1) for k,v in data.items()},episodes


def score_contexts(contexts,locator):
    # Privileged reward channel. No policy-input code calls this function.
    source=torch.as_tensor(np.stack([truth(locator.dataset,x.scene) for x in contexts]),device=locator.device,dtype=locator.xy.dtype)
    distance=(locator.xy[None]-source[:,None]).square().sum(-1)
    p=torch.stack([x.belief.probability for x in contexts])
    mass=torch.stack([(p*(distance<=radius**2)).sum(-1) for radius in (16,32)],-1)
    return (mass+1e-8).log().mean(-1)


@torch.no_grad()
def metric_rows(contexts,locator):
    """Batch scoring on the device; one host transfer for all scalar metrics."""
    p=torch.stack([x.belief.probability for x in contexts])
    source=torch.as_tensor(np.stack([truth(locator.dataset,x.scene) for x in contexts]),device=locator.device,dtype=locator.xy.dtype)
    distance=(locator.xy[None]-source[:,None]).square().sum(-1).sqrt()
    m16=(p*(distance<=16)).sum(-1);m32=(p*(distance<=32)).sum(-1)
    ordered=p.sort(-1,descending=True).values
    index=(ordered.cumsum(-1)<.9).sum(-1).clamp_max(p.shape[1]-1)
    cut=ordered.gather(1,index[:,None]);core=p>=cut
    values=torch.stack(((torch.stack((m16,m32),-1)+1e-8).log().mean(-1),m16,m32,
        distance.gather(1,p.argmax(-1)[:,None])[:,0],(p*distance).sum(-1),
        -(p*p.clamp_min(1e-30).log()).sum(-1),core.sum(-1)*locator.spacing**2,
        core.gather(1,distance.argmin(-1)[:,None])[:,0].float()),-1).cpu().tolist()
    keys=('score','mass16','mass32','map_distance','expected_distance','entropy','core90_area','core90_covers_truth')
    rows=[dict(zip(keys,v)) for v in values]
    for row in rows:row['core90_covers_truth']=bool(row['core90_covers_truth'])
    return rows


def ppo_update(wrapped,buffer,optimizer,c,world):
    adv=buffer['advantage'];stats=torch.stack((adv.sum(),adv.square().sum(),adv.new_tensor(adv.numel())))
    if world>1:dist.all_reduce(stats)
    mean=stats[0]/stats[2];std=(stats[1]/stats[2]-mean.square()).clamp_min(1e-8).sqrt()
    buffer['advantage']=(adv-mean)/std
    totals=torch.zeros(5,device=adv.device);stopped=False
    for _ in range(c['ppo_epochs']):
        order=torch.randperm(len(adv),device=adv.device)
        for ids in order.split(c['minibatch_size']):
            distribution,value=wrapped(*(buffer[k][ids] for k in ('image','scalar','actions','legal')))
            logp=distribution.log_prob(buffer['action'][ids]);lr=logp-buffer['logp'][ids];ratio=lr.exp()
            a=buffer['advantage'][ids]
            policy_loss=-torch.minimum(ratio*a,ratio.clamp(1-c['clip_ratio'],1+c['clip_ratio'])*a).mean()
            value_loss=(value-buffer['returns'][ids]).square().mean()
            action_entropy=distribution.entropy().mean()
            loss=policy_loss+c['value_weight']*value_loss-c['action_entropy_bonus']*action_entropy
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite PPO loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(wrapped.parameters(),c['grad_clip'])
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite PPO gradient')
            optimizer.step()
            kl=((ratio-1)-lr).mean().detach()
            if world>1:dist.all_reduce(kl);kl/=world
            totals+=torch.stack((policy_loss.detach(),value_loss.detach(),action_entropy.detach(),kl,kl.new_tensor(1.)))
            # All ranks use the same reduced decision AFTER completing backward.
            if float(kl)>c['target_kl']:stopped=True;break
        if stopped:break
    if world>1:dist.all_reduce(totals)
    vals=(totals[:4]/totals[4]).cpu().tolist()
    return dict(zip(('policy_loss','value_loss','action_entropy','approx_kl'),vals))


def episode(policy,kind,scene,budget,protocol,c,locator,seed,trace=False):
    x=new_context(scene,protocol['initial_windows'][str(scene)],budget,c,locator)
    acquire([x],[x.env.initial_tiles],locator);source=truth(locator.dataset,scene)
    rows=[];fields=[];rng=np.random.default_rng(seed)
    def record():
        rows.append(dict(spent=x.env.spent,**belief_metrics(x.belief.probability,locator.xy,source,c['spacing'])))
        if trace:fields.append(x.belief.probability.detach().cpu().numpy())
    record();initial=rows[0]['score'];started=time.perf_counter()
    while not x.env.done:
        if kind in ('ppo','initial_policy'):
            with torch.no_grad():d,_=policy(*policy_inputs([x],locator));action=int(d.logits.argmax(1))
        elif kind=='boundary':action=boundary_actions([x],locator)[0]
        else:action=x.env.baseline(kind,rng)
        acquire([x],[x.env.step(action)],locator);record()
    row=dict(scene_id=scene,geometry_id=protocol.get('geometry_groups',{}).get(str(scene)) or geometry_hash(locator.dataset,scene),policy=kind,budget=budget,steps=x.env.steps,
        reason=x.env.reason,source_was_observed=bool(x.env.coverage[int(source[1])//c['tile_size'],int(source[0])//c['tile_size']]),initial_score=initial,return_value=rows[-1]['score']-initial,
        **rows[-1],curve=rows,seconds=time.perf_counter()-started)
    if trace:
        row['history']=x.env.history;row['seen_tiles']=sorted(x.belief.seen)
        row['_fields']=np.stack(fields)
    return row


def evaluate_rows(policy,protocol,c,locator,rank,world,*,split,baselines=False,initial=None):
    if c.get('environment')=='frontier':
        return evaluate_frontier_rows(policy,protocol,c,locator,rank,world,split=split,baselines=baselines,initial=initial)
    rows=[];policies=policy_names(c) if baselines else ('ppo',)
    budgets=c['evaluation_budgets'] if baselines else [c['budget']]
    ids=protocol['scene_splits'][split][rank::world]
    for j,scene in enumerate(ids):
        for budget in budgets:
            for kind in policies:
                seed=int(np.random.SeedSequence([c['seed'],scene,budget,19]).generate_state(1)[0])
                rows.append(episode(initial if kind=='initial_policy' else policy,kind,scene,budget,protocol,c,locator,seed))
        if j%16==0:log(f'RL {split} rank={rank} scenes={j+1}/{len(ids)}')
    if world>1:
        pieces=[None]*world;dist.all_gather_object(pieces,rows);rows=[r for piece in pieces for r in piece]
    return rows


def evaluate_frontier_rows(policy,protocol,c,locator,rank,world,*,split,baselines=False,initial=None,full_validation=False):
    """Parallel environments share GPU Phase1 calls; no per-scene serial query loop."""
    scene_ids=protocol['scene_splits'][split]
    if split=='validation' and not full_validation:scene_ids=scene_ids[:c.get('validation_probe_scenes',len(scene_ids))]
    kinds=policy_names(c) if baselines else ('ppo',)
    budgets=c['evaluation_budgets'] if baselines else [c['budget']]
    jobs=[(scene,budget,kind) for scene in scene_ids[rank::world] for budget in budgets for kind in kinds]
    rows=[];width=c.get('evaluation_envs',c['num_envs']);stride=c.get('evaluation_curve_stride',1)
    for start in range(0,len(jobs),width):
        batch=jobs[start:start+width]
        contexts=[new_context(s,protocol['initial_windows'][str(s)],b,c,locator) for s,b,k in batch]
        rngs=[np.random.default_rng(int(np.random.SeedSequence([c['seed'],s,b,19]).generate_state(1)[0])) for s,b,k in batch]
        acquire(contexts,[x.env.initial_tiles for x in contexts],locator)
        curves=[[dict(spent=x.env.spent,**m)] for x,m in zip(contexts,metric_rows(contexts,locator))]
        tick=time.perf_counter()
        while True:
            active=[i for i,x in enumerate(contexts) if not x.env.done]
            if not active:break
            actions={}
            for kind in kinds:
                ids=[i for i in active if batch[i][2]==kind]
                if not ids:continue
                selected=[contexts[i] for i in ids]
                if kind in ('ppo','initial_policy'):
                    with torch.no_grad():d,_=(initial if kind=='initial_policy' else policy)(*policy_inputs(selected,locator))
                    chosen=d.logits.argmax(1).cpu().tolist()
                elif kind=='boundary':chosen=boundary_actions(selected,locator)
                else:chosen=[x.env.baseline(kind,rngs[i]) for x,i in zip(selected,ids)]
                actions.update(zip(ids,chosen))
            acquire([contexts[i] for i in active],[contexts[i].env.step(actions[i]) for i in active],locator)
            record=[i for i in active if contexts[i].env.done or contexts[i].env.steps%stride==0]
            if record:
                for i,m in zip(record,metric_rows([contexts[i] for i in record],locator)):
                    curves[i].append(dict(spent=contexts[i].env.spent,**m))
        elapsed=time.perf_counter()-tick
        for (scene,budget,kind),x,curve in zip(batch,contexts,curves):
            source=truth(locator.dataset,scene)
            rows.append(dict(scene_id=scene,geometry_id=protocol['geometry_groups'][str(scene)],policy=kind,budget=budget,
                steps=x.env.steps,reason=x.env.reason,source_was_observed=bool(x.env.coverage[int(source[1])//c['tile_size'],int(source[0])//c['tile_size']]),
                initial_score=curve[0]['score'],return_value=curve[-1]['score']-curve[0]['score'],**curve[-1],curve=curve,
                acquired_tiles=[x.env.tiles(Window(*h['window']))[0] for h in x.env.history if h['window']],
                seconds=elapsed/len(batch),timing='parallel batch amortized; shared cache'))
        if start%(width*8)==0:log(f'RL {split} rank={rank} episodes={min(start+width,len(jobs))}/{len(jobs)}')
    if world>1:
        pieces=[None]*world;dist.all_gather_object(pieces,rows);rows=[r for piece in pieces for r in piece]
    return rows


def train(policy,protocol,c,locator,output,rank,world):
    wrapped=DistributedDataParallel(policy,device_ids=[locator.device.index] if locator.device.type=='cuda' else None) if world>1 else policy
    optimizer=torch.optim.Adam(policy.parameters(),lr=c['learning_rate'],eps=1e-5)
    start=0;best=-float('inf');history=[];last=output/'last.pt';fingerprint=sha(output/'protocol.json')
    if last.is_file():
        saved=torch.load(last,map_location='cpu',weights_only=True)
        if saved['protocol_sha256']!=fingerprint:raise ValueError('RL checkpoint/protocol mismatch')
        policy.load_state_dict(saved['policy']);optimizer.load_state_dict(saved['optimizer'])
        start=saved['update'];best=saved['best'] if saved['best'] is not None else -float('inf');history=saved['history']
        log(f'RL resume after update={start}; partial simulator episodes restart')
    elif rank==0:atomic_save(output/'initial.pt',dict(policy=policy.state_dict(),config=c,protocol_sha256=fingerprint))
    if world>1:dist.barrier()
    torch.manual_seed(c['seed']+rank+1009*start)
    rng=np.random.default_rng(np.random.SeedSequence([c['seed'],rank,start,7]))
    training_rows=protocol['scene_splits']['train'][rank::world]
    if not training_rows:raise ValueError('more GPU ranks than training scenes')
    contexts=[None]*c['num_envs'];reset_many(contexts,list(range(len(contexts))),training_rows,protocol,c,locator,rng)
    log(f'RL train | scenes={len(protocol["scene_splits"]["train"])} | ranks={world} | transitions/update={c["rollout_steps"]*c["num_envs"]*world:,} | configured total={c["updates"]*c["rollout_steps"]*c["num_envs"]*world:,} | budgets={c.get("train_budgets",[c["budget"]])}')
    for update in range(start+1,c['updates']+1):
        tick=time.perf_counter();policy.train()
        if locator.device.type=='cuda':torch.cuda.reset_peak_memory_stats(locator.device)
        buffer,episodes=collect(policy,contexts,protocol,c,locator,rng,training_rows)
        metrics=ppo_update(wrapped,buffer,optimizer,c,world)
        validation=None
        if update%c['eval_every']==0 or update==c['updates']:
            policy.eval();rows=evaluate_rows(policy,protocol,c,locator,rank,world,split='validation')
            validation=float(np.mean([r['score'] for r in rows]))
        improved=validation is not None and validation>best
        if improved:best=validation
        record=dict(update=update,seconds=time.perf_counter()-tick,validation_score=validation,
            mean_step_reward=float(buffer['reward'].mean()),completed_local_episodes=len(episodes),**metrics)
        record.update(phase1_queries=locator.queries,phase1_cache_hits=locator.hits,constant_tiles=locator.uniform,
            transitions=c['rollout_steps']*c['num_envs']*world,
            transitions_per_second=c['rollout_steps']*c['num_envs']*world/max(record['seconds'],1e-9),
            peak_allocated_gib=torch.cuda.max_memory_allocated(locator.device)/1024**3 if locator.device.type=='cuda' else 0.)
        history.append(record)
        if rank==0:
            state=dict(schema='rind-active-rl-policy-v1',policy=policy.state_dict(),optimizer=optimizer.state_dict(),
                locator_snapshot_sha256=protocol['snapshot_sha256'],update=update,best=best if math.isfinite(best) else None,history=history,config=c,protocol_sha256=fingerprint)
            atomic_save(last,state)
            if improved:atomic_save(output/'best.pt',state)
            write_json(output/'training.json',dict(history=history,best_validation_score=best if math.isfinite(best) else None,
                policy_parameters=sum(p.numel() for p in policy.parameters()),phase1_frozen=True,
                fusion_method=c.get('fusion_method','product'),environment=c.get('environment','window'),
                total_transitions=sum(h.get('transitions',c['rollout_steps']*c['num_envs']*world) for h in history),
                transitions_per_update=c['rollout_steps']*c['num_envs']*world,world_size=world))
            log(f'RL update={update}/{c["updates"]} reward={record["mean_step_reward"]:.4f} '
                f'val={validation} KL={metrics["approx_kl"]:.4f} seconds={record["seconds"]:.1f} '
                f'transitions/s={record["transitions_per_second"]:.1f} rank0 peak={record["peak_allocated_gib"]:.2f}GiB cache hits={locator.hits}')
        if world>1:dist.barrier()


def save_trace(row,protocol,c,locator,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fields=row.pop('_fields');scene=row['scene_id'];prefix=f'scene{scene}_b{row["budget"]}_{row["policy"]}'
    directory=output/'figures';directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/(prefix+'.npz'),probability=fields,candidate_xy=locator.xy.cpu().numpy())
    write_json(directory/(prefix+'.json'),row)
    source=truth(locator.dataset,scene);world=locator.world;n=world//c['tile_size']
    response=np.full((world,world),np.nan,dtype=np.float32)
    patches=locator.pixels(scene,row['seen_tiles']).to(torch.uint8).cpu().numpy()
    for tile,patch in zip(row['seen_tiles'],patches):
        x,y=(tile%n)*c['tile_size'],(tile//n)*c['tile_size']
        response[y:y+c['tile_size'],x:x+c['tile_size']]=patch
    if c.get('environment')=='frontier':
        from .trajectory_view import save_view
        answers=locator.query([(scene,tile) for tile in row['seen_tiles']])
        raw=[];flags=[]
        for h in row['history']:
            if h['window'] is None:
                raw.append(locator.prior.exp().cpu().numpy());flags.append(False)
            else:
                x,y,size=h['window'];key=(scene,y//size*n+x//size)
                answer=answers[key];raw.append(answer[0].exp().cpu().numpy());flags.append(bool(answer[2]))
        raw=np.stack(raw)
        np.savez_compressed(directory/(prefix+'.npz'),probability=fields,raw_probability=raw,
            candidate_xy=locator.xy.cpu().numpy(),informative=np.array(flags))
        save_view(directory/(prefix+'.html'),row,fields,raw,patches,truth(locator.dataset,scene),
            locator.world,c['tile_size'],locator.grid,c.get('fusion_method','product'),flags)
    fig,axes=plt.subplots(1,4,figsize=(18,4),layout='constrained')
    cmap=plt.get_cmap('inferno').copy();cmap.set_bad('#b9c7d4')
    axes[0].imshow(response,extent=(0,world,world,0),cmap=cmap,vmin=0,vmax=1)
    centers=[]
    for h in row['history']:
        if h['window']:
            x,y,s=h['window'];centers.append((x+s/2,y+s/2))
    if centers:axes[0].plot(*np.array(centers).T,'o-',color='cyan',ms=2)
    axes[0].set_title('Acquired response and path')
    vmax=max(float(fields[0].max()),float(fields[-1].max()))
    for ax,p,label in zip(axes[1:3],(fields[0],fields[-1]),('Initial belief','Final belief')):
        im=ax.imshow(p.reshape(locator.grid,locator.grid),extent=(0,world,world,0),cmap='inferno',vmin=0,vmax=vmax)
        ax.set_title(label);fig.colorbar(im,ax=ax,shrink=.75)
    for ax in axes[:3]:
        ax.scatter(*source,marker='+',c='lime',s=65);ax.set(xlim=(0,world),ylim=(world,0))
    curve=row['curve'];axes[3].plot([r['spent'] for r in curve],[r['mass16'] for r in curve],label='Truth mass @16')
    axes[3].plot([r['spent'] for r in curve],[r['mass32'] for r in curve],label='Truth mass @32')
    axes[3].set(xlabel='Unique native pixels',ylabel='Probability mass',ylim=(0,1));axes[3].legend()
    fig.suptitle(f'Scene {scene} | {row["policy"]} | MAP error {row["map_distance"]:.1f}px | truth annotation only')
    fig.savefig(directory/(prefix+'.png'),dpi=130);plt.close(fig)


def summarize(rows,c,output,protocol,checkpoint):
    expected={(s,b,p) for s in protocol['scene_splits']['test'] for b in c['evaluation_budgets'] for p in policy_names(c)}
    actual=[(r['scene_id'],r['budget'],r['policy']) for r in rows]
    if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('missing/duplicated locked-test episodes')
    groups={};rng=np.random.default_rng(c['seed']);keys=('mass16','mass32','map_distance','expected_distance','core90_area','core90_covers_truth','spent','score','return_value')
    for budget in c['evaluation_budgets']:
        selected=[r for r in rows if r['budget']==budget];group={}
        for policy in policy_names(c):
            values=[r for r in selected if r['policy']==policy]
            group[policy]={key:float(np.mean([r[key] for r in values])) for key in keys}
            group[policy].update(windows=len(values),recall16=float(np.mean([r['map_distance']<=16 for r in values])),
                recall32=float(np.mean([r['map_distance']<=32 for r in values])))
        paired={};ours={r['scene_id']:r for r in selected if r['policy']=='ppo'}
        for baseline in policy_names(c)[1:]:
            other={r['scene_id']:r for r in selected if r['policy']==baseline}
            if set(ours)!=set(other):raise ValueError('unpaired evaluation scenes')
            differences=np.array([[ours[s][key]-other[s][key] for key in ('mass32','map_distance','score')] for s in sorted(ours)])
            scene_order=sorted(ours);clusters={}
            for j,s in enumerate(scene_order):clusters.setdefault(ours[s]['geometry_id'],[]).append(j)
            blocks=[np.array(v,dtype=int) for v in clusters.values()];boots=[]
            for _ in range(c['bootstrap_repetitions']):
                ix=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))]);boots.append(differences[ix].mean(0))
            interval=np.quantile(boots,[.025,.975],axis=0) if boots else np.full((2,3),np.nan)
            paired[baseline]={key:dict(mean_change=float(differences[:,j].mean()),ci95=interval[:,j].tolist()) for j,key in enumerate(('mass32','map_distance','score'))}
        groups[str(budget)]=dict(policies=group,paired_change_vs_baseline=paired)
    result=dict(schema='rind-active-rl-evaluation-v1',budget_unit='unique acquired native pixels, initial included',
        groups=groups,implementation_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        policy_checkpoint=checkpoint,bootstrap_unit='exact obstacle geometry group (paired)',
        scenes=len(protocol['scene_splits']['test']),test_geometry_groups=len({r['geometry_id'] for r in rows}),source_epoch=protocol['source_epoch'],
        source_sha256=protocol['source_sha256'],reward='delta(.5 log mass16 + .5 log mass32), gamma=1',
        location_entropy_reward=False,source_containing_windows_filtered=False,constant_tiles='neutral evidence; cost counted',
        fusion=protocol['inference_fusion'],environment=c.get('environment','window'),fusion_method=c.get('fusion_method','product'),
        tile_size=c['tile_size'],timing='shared locator cache; wall times are not a cold-cache latency benchmark')
    directory=output/'evaluation';write_json(directory/'metrics.json',result)
    write_json(directory/'episodes.json',rows)
    lines=['# 单源主动选窗测试','',f'固定测试场景：{result["scenes"]}；RL best update：{checkpoint["update"]}；冻结Phase1 epoch：{protocol["source_epoch"]}。',
        '','奖励为16/32像素真源邻域对数质量的等权增量。概率场熵不进入奖励；策略动作熵仅用于PPO探索。',
        '','| 预算 | 策略 | M16 | M32 | MAP距离 | Recall@32 | 90%区域面积 | 真源覆盖率 | 实际像素 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for budget,v in groups.items():
        for name,m in v['policies'].items():lines.append(f'| {budget} | {name} | {m["mass16"]:.4f} | {m["mass32"]:.4f} | {m["map_distance"]:.1f} | {m["recall32"]:.1%} | {m["core90_area"]:.0f} | {m["core90_covers_truth"]:.1%} | {m["spent"]:.0f} |')
    lines+=['','差值与按几何组配对bootstrap区间见metrics.json。MAP距离越低越好，真源质量/Recall越高越好；区域缩小应结合真源覆盖率解释。',
        '',f'观测方式：{c.get("environment","window")}；每块{c["tile_size"]}×{c["tile_size"]}；融合：{c.get("fusion_method","product")}。',
        '最小值融合取各个有效小窗原始预测的逐点最小值，然后归一化；全亮/全暗块作为中性证据，不参与最小值，但照常收费。它是共识分布，不是精确后验。'
        if c.get('fusion_method')=='minimum' else '融合为去重后的温和乘积近似，共享未知几何导致的证据相关性尚未被完全消除。',
        '包含真源的块可能超出Phase1原训练分布，不做真值筛除。',
        '','这是策略与固定定位/融合系统的对比，不表示所有场景可唯一定位，也不表示已在真实未知场验证。']
    (directory/'report.zh-CN.md').write_text('\n'.join(lines)+'\n')
    links=[f'<li><a href="figures/{p.name}">{p.name}</a></li>' for p in sorted((output/'figures').glob('*'))
           if p.suffix in ('.html','.png')] if (output/'figures').exists() else []
    (output/'index.html').write_text('<meta charset="utf-8"><title>RIND active windows</title><h1>Single-source RL</h1><p><a href="evaluation/report.zh-CN.md">Report</a></p><ul>'+''.join(links)+'</ul>')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','train','evaluate','all'))
    parser.add_argument('--config');parser.add_argument('--run-name',default='single_source_rl_v1')
    parser.add_argument('--phase1-run');parser.add_argument('--data-root');parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--scene-counts',type=int,nargs=3);parser.add_argument('--updates',type=int)
    parser.add_argument('--num-envs',type=int);parser.add_argument('--rollout-steps',type=int)
    parser.add_argument('--set',action='append',default=[])
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',args.run_name):parser.error('invalid run name')
    output=PROJECT_ROOT/'outputs/active_rl'/args.run_name
    if args.config is None and (output/'protocol.json').exists():
        base=json.loads((output/'protocol.json').read_text())['config']
        c=load_config(args,base)
    else:
        args.config=args.config or 'configs/active_rl.json';c=load_config(args)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if args.command=='prepare':prepare(c,output);return
    device=torch.device(f'cuda:{local}' if args.device=='cuda' else 'cpu')
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; choose --device cpu explicitly')
        torch.cuda.set_device(device);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.benchmark=True
    torch.set_num_threads(max(1,c['threads']//world));torch.manual_seed(c['seed'])
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    try:
        if rank==0:prepare(c,output)
        if world>1:dist.barrier()
        protocol=json.loads((output/'protocol.json').read_text())
        dataset=Phase1Dataset(protocol['data_root'])
        if sha(Path(protocol['data_root'])/'manifest.json')!=protocol['data_manifest_sha256']:raise ValueError('data changed after preparation')
        locator=FrozenLocator(dataset,output/'phase1_frozen.pt',device,tile=c['tile_size'],spacing=c['spacing'],
            chunk=c['candidate_chunk'],batch_size=c['locator_batch_size'],cache_size=c['tile_cache_size'])
        policy=make_policy(c).to(device)
        log(f'environment={c.get("environment","window")} | tile={c["tile_size"]} | fusion={c.get("fusion_method","product")} | rank={rank}/{world} | policy parameters={sum(p.numel() for p in policy.parameters()):,}')
        if args.command in ('train','all'):train(policy,protocol,c,locator,output,rank,world)
        if args.command in ('evaluate','all'):
            metadata=[None]
            if rank==0:
                raw=(output/'best.pt').read_bytes();fingerprint=hashlib.sha256(raw).hexdigest()
                snapshot=output/'evaluation/policy_snapshots'/(fingerprint+'.pt');snapshot.parent.mkdir(parents=True,exist_ok=True)
                if not snapshot.exists():snapshot.write_bytes(raw)
                metadata[0]=dict(snapshot=str(snapshot),sha256=fingerprint)
            if world>1:dist.broadcast_object_list(metadata,src=0)
            checkpoint=metadata[0]
            if sha(checkpoint['snapshot'])!=checkpoint['sha256']:raise ValueError('evaluation policy snapshot changed')
            saved=torch.load(checkpoint['snapshot'],map_location='cpu',weights_only=True)
            checkpoint['update']=saved['update']
            if saved['protocol_sha256']!=sha(output/'protocol.json'):raise ValueError('policy/protocol mismatch')
            policy.load_state_dict(saved['policy']);policy.eval()
            initial=make_policy(c).to(device)
            initial.load_state_dict(torch.load(output/'initial.pt',map_location='cpu',weights_only=True)['policy']);initial.eval()
            if c.get('environment')=='frontier':
                validation_rows=evaluate_frontier_rows(policy,protocol,c,locator,rank,world,split='validation',full_validation=True)
                if rank==0:
                    write_json(output/'evaluation/validation_final.json',dict(
                        scenes=len(validation_rows),selection_probe_scenes=min(c['validation_probe_scenes'],len(validation_rows)),
                        policy_checkpoint=checkpoint,selected_using_test=False,
                        mean={k:float(np.mean([r[k] for r in validation_rows])) for k in ('score','mass16','mass32','map_distance','core90_area','core90_covers_truth')},
                        recall32=float(np.mean([r['map_distance']<=32 for r in validation_rows])),episodes=validation_rows))
            rows=evaluate_rows(policy,protocol,c,locator,rank,world,split='test',baselines=True,initial=initial)
            if rank==0:
                for scene in protocol['scene_splits']['test'][:c['plot_scenes']]:
                    for kind,p in [(name,policy if name=='ppo' else None) for name in policy_names(c) if name!='initial_policy']:
                        seed=int(np.random.SeedSequence([c['seed'],scene,c['budget'],19]).generate_state(1)[0])
                        trace=episode(p,kind,scene,c['budget'],protocol,c,locator,seed,trace=True)
                        save_trace(trace,protocol,c,locator,output)
                result=summarize(rows,c,output,protocol,checkpoint);log(f'RL evaluation saved: {output/"evaluation/report.zh-CN.md"}')
    finally:
        if world>1 and dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
