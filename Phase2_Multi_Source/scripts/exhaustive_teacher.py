"""Shadow-screened finite-grid exhaustive teacher; no layered search is run."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
from rind_phase2.data import Phase2Dataset
from rind_phase2.physics import JointResponseEvaluator
from rind_phase2.search import uniform_source_grid
from compare_teachers import fit_strengths, probabilities


def window_status(response, obstacle):
    values = response[~obstacle]
    if not values.size: return 'no_free_pixels'
    if not np.any(values > 0): return 'all_dark'
    if not np.any(values == 0): return 'all_lit'
    return 'eligible'


def allowed_by_shadow(visibility, dark):
    return not np.any(visibility[dark])


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str))


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    with np.load(out/'screen.npz') as z: d={k:z[k] for k in z.files}
    report=json.loads((out/'report.json').read_text())
    files=[out/f'k{k}.npz' for k in report.get('completed_counts',[])]
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    obs=np.ma.array(d['response'],mask=d['obstacle'])
    im=axs[0].imshow(obs,vmin=0,cmap='viridis'); fig.colorbar(im,ax=axs[0],label='Additive intensity')
    axs[0].set_title('Observation (obstacles masked)')
    axs[1].imshow(d['dark'],vmin=0,vmax=1,cmap='gray'); axs[1].set_title('Zero-response free pixels used for screening')
    size=int(d['axis_size']); status=np.full((size,size),np.nan)
    ij=d['indices']; status[ij[:,0],ij[:,1]]=d['keep'].astype(int)
    extent=(0,size*float(d['spacing']),size*float(d['spacing']),0)
    im=axs[2].imshow(status,extent=extent,vmin=0,vmax=1,cmap='RdYlGn',interpolation='nearest')
    axs[2].set_title('Candidate domain: red excluded, green retained')
    x,y,L=d['window']; axs[2].add_patch(Rectangle((x,y),L,L,fill=False,edgecolor='cyan'))
    fig.tight_layout(); fig.savefig(out/'screening.png',dpi=160); plt.close(fig)
    if not files: return
    fig,axs=plt.subplots(len(files),2,figsize=(11,4.5*len(files)),squeeze=False)
    for row,file in enumerate(files):
        with np.load(file) as z:
            marginal=z['spatial']; best=z['best_sources']; best_response=z['best_response']; k=int(z['k']); cost=float(z['minimum_mae'])
        arr=np.full((size,size),np.nan); retained=ij[d['keep']]
        arr[retained[:,0],retained[:,1]]=marginal
        im=axs[row,0].imshow(arr,extent=extent,cmap='magma',vmin=0,interpolation='nearest')
        axs[row,0].add_patch(Rectangle((x,y),L,L,fill=False,edgecolor='cyan'))
        axs[row,0].scatter(best[:,0],best[:,1],marker='+',c='cyan',s=35)
        axs[row,0].set_title(f'K={k}: E[count at point]/K (sum=1)')
        fig.colorbar(im,ax=axs[row,0])
        im=axs[row,1].imshow(np.ma.array(best_response,mask=d['obstacle']),vmin=0,vmax=max(float(d['response'].max()),float(best_response.max())),cmap='viridis')
        axs[row,1].set_title(f'Best reconstruction | MAE={cost:.5g}')
        fig.colorbar(im,ax=axs[row,1])
    fig.tight_layout(); fig.savefig(out/'distributions.png',dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-id',type=int)
    p.add_argument('--view-id',type=int)
    p.add_argument('--list-views',action='store_true')
    p.add_argument('--data-root')
    p.add_argument('--spacing',type=float,default=16)
    p.add_argument('--counts',nargs='+',type=int,default=[1,2,3,4])
    p.add_argument('--temperature',type=float,default=.01)
    p.add_argument('--min-strength',type=float,default=1e-6)
    p.add_argument('--max-combinations',type=int,default=1000000)
    p.add_argument('--screen-only',action='store_true')
    p.add_argument('--plot-only',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); out=a.output
    if a.plot_only: plot(out); return
    if a.scene_id is None: p.error('--scene-id required')
    if not np.isfinite(a.temperature) or a.temperature<=0 or not 0<a.min_strength<1: p.error('Invalid temperature or strength bound')
    if not a.counts or any(k not in (1,2,3,4) for k in a.counts) or len(set(a.counts))!=len(a.counts): p.error('counts must be distinct values 1..4')
    ds=Phase2Dataset(a.data_root)
    if a.list_views:
        rows=[]
        for vid in range(int(ds._base.view_counts[a.scene_id])):
            o=ds.get_observation(a.scene_id,vid); x,y,L=map(int,o['window'])
            response=ds.get_region(a.scene_id,x,y,L)['response']
            rows.append({'scene_id':a.scene_id,'view_id':vid,'window':o['window'].tolist(),'status':window_status(response,o['obstacle'])})
        out.mkdir(parents=True,exist_ok=True); write_json(out/'views.json',rows)
        print(json.dumps([r for r in rows if r['status']=='eligible'],indent=2)); return
    if a.view_id is None: p.error('--view-id required')
    out.mkdir(parents=True,exist_ok=True)
    # One output directory is one parameter setting; refuse stale mixed results.
    params={k:v for k,v in vars(a).items() if k not in ('screen_only','plot_only','max_combinations','output')}
    params['data_root']=str(ds.root)
    if (out/'parameters.json').exists() and json.loads((out/'parameters.json').read_text())!=params:
        p.error('Output directory belongs to different parameters; choose a new --output')
    write_json(out/'parameters.json',params)
    start=time.perf_counter(); ev=JointResponseEvaluator(ds,a.scene_id,a.view_id)
    response=ev.observed_response; obstacle=ev.observation['obstacle']; status=window_status(response,obstacle)
    report={'parameters':params,'window':ev.observation['window'].tolist(),'status':status,'completed_counts':[],
            'semantics':'Uniform distinct-position combinations on shadow-screened grid, per-K MAE-profile Gibbs distribution; not intensity-integrated posterior.',
            'zero_threshold':0.0,'strength_bounds':[a.min_strength,1.0]}
    if status!='eligible':
        write_json(out/'report.json',report); print(f'Skipped window: {status}. No teacher generated.'); return
    grid=uniform_source_grid(ev.domain,ev,spacing=a.spacing)
    xy=grid.positions[grid.valid]; indices=grid.grid_index[grid.valid]
    dark=(response==0)&~obstacle
    # Fingerprint physical input and implementation; reuse packed masks after interruption.
    h=hashlib.sha256()
    for item in (xy,response,obstacle,ev.obstacle_types,ev.obstacle_params): h.update(np.asarray(item).tobytes())
    h.update(Path(__file__).read_bytes())
    from rind_phase2 import physics
    h.update(Path(physics.__file__).read_bytes())
    from rind_dataset import render, geometry
    for module in (render,geometry): h.update(Path(module.__file__).read_bytes())
    cache=out/f'visibility-{h.hexdigest()[:20]}.npz'
    if cache.exists():
        with np.load(cache) as z: packed=z['packed']; keep=z['keep']
        report['visibility_cache_hit']=True
    else:
        keep=np.zeros(len(xy),dtype=bool); masks=[]
        for i,pos in enumerate(xy):
            vis=ev.visibility_at(pos)
            keep[i]=allowed_by_shadow(vis,dark)
            if keep[i]: masks.append(np.packbits(vis.ravel()))
            if (i+1)%256==0: print(f'Screening {i+1}/{len(xy)}, retained {int(keep.sum())}',flush=True)
        packed=np.array(masks,dtype=np.uint8).reshape(int(keep.sum()),(response.size+7)//8)
        np.savez_compressed(cache,packed=packed,keep=keep)
        report['visibility_cache_hit']=False
    np.savez_compressed(out/'screen.npz',response=response,obstacle=obstacle,dark=dark,
        positions=xy,indices=indices,keep=keep,spacing=a.spacing,axis_size=len(grid.x_axis),window=ev.observation['window'])
    n=int(keep.sum()); counts={k:math.comb(n,k) if n>=k else 0 for k in a.counts}
    report.update(candidate_count_before=len(xy),candidate_count_after=n,rejected_by_shadow=int((~keep).sum()),
                  combinations=counts,screening_seconds=time.perf_counter()-start,
                  retained_invisible_positions='kept: sources invisible throughout the window remain feasible')
    report['status']='screened'; write_json(out/'report.json',report); plot(out)
    print(json.dumps({'n_before':len(xy),'n_after':n,'combinations':counts,'total':sum(counts.values())}),flush=True)
    if a.screen_only: return
    if sum(counts.values())>a.max_combinations:
        report['status']='budget_exceeded'; write_json(out/'report.json',report)
        p.error('Screening saved. Combination budget exceeded; increase --max-combinations explicitly, choose fewer counts or increase spacing. No truncated teacher generated.')
    positions=xy[keep]; vis=np.unpackbits(packed,axis=1,count=response.size).astype(bool)
    _,mask_ids=np.unique(packed,axis=0,return_inverse=True)
    for k,total in counts.items():
        if not total: continue
        path=out/f'k{k}.npz'
        if path.exists():
            with np.load(path) as saved:
                report.setdefault('by_k',{})[k]={'combinations':total,'minimum_mae':float(saved['minimum_mae']),'reused':True}
            report['completed_counts'].append(k); continue
        # Disk arrays avoid building an enormous Python list of all combinations.
        ids_file=out/f'working-k{k}-positions.npy'; cost_file=out/f'working-k{k}-cost.npy'
        ids=np.lib.format.open_memmap(ids_file,mode='w+',dtype=np.int32,shape=(total,k))
        costs=np.lib.format.open_memmap(cost_file,mode='w+',dtype=np.float64,shape=(total,))
        strength_file=out/f'working-k{k}-strengths.npy'
        strengths=np.lib.format.open_memmap(strength_file,mode='w+',dtype=np.float64,shape=(total,k))
        fits={}; t=time.perf_counter()
        for i,combo in enumerate(itertools.combinations(range(n),k)):
            ids[i]=combo; order=np.argsort(mask_ids[ids[i]],kind='stable'); ordered=ids[i,order]
            key=tuple(mask_ids[ordered])
            if key not in fits: fits[key]=fit_strengths(vis[ordered].T,response.ravel(),a.min_strength)
            values,costs[i]=fits[key]; strengths[i,order]=values
            if (i+1)%10000==0: print(f'K={k}: {i+1}/{total}, LPs={len(fits)}',flush=True)
        q=probabilities(costs,a.temperature); spatial=np.zeros(n)
        for col in range(k): np.add.at(spatial,ids[:,col],q/k)
        best=int(np.argmin(costs)); sources=np.c_[positions[ids[best]],strengths[best]]
        reconstruction=strengths[best]@vis[ids[best]].astype(float)
        np.savez_compressed(path,k=k,positions=positions,combos=ids,strengths=strengths,costs=costs,
            probability=q,spatial=spatial,best_sources=sources,best_response=reconstruction.reshape(response.shape),minimum_mae=costs[best])
        report.setdefault('by_k',{})[k]={'combinations':total,'LPs':len(fits),'seconds':time.perf_counter()-t,'minimum_mae':float(costs[best])}
        del ids,costs,strengths
        for file in (ids_file,cost_file,strength_file): file.unlink()
        report['completed_counts'].append(k); write_json(out/'report.json',report); plot(out)
    report['status']='complete' if n else 'no_retained_grid_positions'
    report['elapsed_seconds']=time.perf_counter()-start
    write_json(out/'report.json',report); plot(out)

if __name__=='__main__': main()
