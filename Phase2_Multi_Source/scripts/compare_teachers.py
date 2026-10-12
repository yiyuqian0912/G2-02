"""Finite-grid exhaustive MAE-profile teacher versus the actual layered teacher."""
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
from rind_phase2.layered_teacher import generate_layered_teacher, NoLayeredSolutionError


def fit_strengths(v, y, lower=1e-6):
    """Global bounded L1 optimum. Compress identical (visibility, response) rows."""
    from scipy.optimize import linprog
    rows, weights = np.unique(np.column_stack((v, y)), axis=0, return_counts=True)
    a, b = rows[:, :-1], rows[:, -1]
    m, k = a.shape
    eye = np.eye(m)
    result = linprog(np.r_[np.zeros(k), weights / weights.sum()],
        A_ub=np.vstack((np.c_[a, -eye], np.c_[-a, -eye])), b_ub=np.r_[b, -b],
        bounds=[(lower, 1.0)] * k + [(0, None)] * m, method='highs')
    if not result.success:
        raise RuntimeError('Strength LP failed: ' + result.message)
    strengths = result.x[:k]
    return strengths, float(np.average(np.abs(a @ strengths - b), weights=weights))


def probabilities(cost, tau):
    z = np.exp(-(cost - np.min(cost)) / tau)
    return z / z.sum()


def similarity(p, q):
    m = (p + q) / 2
    def kl(a):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / m[mask])))
    return {'TV': float(np.abs(p-q).sum()/2), 'JS_bits': (kl(p)+kl(q))/2,
            'overlap': float(np.minimum(p,q).sum())}


def marginal(combos, q, n):
    out = np.zeros(n)
    for ids, weight in zip(combos, q):
        np.add.at(out, ids, weight / len(ids))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-id', type=int, required=True)
    p.add_argument('--view-id', type=int, required=True)
    p.add_argument('--data-root')
    p.add_argument('--spacing', type=float, default=256)
    p.add_argument('--counts', type=int, nargs='+', default=[1,2,3,4])
    p.add_argument('--temperature', type=float, default=.01)
    p.add_argument('--min-strength', type=float, default=1e-6)
    p.add_argument('--max-combinations', type=int, default=100000)
    p.add_argument('--seed', type=int, default=20261011)
    p.add_argument('--beam-width', type=int, default=64)
    p.add_argument('--samples-per-pattern', type=int, default=8)
    p.add_argument('--max-final-mae', type=float, default=.01)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not 0 < args.min_strength < 1 or not np.isfinite(args.temperature) or args.temperature <= 0:
        p.error('Require 0 < min-strength < 1 and temperature > 0')
    if any(k < 1 or k > 4 for k in args.counts) or len(set(args.counts)) != len(args.counts):
        p.error('counts must be distinct members of 1,2,3,4')
    ds = Phase2Dataset(args.data_root)
    ev = JointResponseEvaluator(ds, args.scene_id, args.view_id)
    grid = uniform_source_grid(ev.domain, ev, spacing=args.spacing)
    xy = grid.positions[grid.valid]
    n = len(xy)
    counts = {k: math.comb(n,k) if n >= k else 0 for k in args.counts}
    print(json.dumps({'valid_positions':n,'combinations':counts,'total':sum(counts.values())}), flush=True)
    if not n or sum(counts.values()) > args.max_combinations:
        p.error('Exhaustive budget exceeded or empty grid. Increase spacing or explicitly raise max-combinations; no truncation performed.')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    y = ev.observed_response.ravel()
    visibility = np.array([ev.visibility_at(pos).ravel() for pos in xy], dtype=bool).T
    # Cache keyed by actual rendered input, grid, bounds and solver code, not filenames.
    digest = hashlib.sha256(visibility.tobytes()+y.tobytes()+xy.tobytes()+str(args.min_strength).encode()+Path(__file__).read_bytes()).hexdigest()
    np.savez_compressed(args.output/'inputs.npz', positions=xy, visibility=visibility,
                        response=ev.observed_response, window=ev.observation['window'])
    lookup = {tuple(pos):i for i,pos in enumerate(xy)}
    layered = None
    try:
        layered = generate_layered_teacher(ds, args.scene_id, args.view_id,
            spacing=args.spacing, temperature=args.temperature, seed=args.seed,
            max_sources=max(args.counts), beam_width=args.beam_width,
            samples_per_pattern=args.samples_per_pattern, max_final_mae=args.max_final_mae)
        from rind_phase2.teacher import save_teacher_record
        save_teacher_record(layered, args.output/'layered.npz')
        layer_status = {'status':'completed','search':layered['metadata']['layered_search']}
    except NoLayeredSolutionError as exc:
        layer_status = {'status':'no_solution','message':str(exc),'details':exc.report}
    report = {'parameters':{**vars(args),'output':str(args.output)},'data_root':str(ds.root),
              'n':n,'combinations':counts,'input_hash':digest,'layered':layer_status,
              'semantics':'Uniform prior over distinct unordered grid positions, separately normalized per K. Strength-profile Gibbs distribution, not strength-integrated posterior. Maps = expected source count / K. No prior over K.',
              'strength_bounds':[args.min_strength,1.0], 'by_k':{}}
    maps = []
    for k, total in counts.items():
        if not total:
            report['by_k'][k]={'status':'fewer than K positions'}
            continue
        t = time.perf_counter()
        cache = args.output/f'fit-k{k}-{digest[:16]}.npz'
        if cache.exists():
            with np.load(cache) as z:
                combos, strengths, costs = z['combos'], z['strengths'], z['costs']
        else:
            combos = np.array(list(itertools.combinations(range(n),k)),dtype=np.int32)
            strengths, costs = np.empty((total,k)), np.empty(total)
            # Many positions induce the same local mask; solve each mask multiset once.
            _, mask_ids = np.unique(visibility.T, axis=0, return_inverse=True)
            fit_cache = {}
            for i, ids in enumerate(combos):
                order = np.argsort(mask_ids[ids],kind='stable')
                ordered = ids[order]
                key = tuple(mask_ids[ordered])
                if key not in fit_cache:
                    fit_cache[key] = fit_strengths(visibility[:,ordered],y,args.min_strength)
                a, costs[i] = fit_cache[key]
                strengths[i,order] = a
            np.savez_compressed(cache,combos=combos,strengths=strengths,costs=costs)
        q = probabilities(costs,args.temperature)
        uniform_map = marginal(combos,q,n)
        stats = {'combinations':total,'minimum_MAE':float(costs.min()),'fit_seconds':time.perf_counter()-t}
        empirical = np.zeros(total)
        idx = {tuple(ids):i for i,ids in enumerate(combos)}
        excluded = 0.0
        if layered is not None:
            for i in np.flatnonzero(layered['source_counts']==k):
                sources = layered['sources'][i,:k]
                ids = tuple(sorted(lookup[tuple(row[:2])] for row in sources))
                weight = float(layered['teacher_prob'][i])
                if len(set(ids)) != k or np.any(sources[:,2] < args.min_strength):
                    excluded += weight
                else:
                    empirical[idx[ids]] += weight
        stats['excluded_layered_probability'] = excluded
        refit = np.zeros(total)
        layer_map = refit_map = None
        if empirical.sum() > 0:
            stats['layered_retained_K_probability'] = float(empirical.sum())
            empirical /= empirical.sum()
            selected = empirical > 0
            refit = np.zeros(total)
            refit[selected] = probabilities(costs[selected],args.temperature)
            layer_map, refit_map = marginal(combos,empirical,n), marginal(combos,refit,n)
            stats.update(status='compared',selected_position_sets=int(selected.sum()),
                exhaustive_mass_covered=float(q[selected].sum()),
                joint_actual=similarity(q,empirical), joint_refit=similarity(q,refit),
                spatial_actual=similarity(uniform_map,layer_map), spatial_refit=similarity(uniform_map,refit_map))
        else:
            stats['status']='no comparable layered configurations at this K'
        np.savez_compressed(args.output/f'distributions-k{k}.npz', positions=xy,combos=combos,
                            strengths=strengths,costs=costs,exhaustive=q,layered=empirical,refit=refit,
                            exhaustive_spatial=uniform_map,
                            layered_spatial=np.array([]) if layer_map is None else layer_map,
                            refit_spatial=np.array([]) if refit_map is None else refit_map)
        maps.append((k,uniform_map,layer_map,refit_map))
        report['by_k'][k]=stats
        print(f'K={k}: {stats}',flush=True)
    report['seconds']=time.perf_counter()-started
    report['unit_renders_exhaustive']=ev.num_source_renders
    report['visibility_bytes']=visibility.nbytes
    (args.output/'report.json').write_text(json.dumps(report,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else str(a)))
    plot(args, ev, grid, xy, maps)


def plot(args, ev, grid, xy, maps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(len(maps)+1,4,figsize=(16,4*(len(maps)+1)),squeeze=False)
    obs_im=axes[0,0].imshow(ev.observed_response,cmap='viridis',vmin=0,vmax=max(float(ev.observed_response.max()),1e-12))
    fig.colorbar(obs_im,ax=axes[0,0],shrink=.7,label='Intensity')
    axes[0,0].set_title('Observed additive response')
    axes[0,1].imshow(ev.observation['obstacle'],cmap='gray_r')
    axes[0,1].set_title('Local obstacle mask')
    for ax in axes[0,2:]: ax.axis('off')
    axes[0,2].text(0,1,'Each K normalized separately.\nMaps: expected source count / K (sum = 1).\nGrey: excluded grid cells. No interpolation.\nMissing layered result is not a zero distribution.\nRefit uses uniform mass on selected position sets.',va='top')
    ix = grid.grid_index[grid.valid]
    for row,(k,base,actual,refit) in enumerate(maps,1):
        vmax=max(float(base.max()),float(actual.max()) if actual is not None else 0,float(refit.max()) if refit is not None else 0)
        for col,(values,title) in enumerate(((base,'Exhaustive'),(actual,'Layered actual'),(refit,'Layered positions, refit'),(None if actual is None else actual-base,'Actual minus exhaustive'))):
            ax=axes[row,col]
            ax.set_title(f'K={k} | {title}')
            if values is None:
                ax.text(.5,.5,'No comparable layered result',ha='center'); ax.axis('off'); continue
            arr=np.full((len(grid.y_axis),len(grid.x_axis)),np.nan)
            arr[ix[:,0],ix[:,1]]=values
            cmap=plt.get_cmap('coolwarm' if col==3 else 'magma').copy(); cmap.set_bad('#cccccc')
            extent=(0,len(grid.x_axis)*args.spacing,len(grid.y_axis)*args.spacing,0)
            im=ax.imshow(arr,extent=extent,interpolation='nearest',cmap=cmap,vmin=-vmax if col==3 else 0,vmax=vmax)
            x,y,L=ev.observation['window']; ax.add_patch(Rectangle((x,y),L,L,fill=False,edgecolor='cyan'))
            ax.set_xlim(0,ev.world_size); ax.set_ylim(ev.world_size,0)
            fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle(f'Scene {args.scene_id}, view {args.view_id} | finite grid spacing {args.spacing:g}')
    fig.tight_layout(); fig.savefig(args.output/'comparison.png',dpi=150); plt.close(fig)

if __name__=='__main__': main()
