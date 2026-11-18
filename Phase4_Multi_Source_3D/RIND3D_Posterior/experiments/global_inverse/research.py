"""Single CPU global experiment. solve never reads source coordinates or label files."""
import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
import io
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
from rind3d.geometry import load_geometry
from rind3d.rendering.polyhedra import decode_convex
from solver import Budget, Limit, Observation, Poly, apply_pair, choose_pairs, cone, integer_row
from poly import grid_pieces, within
from pilot import cropz


def task_manifest(comparison, root, exclude=()):
    original=json.loads((comparison/'tasks.json').read_bytes())
    old={int(r['slot']) for r in original}|set(exclude)
    selected={}
    for f in sorted(root.glob('scenes/*/*.npz')):
        with np.load(f,allow_pickle=False) as z:
            meta=json.loads(z['meta'].tobytes())
            slot=int(meta['slot'])
            if slot in old: continue
            for w,win in enumerate(z['windows']):
                side=int(win['L'])
                family=meta['family']
                mask=int(win['mask'])
                if not mask or (family,side) in selected: continue
                channel=(mask & -mask).bit_length()-1
                selected[family,side]=dict(slot=slot,w=w,s=channel,L=side,family=family,
                                           key=f'{slot:06d}_w{w:02d}_s{channel}',container=str(f))
    def compact(r):
        return {k:r[k] for k in ('slot','w','s','L','family','key')}
    return {'archive':[compact(r) for r in original],
            'holdout':[compact(selected[k]) for k in sorted(selected)]}


def read_obs(root,t):
    slot=t['slot']
    f=root/'scenes'/f'{slot//1000:04d}'/f'{slot:06d}.npz'
    with np.load(f,allow_pickle=False) as z:
        meta=json.loads(z['meta'].tobytes())
        w=t['w']
        win=z['windows'][w]
        side=int(win['L'])
        if side!=t['L'] or meta['family']!=t['family'] or not (int(win['mask'])>>t['s']&1):
            raise ValueError('manifest does not identify an eligible channel of this geometry')
        origin=tuple(int(win[k]) for k in ('x','y','z'))
        offsets={(int(i),int(j)):(int(a),int(n)) for i,j,a,n in z['obs_index']}
        blob=z['obs'].tobytes()
        cls=cropz(blob,offsets,w,t['s'],side)
        solid=cropz(blob,offsets,w,-1,side).astype(bool)
        if np.any(cls & solid): raise ValueError('active SOLID voxel in stored observation')
        cls[solid]=2
        geom=z['geometry'].tobytes()
    bodies=[tuple(integer_row(r) for r in b.rows) for b in decode_convex(load_geometry(io.BytesIO(geom)))]
    return Observation(bodies,cls,origin,int(meta['resolution'])),sha256(geom).hexdigest()


def mass_summary(yes,maybe,grid_limit,additional=()):
    scale=1<<160
    def enclosure(parts):
        lo=hi=0
        for p in parts:
            v=p.volume*scale
            lo+=v.numerator//v.denominator
            hi+=-(-v.numerator//v.denominator)
        return lo,hi
    lower,upper_yes=enclosure(yes)
    _,upper_unknown=enclosure(maybe)
    extra,_=enclosure(additional)
    outer=upper_yes+upper_unknown
    lower+=extra
    if lower>outer: raise ArithmeticError("inner volume exceeds outer bound")
    gap=outer-lower
    out=dict(scale_bits=160,known_lower=lower,outer_upper=outer,unknown_upper=gap,
             outer=float(Fraction(upper_yes+upper_unknown,scale)),positive=lower>0,
             tv_upper=float(Fraction(gap,outer)) if lower else None)
    inside=[*yes,*additional]
    out['certified_tv_005']=bool(lower and gap*20<=outer)
    if inside:
        # Deliver a distribution even when the deterministic error bound is loose.
        masses={}
        count_pieces=0
        grid_clock=perf_counter()
        for p in inside:
            dims=p.grid_bounds(128)
            count=1
            for l,h in dims: count*=h-l+1
            if gap*20>outer and (count>grid_limit or len(masses)+count>grid_limit):
                out['grid_status']='lazy_polytope_queries'
                break
            for ijk,piece in grid_pieces(p,128):
                cell=(ijk[0]*128+ijk[1])*128+ijk[2]
                masses[cell]=masses.get(cell,0)+piece.volume
                count_pieces+=1
                if count_pieces>=grid_limit or perf_counter()-grid_clock>6:
                    out['grid_status']='lazy_polytope_queries'
                    break
            if out.get('grid_status')=='lazy_polytope_queries': break
        else:
            total=sum(masses.values())
            if total!=sum(p.volume for p in inside): raise ArithmeticError('cell volumes do not sum')
            normalized={c:m/total for c,m in sorted(masses.items())}
            if any(max(v.numerator.bit_length(), v.denominator.bit_length())>13000 for v in normalized.values()):
                # Avoid impractically large decimal JSON entries. The exact mesh is retained.
                out['grid_status']='lazy_polytope_queries'
            else:
                out['grid_status']='exact_normalized' if not maybe else 'inner_normalized'
                out['probabilities']={c:str(v) for c,v in normalized.items()}
        out['inner_pieces']=[p.faces for p in inside]
    return out


def grid_atom_bound(inner,outer):
    """TV(delta_c,q_F) = 1-q_F(c). Bound that quantity without solving F inside c."""
    if not inner:
        return None
    peak=max(inner,key=lambda p:p.volume)
    v=tuple(map(sum,zip(*peak.vertices)))
    xyz=tuple(x*128//v[3] for x in v[:3])
    rows=[]
    for axis,i in enumerate(xyz):
        n=tuple(128*int(j==axis) for j in range(3))
        rows.extend([(*n,-i-1),(*(-a for a in n),i)])
    def in_cell(piece):
        clipped=within(piece,rows)
        return clipped.volume if clipped is not None else 0
    # Aggregate on the same directed lattice as the continuous mass proof.
    # Accumulating hundreds of unrelated rational denominators is unnecessary.
    scale=1<<160
    lower=sum(int(in_cell(p)*scale) for p in inner)
    upper=0
    for p in outer:
        v=(p.volume-in_cell(p))*scale
        if v<0: raise ArithmeticError('negative outside volume')
        upper+=-(-v.numerator//v.denominator)
    if not lower: return None
    return dict(cell=(xyz[0]*128+xyz[1])*128+xyz[2],lower_inside=lower,
                upper_outside=upper,scale_bits=160,tv=str(Fraction(upper,upper+lower)),
                certified_tv_005=20*upper<=upper+lower)


def run(obs, strategy, seed, rounds, seconds, cap, depth, method, project=False):
    t0=perf_counter()
    edges=choose_pairs(obs.cls,rounds,seed,strategy)
    parts=[Poly.cube()]
    trace=[]
    discovery=Budget(seconds,cap)
    reason='all_pairs'
    for dark,active in edges:
        try:
            d,a=obs.world(dark),obs.world(active)
            dk=[cone(body,d) for body in obs.bodies]
            ak=[cone(body,a) for body in obs.bodies]
            new=[]
            for p in parts:
                new.extend(apply_pair(p,dk,ak,discovery))
                discovery.check(len(new))
            parts=new
            trace.append(dict(d=dark,a=active,pieces=len(parts)))
        except Limit as e:
            reason=str(e)
            break
        if not parts: break
    discovery_s=perf_counter()-t0
    # A separate bounded finishing attempt; an interrupted operation restores
    # its parent. Keep all unexplored pieces for the upper volume certificate.
    finish=Budget(seconds,cap)
    yes,maybe=[],[]
    stats=Counter()
    phase='complete_traversal'
    additional=[]
    lower_stop='not_needed'
    try:
        parts=obs.exclude_window(parts,finish)
    except Limit as e:
        phase='window_'+str(e)
        maybe=parts
    else:
        if method in ("raster","raster_support"):
            obs.use_support=method=="raster_support"
            from raster import finish_raster,inner_raster
            yes,maybe,stats,phase=finish_raster(obs,parts,finish)
            additional,extra_stats,lower_stop=inner_raster(obs,maybe,Budget(seconds,cap))
            stats.update(extra_stats)
            parts=[]
        if method=="envelope":
            from envelopes import Compiler
            compiler=Compiler(obs,depth)
            yes,maybe,phase,visibility_only=compiler.finish(parts,finish)
            from envelopes import legal_inner
            proved={id(p) for p in visibility_only}
            lower_budget=Budget(seconds,cap)
            try:
                for p in maybe:
                    subset=([p] if id(p) in proved else
                            compiler.lower_block([p],(0,0,0),obs.side,depth,lower_budget))
                    addition=[]
                    for q in subset:
                        addition.extend(legal_inner(obs,q,lower_budget))
                    lower_budget.check(len(addition))
                    additional.extend(addition)
                lower_stop='complete_traversal'
            except Limit as e:
                lower_stop=str(e)
            stats=compiler.stats
            parts=[]
        for i,p in enumerate(parts):
            try:
                q,proved=obs.certify_piece(p,depth,finish,stats)
            except Limit as e:
                phase=str(e)
                maybe.extend(parts[i:])
                break
            if q is not None: (yes if proved else maybe).append(q)
    elapsed=perf_counter()-t0
    result=dict(discovery_s=discovery_s,finish_s=elapsed-discovery_s,trace=trace,
                discovery_stop=reason,finish_stop=phase,known_pieces=len(yes),unknown_pieces=len(maybe),
                additional_inner_pieces=len(additional),lower_stop=lower_stop,receiver_product_counts=stats,**mass_summary(yes,maybe,2048,additional))
    if project:
        start=perf_counter()
        result['atom']=grid_atom_bound(yes+additional,yes+maybe)
        result['atom_s']=perf_counter()-start
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('manifest',type=Path)
    p.add_argument('--make-manifest',type=Path,metavar='COMPARISON')
    p.add_argument('--split',choices=['archive','holdout'],default='archive')
    p.add_argument('--indices',type=int,nargs='+')
    p.add_argument('--strategy',choices=['interface','random_classes'],default='interface')
    p.add_argument('--seconds',type=float,default=3)
    p.add_argument('--rounds',type=int,default=24)
    p.add_argument('--cap',type=int,default=128)
    p.add_argument('--depth',type=int,default=2)
    p.add_argument('--method',choices=['pencil','envelope','raster','raster_support'],default='envelope')
    p.add_argument('--project', action='store_true', help='also bound the global TV of a single-cell output')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.make_manifest:
        args.manifest.write_text(json.dumps(task_manifest(args.make_manifest,args.root),indent=1)+'\n')
        return
    tasks=json.loads(args.manifest.read_bytes())[args.split]
    with args.output.open('x') as fp:
        for i in args.indices if args.indices is not None else range(len(tasks)):
            task=tasks[i]
            start,cpu=perf_counter(),process_time()
            obs,digest=read_obs(args.root,task)
            seed=int.from_bytes(sha256(task['key'].encode()).digest()[:4],'big')
            r=run(obs,args.strategy,seed,args.rounds,args.seconds,args.cap,args.depth,args.method,args.project)
            r.update(task=task,geometry_sha256=digest,wall_s=perf_counter()-start,cpu_s=process_time()-cpu,
                     parameters=dict(strategy=args.strategy,seconds_per_stage=args.seconds,cap=args.cap,
                                     rounds=args.rounds,depth=args.depth,seed=seed,method=args.method))
            fp.write(json.dumps(r,separators=(',',':'))+'\n')
            fp.flush()
            print(i,task['key'],len(r['trace']),r['known_pieces'],r['unknown_pieces'],r['outer'],round(r['wall_s'],3),flush=True)


if __name__=='__main__':
    main()
