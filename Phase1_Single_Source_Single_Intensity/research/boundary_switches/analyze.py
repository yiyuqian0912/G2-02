#!/usr/bin/env python3
"""Truth-based boundary audit. Does not modify data or training. See README."""
from pathlib import Path
import os,sys,time,json,argparse,csv,gzip,inspect,hashlib
os.environ.setdefault('NUMBA_NUM_THREADS','2')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/rind-boundary-numba')
os.environ.setdefault('MPLCONFIGDIR','/tmp/rind-boundary-mpl')
ROOT=Path(__file__).resolve().parents[2]
VENDOR=ROOT.parent/'Phase2_Multi_Source/vendor/rind-dataset'
sys.path.insert(0,str(VENDOR))
import numpy as np
from numba import njit
from numba.extending import register_jitable
from rind_dataset import RINDDataset
from rind_dataset.runtime import unpack_region_bits,render_scene_region
import rind_dataset.geometry as geo
for name,fn in list(vars(geo).items()):
    if inspect.isfunction(fn) and fn.__module__==geo.__name__:register_jitable(fn)
intersects=geo.segment_intersects_obstacle

@njit(cache=True)
def find(par,a):
    while par[a]!=a:
        par[a]=par[par[a]];a=par[a]
    return a

@njit(cache=True)
def graph(free,levels,x0,y0,types,params,aabbs):
    h,w=free.shape;n=h*w;par=np.arange(n);u=np.empty(2*n,np.int64);v=np.empty(2*n,np.int64)
    count=0;blocked=0
    for y in range(h):
        for x in range(w):
            if not free[y,x]:continue
            for axis in range(2):
                xx=x+int(axis==0);yy=y+int(axis==1)
                if xx>=w or yy>=h or not free[yy,xx]:continue
                px=x0+x+.5;py=y0+y+.5;qx=x0+xx+.5;qy=y0+yy+.5;hit=False
                for j in range(len(types)):
                    if qx<aabbs[j,0] or px>aabbs[j,2] or qy<aabbs[j,1] or py>aabbs[j,3]:continue
                    if intersects(px,py,qx,qy,int(types[j]),params[j]):hit=True;break
                if hit:blocked+=1;continue
                a=y*w+x;b=yy*w+xx;u[count]=a;v[count]=b;count+=1
                if levels[y,x]==levels[yy,xx]:
                    ra=find(par,a);rb=find(par,b)
                    if ra!=rb:par[rb]=ra
    for p in range(n):par[p]=find(par,p)
    return par,u[:count],v[:count],blocked

def response_bins(y,tol):
    vals,inv=np.unique(y,return_inverse=True);labels=np.zeros(len(vals),np.int64)
    start=vals[0];label=0
    for i,a in enumerate(vals):
        if a-start>tol:label+=1;start=a
        labels[i]=label
    return labels[inv].reshape(y.shape)

def ratio(a,b):return a/b if b else None

def writecsv(path,rows):
    if not rows:return
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def boundary_segments(ru,rv,u,v,L):
    """Same unordered region pair; join dual edges only at degree-two vertices."""
    m=len(u);par=list(range(m));verts={}
    def root(i):
        while par[i]!=i:par[i]=par[par[i]];i=par[i]
        return i
    for i,(a,b) in enumerate(zip(u,v)):
        y,x=divmod(int(a),L)
        ends=((x+1,y),(x+1,y+1)) if b-a==1 else ((x,y+1),(x+1,y+1))
        pair=(min(int(ru[i]),int(rv[i])),max(int(ru[i]),int(rv[i])))
        for pt in ends:verts.setdefault((pair,pt),[]).append(i)
    for ids in verts.values():
        if len(ids)==2:
            a,b=map(root,ids)
            if a!=b:par[b]=a
    groups={}
    for i in range(m):groups.setdefault(root(i),[]).append(i)
    return list(groups.values())

def dilate(mask,radius):
    out=np.zeros_like(mask)
    h,w=mask.shape
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            out[max(0,dy):min(h,h+dy),max(0,dx):min(w,w+dx)] |= mask[max(0,-dy):min(h,h-dy),max(0,-dx):min(w,w-dx)]
    return out

def proximity(V,obs,radius):
    # Pixel-center raster proxies; no claim of exact continuous intersection.
    near_obs=dilate(obs,radius)
    code=np.sum(V.astype(np.uint16)*(1<<np.arange(len(V)))[:,None,None],axis=0)
    a,b,c,d=code[:-1,:-1],code[:-1,1:],code[1:,:-1],code[1:,1:]
    seed=((a!=b)&(a!=c)&(b!=c))|((a!=b)&(a!=d)&(b!=d))|((a!=c)&(a!=d)&(c!=d))|((b!=c)&(b!=d)&(c!=d))
    seed &= ~obs[:-1,:-1]&~obs[:-1,1:]&~obs[1:,:-1]&~obs[1:,1:]
    junction=np.zeros_like(obs)
    junction[:-1,:-1]|=seed;junction[:-1,1:]|=seed;junction[1:,:-1]|=seed;junction[1:,1:]|=seed
    return near_obs.ravel(),dilate(junction,radius).ravel()

def analyze_window(ds,sid,vid,tol,radius):
    scene=ds.get_scene(sid); k=int(ds.driver_counts[sid]);x,y,L=map(int,ds.local_views[sid,vid]); item=ds.get_view(sid,vid)
    obs=item['obstacle'];response=item['response'];a=item['driver_strengths']
    V=unpack_region_bits(ds.visibility_bits[sid,:k],ds.global_size,x,y,L,L)&(a[:,None,None]>0)
    assert np.max(abs(np.sum(V*a[:,None,None],axis=0)-response))==0
    code=np.sum(V.astype(np.uint16)*(1<<np.arange(k))[:,None,None],axis=0).ravel()
    types=scene['obstacle_types'];params=scene['obstacle_params']
    aabbs=np.array([geo.compute_obstacle_aabb(int(t),p) for t,p in zip(types,params)])
    par,u,v,blocked=graph(~obs,response_bins(response,tol),x,y,types,params,aabbs)
    yy=response.ravel();changed=np.abs(yy[u]-yy[v])>tol
    plus=code[v]&~code[u];minus=code[u]&~code[v];mixed=(plus!=0)&(minus!=0)
    hidden=(~changed)&(code[u]!=code[v])
    # Whole-region heterogeneity, including hidden within-bin changes.
    rmin=np.full(L*L,65535,np.int64);rmax=np.zeros(L*L,np.int64);freeids=np.flatnonzero(~obs.ravel())
    np.minimum.at(rmin,par[freeids],code[freeids]);np.maximum.at(rmax,par[freeids],code[freeids]);hetero=rmin!=rmax
    regions=np.unique(par[freeids]);sel=par[u]!=par[v]
    bu,bv=u[sel],v[sel];bm=mixed[sel];bc=changed[sel]
    ru,rv=par[bu],par[bv];pairs={};segments=[];segmentids=[]
    no,nj=proximity(V,obs,radius)
    labels=np.where(no[bu]|no[bv],'obstacle',np.where(nj[bu]|nj[bv],'junction_proxy','ordinary'))
    def unit_stats(ids):
        ids=np.asarray(ids);valid=ids[bc[ids]];n=len(valid);m=int(bm[valid].sum())
        return valid,n,m
    for indices in boundary_segments(ru,rv,bu,bv,L):
        valid,n,m=unit_stats(indices)
        if not n:continue
        first=indices[0];p=tuple(sorted((int(ru[first]),int(rv[first]))));pairkey=f'{sid}:{vid}:{p[0]}:{p[1]}'
        cls='none' if m==0 else ('all' if m==n else 'partial')
        # Orient consistently low response -> high response.
        enter=np.where(yy[bv[valid]]>yy[bu[valid]],code[bv[valid]]&~code[bu[valid]],code[bu[valid]]&~code[bv[valid]])
        leave=np.where(yy[bv[valid]]>yy[bu[valid]],code[bu[valid]]&~code[bv[valid]],code[bv[valid]]&~code[bu[valid]])
        mx=bm[valid];largest=0
        for i in range(k):
            for j in range(k):
                if i!=j:largest=max(largest,int((((enter>>i)&1)&((leave>>j)&1)).sum()))
        row=dict(scene_id=sid,view_id=vid,K=k,size=L,segment_id=len(segments),pair_id=pairkey,
                 length_raw=len(indices),length=n,mixed=m,f=m/n,category=cls,
                 heterogeneous_sides=int(hetero[p[0]] or hetero[p[1]]),
                 persistent_pair_max_edges=largest,persistent_pair_fraction=largest/n,
                 same_pair_all_mixed_edges=int(m>0 and largest==m),same_pair_entire_segment=int(m>0 and largest==n))
        for z in ('ordinary','junction_proxy','obstacle'):
            take=labels[valid]==z;row[z+'_edges']=int(take.sum());row[z+'_mixed']=int(bm[valid][take].sum())
        segments.append(row);segmentids.append(np.asarray(indices))
        pairs.setdefault(p,[]).extend(indices)
    pairrows=[]
    for p,ids in pairs.items():
        valid,n,m=unit_stats(ids);he=bool(hetero[p[0]] or hetero[p[1]])
        ca=int(rmin[p[0]]);cb=int(rmin[p[1]])
        nested=not he and ((ca&cb)==ca or (ca&cb)==cb)
        pairrows.append(dict(scene_id=sid,view_id=vid,K=k,size=L,pair_id=f'{sid}:{vid}:{p[0]}:{p[1]}',
                             edges=n,mixed=m,heterogeneous=int(he),nested=int(nested),nonnested=int(not he and not nested)))
    row=dict(scene_id=sid,view_id=vid,K=k,x=x,y=y,size=L,regions=len(regions),heterogeneous_regions=int(hetero[regions].sum()),
             legal_edges=len(u),cross_obstacle_excluded=int(blocked),changed_edges=int(changed.sum()),mixed_edges=int((changed&mixed).sum()),
             hidden_changed_edges=int(hidden.sum()),hidden_mixed_edges=int((hidden&mixed).sum()),segments=len(segments),pairs=len(pairrows),
             no_boundary=int(not changed.any()),has_mixed=int((changed&mixed).any()),
             has_partial=int(any(z['category']=='partial' for z in segments)),has_all=int(any(z['category']=='all' for z in segments)))
    assert sum(z['length'] for z in segments)==row['changed_edges']
    assert sum(z['mixed'] for z in segments)==row['mixed_edges']
    # Save exact dual-edge membership per segment for reproducibility/visualization.
    payload=dict(u=bu,v=bv,segment_offsets=np.r_[0,np.cumsum([len(z) for z in segmentids])],
                 segment_edge_indices=np.concatenate(segmentids) if segmentids else np.array([],int),
                 changed=bc,mixed=bm,region_u=ru,region_v=rv)
    return row,segments,pairrows,payload

def metrics(windows,segments,pairs):
    den=sum(z['changed_edges'] for z in windows);mix=sum(z['mixed_edges'] for z in windows)
    nw=sum(not z['no_boundary'] for z in windows);nc=len(segments);npair=len(pairs)
    out=dict(scenes=len(set(z['scene_id'] for z in windows)),windows=len(windows),boundary_windows=nw,no_boundary_windows=len(windows)-nw,
             changed_edges=den,mixed_edges=mix,pixel_mixed_rate=ratio(mix,den),pixel_nested_rate=ratio(den-mix,den),segments=nc,pairs=npair,
             window_mixed_rate=ratio(sum(z['has_mixed'] for z in windows),nw),window_partial_rate=ratio(sum(z['has_partial'] for z in windows),nw),
             window_all_rate=ratio(sum(z['has_all'] for z in windows),nw),
             hidden_changed_edges=sum(z['hidden_changed_edges'] for z in windows),hidden_mixed_edges=sum(z['hidden_mixed_edges'] for z in windows),
             excluded_cross_obstacle=sum(z['cross_obstacle_excluded'] for z in windows))
    for label in ('none','partial','all'):
        count=sum(z['category']==label for z in segments);out['segment_'+label+'_count']=count;out['segment_'+label+'_rate']=ratio(count,nc)
    for label in ('nested','nonnested','heterogeneous'):
        count=sum(z[label] for z in pairs);out['pair_'+label+'_count']=count;out['pair_'+label+'_rate']=ratio(count,npair)
    out['f_quantiles']=dict(zip(['0','.25','.5','.75','.9','.95','.99','1'],np.quantile([z['f'] for z in segments],[0,.25,.5,.75,.9,.95,.99,1]).tolist())) if segments else {}
    out['mixed_segments_same_pair_throughout']=sum(z['same_pair_entire_segment'] for z in segments)
    out['mixed_segments_same_pair_on_all_mixed_edges']=sum(z['same_pair_all_mixed_edges'] for z in segments)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=ROOT.parent/'Phase2_Multi_Source/data/raw/phase2-multi-source');ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--per-k',type=int,default=25);ap.add_argument('--seed',type=int,default=20261010);ap.add_argument('--tol',type=float,default=1e-12)
    ap.add_argument('--radius',type=int,default=2);ap.add_argument('--bootstrap',type=int,default=2000);ap.add_argument('--sensitivity',action='store_true')
    args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True);(out/'edges').mkdir(exist_ok=True)
    ds=RINDDataset(args.data);rng=np.random.default_rng(args.seed);selected=[]
    for k in np.unique(ds.driver_counts):
        ids=np.flatnonzero(ds.driver_counts==k);selected.extend(rng.choice(ids,min(args.per_k,len(ids)),replace=False).tolist())
    selected=sorted(selected);manifest=dict(vars(args));manifest['data']=str(args.data.resolve());manifest['out']=str(out.resolve());manifest['scene_ids']=selected
    manifest['dataset_manifest']=ds.manifest;manifest['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['source_configurations']=[dict(scene_id=s,positions=ds.get_scene(s)['drivers'].tolist(),strengths=ds.get_scene(s)['driver_strengths'].tolist()) for s in selected]
    (out/'sample.json').write_text(json.dumps(manifest,indent=2))
    windows=[];segments=[];pairs=[];scene_stats=[];sensitivity=[];maxerr=0;timings=[];started=time.perf_counter()
    for seq,sid in enumerate(selected):
        t=time.perf_counter(); ds.get_view_tree(sid) # validate actual disjoint leaf policy
        sw=[];ss=[];sp=[]
        for vid in range(int(ds.view_counts[sid])):
            w,se,pa,payload=analyze_window(ds,sid,vid,args.tol,args.radius)
            sw.append(w);ss.extend(se);sp.extend(pa)
            if se:np.savez_compressed(out/'edges'/f'{sid}_{vid}.npz',**payload)
            if sid==min(z for z in selected if ds.driver_counts[z]==ds.driver_counts[sid]) and vid==0:
                item=ds.get_view(sid,vid);sc=ds.get_scene(sid);x,y,L=map(int,item['view'])
                fresh=render_scene_region(ds,sid,x,y,L,np.column_stack([sc['drivers'],sc['driver_strengths']]),backend='numba')
                err=float(np.max(abs(fresh['response']-item['response'])));maxerr=max(maxerr,err)
                assert np.array_equal(fresh['channels']>0,item['channels']>0)
                assert np.array_equal(fresh['obstacle'],item['obstacle'])
            if args.sensitivity and sid==min(z for z in selected if ds.driver_counts[z]==ds.driver_counts[sid]):
                for tol in (0.,1e-14,1e-10):
                    ww,sss,ppp,_=analyze_window(ds,sid,vid,tol,args.radius)
                    sensitivity.append(dict(scene_id=sid,view_id=vid,tol=tol,changed_edges=ww['changed_edges'],mixed_edges=ww['mixed_edges'],segments=len(sss),pairs=len(ppp),heterogeneous_regions=ww['heterogeneous_regions']))
        windows.extend(sw);segments.extend(ss);pairs.extend(sp)
        sm=metrics(sw,ss,sp);sm.update(scene_id=sid,K=int(ds.driver_counts[sid]));scene_stats.append(sm)
        timings.append(dict(scene_id=sid,seconds=time.perf_counter()-t));print(f'{seq+1}/{len(selected)} scene={sid} windows={len(sw)} edges={sm["changed_edges"]} mixed={sm["mixed_edges"]} seconds={timings[-1]["seconds"]:.2f}',flush=True)
    writecsv(out/'windows.csv',windows);writecsv(out/'segments.csv',segments);writecsv(out/'region_pairs.csv',pairs);writecsv(out/'timing.csv',timings);writecsv(out/'sensitivity.csv',sensitivity)
    (out/'scenes.json').write_text(json.dumps(scene_stats,indent=2))
    overall=metrics(windows,segments,pairs)
    strata={}
    for key in ('K','size'):
        strata[key]={str(val):metrics([z for z in windows if z[key]==val],[z for z in segments if z[key]==val],[z for z in pairs if z[key]==val]) for val in sorted(set(z[key] for z in windows))}
    strata['length']={}
    for lo,hi in ((1,1),(2,4),(5,16),(17,64),(65,256),(257,10**9)):
        rows=[z for z in segments if lo<=z['length']<=hi];n=sum(z['length'] for z in rows);m=sum(z['mixed'] for z in rows)
        strata['length'][f'{lo}-{hi}']=dict(segments=len(rows),edges=n,mixed=m,pixel_mixed_rate=ratio(m,n),**{c:sum(z['category']==c for z in rows) for c in ('none','partial','all')})
    strata['proximity']={}
    for c in ('ordinary','junction_proxy','obstacle'):
        n=sum(z[c+'_edges'] for z in segments);m=sum(z[c+'_mixed'] for z in segments)
        strata['proximity'][c]=dict(edges=n,mixed=m,rate=ratio(m,n),coverage=ratio(n,overall['changed_edges']))
    # Scene-cluster stratified bootstrap, preserve selected numbers per K.
    keys=['pixel_mixed_rate','segment_none_rate','segment_partial_rate','segment_all_rate','pair_nested_rate','pair_heterogeneous_rate','window_mixed_rate']
    sums=np.array([[s['mixed_edges'],s['changed_edges'],s['segment_none_count'],s['segments'],s['segment_partial_count'],s['segment_all_count'],s['pair_nested_count'],s['pairs'],s['pair_heterogeneous_count'],sum(w['has_mixed'] for w in windows if w['scene_id']==s['scene_id']),s['boundary_windows']] for s in scene_stats],float)
    def boot(idx):
        groups=[np.array([i for i in idx if scene_stats[i]['K']==k]) for k in range(1,5)]; vals=[]
        for _ in range(args.bootstrap):
            draw=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups if len(g)]);a=sums[draw].sum(0)
            vals.append([ratio(a[0],a[1]),ratio(a[2],a[3]),ratio(a[4],a[3]),ratio(a[5],a[3]),ratio(a[6],a[7]),ratio(a[8],a[7]),ratio(a[9],a[10])])
        ar=np.array(vals,float)
        return {k:np.nanquantile(ar[:,i],[.025,.975]).tolist() if np.isfinite(ar[:,i]).any() else None for i,k in enumerate(keys)}
    ci={'overall':boot(list(range(len(scene_stats))))}
    for k in range(1,5):ci[str(k)]=boot([i for i,s in enumerate(scene_stats) if s['K']==k])
    result=dict(overall=overall,strata=strata,cluster_bootstrap_95=ci,max_rerender_error=maxerr,
                elapsed_seconds=time.perf_counter()-started,scene_count=len(selected),source_config_count=len(selected),
                physical_edge_dedup='Each scene tree validated; source-free leaves have disjoint interiors. No within-scene physical pixel edge repeats across windows.',
                weighting='equal sampled scene counts per K; pooled counts describe this stratified sample, not exact release-population weighting')
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(overall,indent=2),flush=True)
if __name__=='__main__':main()
