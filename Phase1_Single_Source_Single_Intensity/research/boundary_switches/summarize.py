"""Scene-cluster CIs for every reported stratum; representative real-data figures."""
from analyze import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle,Ellipse,Polygon

def readcsv(path):
    rows=[]
    with open(path) as f:
        for r in csv.DictReader(f):
            for k,v in r.items():
                try:r[k]=float(v) if '.' in v or 'e-' in v else int(v)
                except ValueError:pass
            rows.append(r)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();run=args.run
    meta=json.loads((run/'sample.json').read_text());ws=readcsv(run/'windows.csv');ss=readcsv(run/'segments.csv');ps=readcsv(run/'region_pairs.csv')
    sid=meta['scene_ids'];idx={s:i for i,s in enumerate(sid)};N=len(sid);K={z['scene_id']:z['K'] for z in ws}
    rng=np.random.default_rng(meta['seed']+1);B=2000
    counts=np.zeros((B,N))
    for k in range(1,5):
        ids=np.array([i for i,s in enumerate(sid) if K[s]==k]);draw=rng.choice(ids,(B,len(ids)),replace=True)
        for b in range(B):counts[b]+=np.bincount(draw[b],minlength=N)
    rows=[]
    def record(group,level,metric,items,num,den):
        an=np.zeros(N);ad=np.zeros(N)
        for r in items:an[idx[r['scene_id']]]+=num(r);ad[idx[r['scene_id']]]+=den(r)
        totals=counts@np.stack([an,ad],axis=1);valid=totals[:,1]>0;boot=totals[valid,0]/totals[valid,1]
        low,high=np.quantile(boot,[.025,.975]) if len(boot) else (None,None)
        rows.append(dict(group=group,level=level,metric=metric,numerator=int(an.sum()),denominator=int(ad.sum()),rate=ratio(an.sum(),ad.sum()),ci_low=low,ci_high=high,contributing_scenes=int((ad>0).sum()),zero_event_bootstrap_degenerate=int(an.sum()==0 and ad.sum()>0)))
    def group_metrics(group,level,w,s,p):
        record(group,level,'pixel_mixed',w,lambda z:z['mixed_edges'],lambda z:z['changed_edges'])
        for cat in ('none','partial','all'):record(group,level,'segment_'+cat,s,lambda z:int(z['category']==cat),lambda z:1)
        for cat in ('nested','nonnested','heterogeneous'):record(group,level,'pair_'+cat,p,lambda z:z[cat],lambda z:1)
        record(group,level,'pair_nested_given_homogeneous',p,lambda z:z['nested'],lambda z:1-z['heterogeneous'])
        for cat in ('mixed','partial','all'):record(group,level,'window_'+cat,w,lambda z:z['has_'+cat],lambda z:1-z['no_boundary'])
    group_metrics('overall','all',ws,ss,ps)
    for field in ('K','size'):
        for value in sorted(set(z[field] for z in ws)):
            group_metrics(field,value,[z for z in ws if z[field]==value],[z for z in ss if z[field]==value],[z for z in ps if z[field]==value])
    for lo,hi in ((1,1),(2,4),(5,16),(17,64),(65,256),(257,10**9),(17,10**9)):
        s=[z for z in ss if lo<=z['length']<=hi];key=f'{lo}-{hi}'
        record('length',key,'pixel_mixed',s,lambda z:z['mixed'],lambda z:z['length'])
        for cat in ('none','partial','all'):record('length',key,'segment_'+cat,s,lambda z:int(z['category']==cat),lambda z:1)
        record('length',key,'segment_coverage',ss,lambda z:int(lo<=z['length']<=hi),lambda z:1)
        record('length',key,'edge_coverage',ss,lambda z:z['length']*int(lo<=z['length']<=hi),lambda z:z['length'])
    for cat in ('ordinary','junction_proxy','obstacle'):
        record('proximity',cat,'pixel_mixed',ss,lambda z:z[cat+'_mixed'],lambda z:z[cat+'_edges'])
        record('proximity',cat,'edge_coverage',ss,lambda z:z[cat+'_edges'],lambda z:z['length'])
    writecsv(run/'strata_ci.csv',rows)
    summary=json.loads((run/'summary.json').read_text())
    summary['cluster_bootstrap_95']={}
    names={'pixel_mixed':'pixel_mixed_rate','segment_none':'segment_none_rate','segment_partial':'segment_partial_rate','segment_all':'segment_all_rate','pair_nested':'pair_nested_rate','pair_heterogeneous':'pair_heterogeneous_rate','window_mixed':'window_mixed_rate'}
    for r in rows:
        if r['group'] in ('overall','K') and r['metric'] in names:
            key='overall' if r['group']=='overall' else str(r['level'])
            summary['cluster_bootstrap_95'].setdefault(key,{})[names[r['metric']]]=[r['ci_low'],r['ci_high']]
    summary['final_bootstrap']={'seed':meta['seed']+1,'replicates':B,'method':'within-K scene-cluster bootstrap; no finite-population correction','postprocess_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (run/'summary.json').write_text(json.dumps(summary,indent=2))
    # Reweight sampled scenes to actual release K proportions; a ratio estimator.
    pop=meta['dataset_manifest']['num_scenes'];ds=RINDDataset(Path(meta['data']));weights={k:float((ds.driver_counts==k).sum())/sum(K[s]==k for s in sid) for k in range(1,5)}
    weighted={}
    for name,items,num,den in [('pixel_mixed',ws,lambda z:z['mixed_edges'],lambda z:z['changed_edges']),('segment_all',ss,lambda z:z['category']=='all',lambda z:1),('pair_nested',ps,lambda z:z['nested'],lambda z:1),('window_mixed',ws,lambda z:z['has_mixed'],lambda z:1-z['no_boundary'])]:
        weighted[name]=sum(weights[z['K']]*num(z) for z in items)/sum(weights[z['K']]*den(z) for z in items)
    (run/'release_weighted_estimates.json').write_text(json.dumps(weighted,indent=2))
    # Plot longest and short complete-mixed segments plus nonmixed counterparts.
    figdir=run/'figures';figdir.mkdir(exist_ok=True)
    allmix=sorted([z for z in ss if z['category']=='all'],key=lambda z:z['length'],reverse=True)
    choices=[allmix[0],next(z for z in allmix if z['length']==1),next(z for z in allmix if z['K']==2)]
    normal=[z for z in ss if z['category']=='none' and z['K']>=3 and z['length']>=17]
    choices.extend([normal[0],max(normal,key=lambda z:z['length'])]);choices.extend([z for z in ss if z['category']=='partial'][:2]);figure_records=[]
    for i,r in enumerate(choices):
        sid0,vid=int(r['scene_id']),int(r['view_id']);item=ds.get_view(sid0,vid);sc=ds.get_scene(sid0);x,y,L=map(int,item['view']);k=len(sc['drivers'])
        cache=np.load(run/'edges'/f'{sid0}_{vid}.npz');off=cache['segment_offsets'];j=int(r['segment_id']);which=cache['segment_edge_indices'][off[j]:off[j+1]]
        u,v=cache['u'][which],cache['v'][which];lines=[]
        for aa,bb in zip(u,v):
            yy,xx=divmod(int(aa),L)
            lines.append([(x+xx+1,y+yy),(x+xx+1,y+yy+1)] if bb-aa==1 else [(x+xx,y+yy+1),(x+xx+1,y+yy+1)])
        q=int(np.flatnonzero(cache['changed'][which])[0]);p1,p2=int(u[q]),int(v[q]);resp=item['response'].ravel()
        low,high=(p1,p2) if resp[p1]<resp[p2] else (p2,p1)
        masks=item['channels']>0;lo=[j+1 for j in range(k) if masks.reshape(k,-1)[j,low]];hi=[j+1 for j in range(k) if masks.reshape(k,-1)[j,high]]
        entering=sorted(set(hi)-set(lo));exiting=sorted(set(lo)-set(hi))
        fig,axs=plt.subplots(2,4,figsize=(18,9),layout='constrained');axs=axs.ravel();ax=axs[0]
        for t,pa in zip(sc['obstacle_types'],sc['obstacle_params']):
            if t==0:
                cx,cy,hw,hh,theta=pa[:5];corners=np.array([[-hw,-hh],[hw,-hh],[hw,hh],[-hw,hh]]);rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);patch=Polygon(corners@rot.T+[cx,cy],color='.65')
            elif t==1:patch=Ellipse(pa[:2],2*pa[2],2*pa[3],angle=np.degrees(pa[4]),color='.65')
            elif t==2:patch=Polygon(pa[:6].reshape(3,2),color='.65')
            else:patch=Polygon(pa[1:1+2*int(pa[0])].reshape(-1,2),color='.65')
            ax.add_patch(patch)
        ax.add_patch(Rectangle((x,y),L,L,fill=False,edgecolor='magenta',linewidth=2))
        for j,((sx,sy),a) in enumerate(zip(sc['drivers'],sc['driver_strengths'])):
            ax.scatter([sx],[sy],c='red',s=30);ax.annotate(f'S{j+1}: {a:.4f}',(sx,sy),fontsize=8)
        ax.set(xlim=(0,1024),ylim=(1024,0),aspect='equal',title='Full scene: sources and observation window')
        extent=(x,x+L,y+L,y)
        gray=plt.get_cmap('gray').copy();gray.set_bad('.65')
        viridis=plt.get_cmap('viridis').copy();viridis.set_bad('.65')
        pts=np.array(lines).reshape(-1,2);xmin=max(x,float(pts[:,0].min())-4);xmax=min(x+L,float(pts[:,0].max())+4);ymin=max(y,float(pts[:,1].min())-4);ymax=min(y+L,float(pts[:,1].max())+4)
        im=axs[1].imshow(np.ma.array(item['response'],mask=item['obstacle']),extent=extent,cmap=viridis,interpolation='nearest');fig.colorbar(im,ax=axs[1],shrink=.8)
        axs[1].set_title('Total additive response; highlighted shared segment')
        axs[2].imshow(np.ma.array(item['response'],mask=item['obstacle']),extent=extent,cmap=viridis,interpolation='nearest');axs[2].set_title('Response detail; cyan=low, orange=high')
        axs[1].add_patch(Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,fill=False,edgecolor='red',linewidth=1.5))
        for j in range(k):
            axs[j+3].imshow(np.ma.array(masks[j],mask=item['obstacle']),extent=extent,cmap=gray,vmin=0,vmax=1,interpolation='nearest');axs[j+3].set_title(f'S{j+1} visibility (white=visible), a={sc["driver_strengths"][j]:.5f}')
        for ax in axs[2:k+3]:
            ax.set_xlim(xmin,xmax);ax.set_ylim(ymax,ymin)
            ax.add_collection(LineCollection(lines,colors='red',linewidths=2))
            for p0,c,label in ((low,'cyan','low'),(high,'orange','high')):
                yy,xx=divmod(p0,L);ax.scatter([x+xx+.5],[y+yy+.5],color=c,s=25)
        for ax in axs[k+3:]:ax.axis('off')
        fig.suptitle(f'Scene {sid0}, view {vid}, K={k}, size={L}; {r["category"]}, length={r["length"]}, f={r["f"]:.0f}\nLow visible={lo}; high visible={hi}; ENTER={entering}; EXIT={exiting}',fontsize=13)
        path=figdir/f'{i+1}_{r["category"]}_scene{sid0}_view{vid}.png';fig.savefig(path,dpi=150);plt.close(fig)
        figure_records.append({**r,'image':str(path.relative_to(run)),'low_visible':lo,'high_visible':hi,'enter':entering,'exit':exiting})
    (figdir/'index.json').write_text(json.dumps(figure_records,indent=2))
    if not any(z['category']=='partial' for z in ss):
        fig,ax=plt.subplots(figsize=(9,3));ax.axis('off')
        ax.text(.5,.5,f'No partially mixed segments observed\n0 < f < 1: 0 / {len(ss):,} segments\nNo synthetic example substituted for this dataset result.',ha='center',va='center',fontsize=14)
        fig.savefig(figdir/'partial_not_observed.png',dpi=150,bbox_inches='tight');plt.close(fig)
    print(json.dumps({'weighted_estimates':weighted,'strata_rows':len(rows),'figures':len(figure_records)},indent=2))
if __name__=='__main__':main()
