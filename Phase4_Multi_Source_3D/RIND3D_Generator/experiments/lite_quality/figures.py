"""Standalone scientific plots from recorded measurements, never extrapolated fields."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('outputs/lite-v02-research')

DEST=Path('docs/lite-quality-assets')
def read(p):return json.loads(Path(p).read_text())
def save(fig,name):fig.savefig(DEST/name,dpi=170,bbox_inches='tight');plt.close(fig)

def run():
    DEST.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    natural=read(OUT/'frozen-distribution-audit.json');uniform=read(OUT/'sampling/summary.json')['stratified'];mixed=read(OUT/'sampling-mixed-current/summary.json')['mixed']
    fig,ax=plt.subplots(figsize=(11,4));x=np.arange(10)
    for k,(label,data,color) in enumerate([('Natural',natural,'#3c6382'),('Stratified',uniform,'#38916a'),('30/50/20 mixture',mixed,'#e0a135')]):
        values=np.array(data['bins']);ax.bar(x+(k-1)*.25,values/values.sum()*100,.24,label=label,color=color)
    ax.set(xticks=x,xticklabels=[f'{i*10}-{i*10+10}' for i in range(10)],xlabel='Lit / free voxels (%)',ylabel='Windows (%)',title='Same 100 frozen geometries and sources; 25,600 windows per selection')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);save(fig,'lit-ratio-distribution.png')
    families=list(natural['by_family']);heat=np.array([natural['by_family'][f] for f in families],float);heat=heat/heat.sum(axis=1)[:,None]*100
    fig,ax=plt.subplots(figsize=(10,4.7));im=ax.imshow(heat,cmap='YlGnBu',aspect='auto',vmin=0,vmax=60);ax.set(yticks=np.arange(len(families)),yticklabels=families,xticks=x,xticklabels=[f'{i*10}-{i*10+10}' for i in range(10)],xlabel='Lit / free voxels (%)',title='Frozen natural selection by geometry family')
    for i in range(len(families)):
        for j in range(10):ax.text(j,i,f'{heat[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if heat[i,j]>35 else '#16324a')
    fig.colorbar(im,ax=ax,label='Within-family windows (%)');save(fig,'family-lit-ratios.png')
    rows=read(OUT/'verification-v3/summary.json')['scenes'];fig,axes=plt.subplots(1,2,figsize=(11,4))
    before=np.array([r['baseline_frozen_stages'][1]['cpu_seconds'] for r in rows]);after=np.array([r['B2']['cpu_seconds']+r['B2_setup_seconds'] for r in rows])
    axes[0].scatter(before,after,c='#38916a',s=34);axes[0].plot([0,before.max()],[0,before.max()],':',color='gray');axes[0].set(xlabel='Original B CPU seconds',ylabel='New B CPU seconds incl. setup',title='20 slow frozen scenes: same physical bricks')
    full=np.array([sum(r['planes']['original_fme_planes']) for r in rows]);reduced=np.array([sum(r['planes']['retained_planes']) for r in rows]);axes[1].scatter(full,reduced,c='#3c6382',s=34);axes[1].plot([0,full.max()],[0,full.max()],':',color='gray');axes[1].set(xlabel='Full FME constraints',ylabel='Certified silhouette constraints',title='Exact redundancy reduction')
    fig.tight_layout();save(fig,'independent-verifier.png')
    rows=read(OUT/'ambiguity-transition/summary.json')['records'];fig,ax=plt.subplots(figsize=(11,4));x=np.arange(len(rows))
    for k,(label,color) in enumerate([('Spatial observations','#3c6382'),('Exact adjacent transitions','#e0a135')]):
        values=[r['variants'][k]['outer']['outer_cover_volume_upper_bound'] for r in rows];ax.scatter(x+(k-.5)*.2,values,s=48,color=color,label=label,zorder=3)
    ax.set(yscale='log',xticks=x,xticklabels=[r['window']['family'] for r in rows],ylabel='Certified outer volume bound',title='64 observations, bounded subdivision: outer covers, not feasible volumes')
    ax.tick_params(axis='x',rotation=25);ax.legend(frameon=False);save(fig,'inverse-outer-volumes.png')
    cert=read(OUT/'inverse-outer-certificates/summary.json')['cases'];from fractions import Fraction
    radii=[np.sqrt(float(Fraction(c['maximum_truth_distance_squared_upper_exact']))) for c in cert]
    fig,ax=plt.subplots(figsize=(7,3.8));ax.bar([c['family'] for c in cert],radii,color='#38916a');ax.axhline(1/32,c='#bf684d',linestyle='--',label='Required reference separation: 32/1024');ax.set(ylabel='Distance from truth: certified upper bound',title='Independent full-FME + complete partition recheck');ax.legend(frameon=False,fontsize=9);save(fig,'separation-certificates.png')
    comparison=next((OUT/name/'summary.json' for name in ('inverse-five-method-cold-v2','inverse-five-method-cold') if (OUT/name/'summary.json').exists()),OUT/'inverse-five-method-cold/summary.json')
    if comparison.exists():
        data=read(comparison)['methods'];methods=('random','local','silhouette','inverse','hybrid');fig,ax=plt.subplots(figsize=(8,4))
        ax.bar(methods,[data[m]['references_per_total_wall_second'] for m in methods],color=['#3c6382','#638eaa','#a7b1a3','#38916a','#e0a135'])
        ax.set(ylabel='Verified extra window references\nper total wall second',title='Nine matched 64-cube windows; method-local caches reset')
        ax.text(.02,.98,'Includes geometry / constraints / screening / complete A+B verification\nSmall sample; source count never implies uniqueness',transform=ax.transAxes,va='top',fontsize=8)
        ax.margins(y=.28);save(fig,'inverse-search-yield.png')
    else:
        methods=('random','local','inverse','hybrid');results=[]
        for file in ('inverse-v2/summary.json','inverse-v4-cold/summary.json'):
            data=read(OUT/file)['records'];values=[]
            for method in methods:
                tests=[t for r in data for t in r['tests'] if t['budget']==2 and t['method']==method];yield_count=sum(len(t['positions'])-1 for t in tests);seconds=sum(t['report']['elapsed_seconds'] for t in tests);values.append(yield_count/seconds)
            results.append(values)
        fig,ax=plt.subplots(figsize=(8,4));x=np.arange(4)
        for k,(label,color) in enumerate([('Spatial observations','#3c6382'),('Exact transitions, cold decoded/model caches','#e0a135')]):ax.bar(x+(k-.5)*.3,results[k],.29,label=label,color=color)
        ax.set(xticks=x,xticklabels=methods,ylabel='Verified additional window references / wall second',title='Nine frozen 64-cube windows; 2 s nominal budget incl. preparation');ax.legend(frameon=False,fontsize=8);save(fig,'inverse-search-yield.png')
    studies=[]
    for name in ('core-sequential-10','core-pipeline-1-10','core-smoke-10','core-pipeline-4-10','core-fast-natural-10','core-fast-mixed-10','core-matched-100','core-fast-mixed-100','core-final-100','core-fast-mixed-300'):
        path=OUT/name/'report.json'
        if path.exists():
            data=read(path)
            if data['complete']:studies.append((name,data))
    if studies:
        fig,ax=plt.subplots(figsize=(11,4.5));ax.barh([n.replace('core-','') for n,_ in studies],[3600*r['ready_scenes']/r['wall_seconds_this_run'] for _,r in studies],color=['#3c6382' if 'fast' not in n else '#38916a' for n,_ in studies]);ax.axvline(3600/72.113,c='#bf684d',linestyle='--',label='Frozen original 100-scene throughput');ax.set(xlabel='Complete READY_LITE scenes / hour',title='Whole-pilot time; study sizes and changed fast profiles are named');ax.legend(frameon=False,fontsize=8);save(fig,'ready-throughput.png')
    path=OUT/'extended-results.json'
    if path.exists():
        data=read(path)['pilots'];run=data.get('core-fast-mixed-300')
        if run:
            batches=[b for b in run['ready_batches'] if b['end_ready']-b['start_ready']==49]
            fig,ax=plt.subplots(figsize=(9,4));ax.plot([b['end_ready'] for b in batches],[b['ready_per_hour'] for b in batches],'o-',color='#38916a')
            ax.axhline(120,c='#bf684d',linestyle='--',label='Engineering target: 1 READY / 30 s')
            ax.set(xlabel='Completed READY_LITE scenes (successive groups of 50)',ylabel='READY_LITE / hour',title='Bounded 300-scene run, including cold start and slow jobs');ax.legend(frameon=False);save(fig,'sustained-throughput.png')

if __name__=='__main__':run()
