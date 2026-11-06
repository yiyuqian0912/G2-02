"""Static scientific plots from complete measured fields, never interpolated truth."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator, NullFormatter
import numpy as np

from .fixtures import SOURCE,RECEIVER,INNER_LO,INNER_HI,union_scene


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    records=[]
    for root in args.runs:
        for path in sorted(root.glob('gaussian*.json')):
            row=json.loads(path.read_text())
            if row.get('complete') and 'method' in row:records.append((row,path))
    methods=['A_fixed6','A_fixed7','B_spatial','C_angular','D_union','E_hybrid']
    names=['Fixed 6','Fixed 7','Spatial','Angular 1','Angular union','Hybrid']
    cohorts=[]
    for root in args.runs:
        cases={r['case'] for r,path in records if path.parent==root and r['resolution']==64 and r['method']=='A_fixed6'}
        if len(cases)>=4:cohorts.append(root)
    if not cohorts:raise ValueError('Policy figure needs a common four-case 64³ sweep')
    cohort=cohorts[0]
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for ax,backend in zip(axes,('primary','independent')):
        values=[]
        for method in methods:
            candidates=[r for r,path in records if path.parent==cohort and r['backend']==backend and r['method']==method and r['resolution']==64]
            # Prefer the complete common depth-6 sweep, avoiding optional repeats.
            unique={r['case']:r for r in candidates if r.get('effective_settings',{}).get('cover_depth',6)==6 or method=='A_fixed7'}
            values.append(sum(r['cold_total_seconds'] for r in unique.values()))
        bars=ax.bar(names,values,color=['#9aa9ba','#657791','#ad8bbd','#66a79b','#3c8379','#d59b57'])
        ax.bar_label(bars,fmt='%.1f',padding=3);ax.set_ylim(0,max(values)*1.18);ax.tick_params(axis='x',rotation=25)
        ax.set_ylabel('Sum of four cold complete-field times (s)');ax.set_title(f'{backend.capitalize()} — four frozen 64³ fields')
    fig.suptitle('All setup and failed certificate attempts included; one cold trial per configuration')
    fig.savefig(args.output/'angular-policy-ablation.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    ax=axes[0];scene=union_scene();x=np.linspace(.1,.96,350);y=np.linspace(.34,.66,200);xx,yy=np.meshgrid(x,y)
    points=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,.5)]
    value=scene.approx_values(points).reshape(xx.shape)
    ax.contourf(xx,yy,value,levels=[value.min(),0,value.max()],colors=['#f6f6f6','#cbd2da'])
    for a,b,color in zip(INNER_LO,INNER_HI,('#4c87b6','#df9850')):
        ax.add_patch(Rectangle((a[0],a[1]),b[0]-a[0],b[1]-a[1],facecolor=color,edgecolor='black',alpha=.8))
    for endpoint in ([.875,.375],[.875,.625]):ax.plot([SOURCE[0],endpoint[0]],[SOURCE[1],endpoint[1]],color='#777',lw=.8)
    ax.scatter([SOURCE[0]],[SOURCE[1]],c='#d43b47',s=35,label='Source')
    ax.add_patch(Rectangle((RECEIVER[0][0],RECEIVER[0][1]),RECEIVER[1][0]-RECEIVER[0][0],RECEIVER[1][1]-RECEIVER[0][1],fill=False,lw=2,color='#378572'))
    ax.set(xlim=(.1,.96),ylim=(.34,.66),xlabel='World x',ylabel='World y',title='Continuous inner boxes at different depths')
    ax.text(.12,.345,'Gray Gaussian slice: illustration only; boxes have continuous proofs',fontsize=8)
    ax=axes[1]
    for u0,u1,depth,color in [(-1/6,0,'ρ = 1/4','#4c87b6'),(0,1/6,'ρ = 7/16','#df9850')]:
        ax.add_patch(Rectangle((u0,-1/24),u1-u0,1/12,facecolor=color,edgecolor='black'))
        ax.text((u0+u1)/2,0,depth,ha='center',va='center',fontsize=12)
    ax.set(xlim=(-.19,.19),ylim=(-.06,.06),xlabel='Chart u',ylabel='Chart v',title='Exact affine-extrema union proof, not angular samples')
    ax.text(0,-.054,'Shared closed edge u=0; no single blocker covers the rectangle',ha='center',fontsize=8)
    fig.savefig(args.output/'continuous-union-proof.png',dpi=160);plt.close(fig)

    # Homogeneous depth-7 scaling only; never join depth-6 and depth-7 timings.
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    styles={'A_fixed7':('#798694','--'),'D_union':('#2b8879','-'),'R_point_cover':('#be804a',':')}
    for ax,case in zip(axes,('gaussian_sum-014','gaussian_sum-007')):
        for method,(color,style) in styles.items():
            for backend,marker in [('primary','o'),('independent','s')]:
                points={}
                for row,_ in records:
                    # Local 1024-grid ROIs are not complete whole-world fields.
                    if (row.get('world_resolution',row['resolution']) != row['resolution']
                            or any(row.get('origin_xyz',[0,0,0]))):
                        continue
                    depth=row.get('effective_settings',{}).get('cover_depth',7 if method=='A_fixed7' else 6)
                    if row['case']==case and row['backend']==backend and row['method']==method and depth==7:points[row['resolution']]=row['cold_total_seconds']
                if points:
                    sizes=sorted(points);ax.plot(sizes,[points[n] for n in sizes],marker=marker,ls=style,color=color,label=f'{method[2:]} / {backend}')
        ax.set_xscale('log',base=2)
        ax.set(yscale='log',xlabel='Complete whole-world sampling side N',ylabel='Cold field time (s)',title=f'{case} — same proof depth 7')
        ax.xaxis.set_minor_locator(NullLocator());ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([32,64,128],[32,64,128]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Finite-size measurements, not an asymptotic scaling proof')
    fig.savefig(args.output/'angular-scaling.png',dpi=160);plt.close(fig)

    choices=[(r,p) for r,p in records if r['case']=='gaussian_sum-014' and r['resolution']==128 and r['method']=='D_union' and r['backend']=='independent']
    if choices:
        row,path=choices[-1];a=np.load(path.with_suffix('.npz'));middle=64
        fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
        image=np.where(a['solid'][middle],2,np.where(a['response'][middle],0,1))
        axes[0].imshow(image,origin='lower',interpolation='nearest',cmap=ListedColormap(['#fafafa','#526b90','#171e28']),vmin=0,vmax=2)
        axes[0].set_title('Actual complete binary field, z=64\nWhite lit / blue shadow / black solid')
        t=a['ray_nanoseconds'][middle]/1e9;masked=np.ma.masked_where(t<=0,np.log10(np.maximum(t,1e-12)))
        cmap=plt.get_cmap('magma').copy();cmap.set_bad('#eee')
        im=axes[1].imshow(masked,origin='lower',interpolation='nearest',cmap=cmap,vmin=-5,vmax=-1)
        axes[1].set_title('Actual remaining scalar-ray time\nGray means no scalar call');fig.colorbar(im,ax=axes[1],label='log10(seconds)')
        tiles=np.asarray(a['angular_tiles']);finite=np.isfinite(tiles[:,5]);counts=np.bincount(tiles[finite,0].astype(int),minlength=6);face=int(np.argmax(counts))
        tiles=tiles[tiles[:,0]==face];patches=[Rectangle((q[1],q[3]),q[2]-q[1],q[4]-q[3]) for q in tiles]
        collection=PatchCollection(patches,cmap='viridis',edgecolor='none');collection.set_array(np.ma.masked_invalid(np.where(np.isfinite(tiles[:,5]),tiles[:,5],np.nan)));collection.cmap.set_bad('#ddd')
        axes[2].add_collection(collection);axes[2].set(xlim=(-1,1),ylim=(-1,1),xlabel='u',ylabel='v')
        axes[2].set_title(f'Angular leaf upper-depth proofs, chart {face}\nGray: no leaf-only bound; not a sampled T map')
        fig.colorbar(collection,ax=axes[2],label='Certified first-hit upper depth ρ')
        for ax in axes[:2]:ax.set(xlabel='Voxel x',ylabel='Voxel y')
        fig.savefig(args.output/'angular-residual-and-depth-map.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
