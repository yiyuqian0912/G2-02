"""Measured scaling and work figures; no inferred continuous feasible geometry."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads(a.summary.read_text());a.output.mkdir(parents=True,exist_ok=True)
    rows=m['runs']['ablation32-v2']['results'];modes=['A_scalar','B_old_regions','C_ray_volume','D_fused_shadow','E_combined','F_proof_reuse']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,backend in zip(axes,('primary','independent')):
        values=[sum(r['cold_total_seconds'] for r in rows if r['resolution']==32 and r['backend']==backend and r['mode']==mode) for mode in modes]
        ax.bar(range(6),values,color=['#a8adb4','#a8adb4','#489b9d','#c08c5b','#489b9d','#3d7e86'])
        ax.set_xticks(range(6),['Scalar','Old regions','Ray volume','Fused section','Combined','Proof reuse'],rotation=20,ha='right')
        ax.set(title=backend,ylabel='Cold wall seconds (sum of four 32³ fields)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same frozen B2 fields; setup and lossless diagnostic writing included')
    fig.subplots_adjust(bottom=.25,top=.82,left=.07,right=.99,wspace=.3);fig.savefig(a.output/'region-policy-ablation.png',dpi=155);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for name,label,color in [('case14-128-v3','cover depth 6','#438e96'),('depth7-case14-v1','cover depth 7','#b47e43')]:
        rr=sorted((r for r in m['runs'][name]['results'] if r['backend']=='independent' and r['complete']),key=lambda r:r['resolution'])
        axes[0].plot([r['resolution'] for r in rr],[r['cold_total_seconds'] for r in rr],'o-',label=label,color=color)
        axes[1].plot([r['resolution'] for r in rr],[r['derived']['scalar_ray_queries'] for r in rr],'o-',label=label,color=color)
    axes[0].scatter([128],[480],marker='x',c='#bd4653',s=80,label='old: incomplete at 480s')
    for ax in axes:
        ax.set(xscale='log',yscale='log',xlabel='World sampling side N');ax.set_xticks([8,32,64,128],['8','32','64','128']);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Independent cold wall seconds');axes[1].set_ylabel('Actual scalar physical rays')
    fig.suptitle('Case 14: completed new fields; old 128³ reference was only 68.1% visited')
    fig.tight_layout();fig.savefig(a.output/'region-scaling.png',dpi=155);plt.close(fig)

    row=next(r for r in m['runs']['depth7-case14-v1']['results'] if r['resolution']==128 and r['backend']=='independent')
    sizes=[2,4,8,16,32,64,128];counts=row['derived']['certificate_size_counts'];fractions=[counts.get(str(s),0)*s**3/128**3 for s in sizes]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(range(len(sizes)),[counts.get(str(s),0) for s in sizes],color='#438e96');axes[0].set(ylabel='Number of whole-region certificates')
    axes[1].bar(range(len(sizes)),np.array(fractions)*100,color='#b47e43');axes[1].set(ylabel='Percent of all receivers filled by that side')
    for ax in axes:ax.set_xticks(range(len(sizes)),list(map(str,sizes)));ax.set_xlabel('Receiver cell side, voxels');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Complete independent case 14 /128³: counts and voxel coverage are different')
    fig.tight_layout();fig.savefig(a.output/'certificate-size-distribution.png',dpi=155);plt.close(fig)

    path=a.root/'depth7-case14-v1/gaussian_sum-014-128-F_proof_reuse-independent.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['archive_SHA256'];data=np.load(path);assert data['valid'].all()
    z=64;solid=data['solid'][z];lit=data['response'][z];image=np.zeros((*lit.shape,3));image[:]=[.24,.43,.58];image[lit]=[.97,.82,.43];image[solid]=[.15,.17,.19]
    times=data['ray_nanoseconds'][z].astype(float)/1e6;valid=times[times>0]
    fig,axes=plt.subplots(1,2,figsize=(10,4.6));axes[0].imshow(image,origin='lower',extent=(0,1,0,1));axes[0].set_title('Exact sampled response + solid\ngold: lit FREE; blue: shadow FREE; gray: solid',fontsize=9)
    cmap=plt.get_cmap('magma').copy();cmap.set_bad('white');im=axes[1].imshow(np.ma.masked_where(times==0,times),origin='lower',extent=(0,1,0,1),cmap=cmap,norm=LogNorm(max(.001,float(valid.min())),max(.002,float(np.percentile(valid,99)))))
    axes[1].set_title('Timed scalar rays only\nwhite: no ray (proof / known solid)',fontsize=9);fig.colorbar(im,ax=axes[1],label='Native scalar-ray wall ms',shrink=.8)
    for ax in axes:ax.set(xlabel='world x',ylabel='world y')
    fig.suptitle('Certified 128³ case 14, slice z=64.5/128; sparse work is not zero physical cost')
    fig.tight_layout();fig.savefig(a.output/'response-and-residual-work.png',dpi=155);plt.close(fig)


if __name__=='__main__':main()
