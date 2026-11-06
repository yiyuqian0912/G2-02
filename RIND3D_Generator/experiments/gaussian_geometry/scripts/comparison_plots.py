"""Display-only placement/continuous-bridge examples and certified sampled slices."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from experiments.gaussian_geometry.models import load
from experiments.gaussian_geometry.plots import draw_shape


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();base=args.root
    args.output.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(12,7))
    for col,ordinal in enumerate((7,14,22)):
        for row,run in enumerate(('distribution-v2','capped-convex-v1')):
            path=base/run/f'gaussian_convex-{ordinal:03d}'
            scene=load(path/'geometry.json');r=json.loads((path/'result.json').read_text())
            ax=fig.add_subplot(2,3,row*3+col+1,projection='3d');draw_shape(ax,scene)
            ax.set_title(f'{"Global worst-object cap" if row==0 else "Individual object caps"}\n'
                         f'case {ordinal}; estimated occupancy {100*r["quality"]["occupied_ratio_estimate"]:.2f}%',fontsize=9)
    fig.suptitle('Same Gaussian centers and primitive proposals; different declared world-fitting policies',fontsize=12)
    fig.text(.5,.012,'64³ display meshes only; each continuous convex union separately passes unchanged world/interior proofs.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.89,bottom=.06,left=.02,right=.98,hspace=.13,wspace=.03)
    fig.savefig(args.output/'gaussian-placement-ablation.png',dpi=150);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,5.5));n=360
    axis=np.linspace(.2,.85,n);y,x=np.meshgrid(axis,axis,indexing='ij')
    points=np.c_[x.ravel(),y.ravel(),np.full(n*n,.5)]
    for ax,name,title in zip(axes,('independent-levels','summed-kernels'),('B1: exactly 7 disjoint ellipsoids','B2: all 7 centers continuously connected')):
        scene=load(base/'bridge-control'/f'{name}.json')
        ax.contourf(x,y,(scene.approx_values(points)>=0).reshape(n,n),levels=[-.5,.5,1.5],colors=['#f6f7f8','#669c9f'])
        centers=np.array([k.center for k in scene.kernels]);ax.scatter(centers[:,0],centers[:,1],s=17,c='#183c47',zorder=4)
        if name=='summed-kernels':
            proof=json.loads((base/'bridge-control/proof.json').read_text())
            for edge in proof['edges']:
                a,b=centers[edge['centers']];ax.plot([a[0],b[0]],[a[1],b[1]],color='#d38b26',lw=2)
        ax.set(xlim=(.2,.85),ylim=(.2,.85),aspect='equal',xlabel='X',ylabel='Y',title=title)
    fig.suptitle('Identical kernel parameters and level exp(-1/2); Z=0.5 display slice')
    fig.subplots_adjust(left=.07,right=.98,bottom=.20,top=.84,wspace=.23)
    fig.text(.5,.035,'Orange connections are proved SOLID over closed continuous segments by both Arb and MPFR paths.\n'
             'Display contours are approximate. Total B2 component count is not certified.',ha='center',fontsize=9)
    fig.savefig(args.output/'gaussian-bridge-proof.png',dpi=155,bbox_inches='tight');plt.close(fig)

    # One common 8³ receiver lattice. Unknown voxels have their own color;
    # a partially checked field must never look like a complete label map.
    fig,axes=plt.subplots(1,4,figsize=(13,4.5))
    for ax,model in zip(axes,('gaussian_convex','ellipsoid_union','gaussian_sum','spectral_grf')):
        case=f'{model}-014';path=base/'fields-v3'/case
        field=np.load(path/'field-8.npz');y,m,v=field['response'],field['solid'],field['valid']
        z=3;labels=np.where(~v[z],3,np.where(m[z],2,np.where(y[z],1,0)))
        ax.imshow(labels,origin='lower',extent=(0,1,0,1),interpolation='nearest',
                  cmap=ListedColormap(['#2b3857','#f0d394','#8b919c','#d95ba2']),vmin=0,vmax=3)
        ax.set_title(f'{model}\n8³ receivers; valid {int(v.sum())}/512',fontsize=9)
        ax.set(xlabel='X',ylabel='Y')
    fig.suptitle('Actual rigorous-predicate response slices, Z receiver index 3; case 14')
    fig.subplots_adjust(left=.045,right=.99,bottom=.23,top=.79,wspace=.28)
    fig.text(.5,.025,'Gold: lit FREE. Navy: shadowed FREE. Gray: solid. Magenta: unresolved.\n'
             'These are sampled research fields, not production scenes or continuous-space reconstructions.',ha='center',fontsize=9)
    fig.savefig(args.output/'gaussian-physical-slices.png',dpi=150,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
