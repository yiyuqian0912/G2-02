"""DISPLAY ONLY isosurfaces and finite-sample study figures; no truth meshes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage.measure import marching_cubes

from .models import load
from .generators import grid_points

NAMES={'gaussian_convex':'A  Gaussian-placed convex', 'ellipsoid_union':'B1  Independent ellipsoid union',
       'gaussian_sum':'B2  Thresholded Gaussian sum', 'spectral_grf':'C  Tapered finite Gaussian field'}
COLORS={'gaussian_convex':'#a38461','ellipsoid_union':'#72a2b0','gaussian_sum':'#4e8a87','spectral_grf':'#8c82b4'}


def draw_shape(ax,scene,n=64,color='#60969d'):
    values=scene.approx_values(grid_points(n)).reshape((n,)*3)
    if values.min()<0<values.max():
        vertices,faces,_,_=marching_cubes(values.astype(np.float32),level=0,spacing=(1/n,)*3)
        vertices=vertices[:,::-1]+.5/n
        mesh=Poly3DCollection(vertices[faces],facecolor=color,edgecolor='none',alpha=.85,linewidth=0)
        mesh.set_rasterized(True);ax.add_collection3d(mesh)
    ax.set(xlim=(0,1),ylim=(0,1),zlim=(0,1));ax.set_box_aspect((1,1,1));ax.view_init(24,-56)
    ax.set_xticks([0,1]);ax.set_yticks([0,1]);ax.set_zticks([0,1]);ax.tick_params(labelsize=6,pad=-3)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.pane.fill=False


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    results=json.loads((args.input/'summary.json').read_text())['cases'];models=list(NAMES)
    fig=plt.figure(figsize=(13,12),facecolor='white');ordinals=[3,7,14,22]
    for row,ordinal in enumerate(ordinals):
        for col,model in enumerate(models):
            ax=fig.add_subplot(4,4,row*4+col+1,projection='3d')
            path=args.input/f'{model}-{ordinal:03d}'
            if not (path/'geometry.json').exists():ax.text(.2,.5,.5,'Missing / incomplete case');continue
            scene=load(path/'geometry.json');draw_shape(ax,scene,color=COLORS[model])
            r=json.loads((path/'result.json').read_text());q=r.get('quality',{})
            label=f"case {ordinal} | occupancy ~{q.get('occupied_ratio_estimate',0)*100:.2f}% | voxel components {q.get('components_6_connected_estimate','?')}"
            ax.set_title((NAMES[model]+'\n' if row==0 else '')+label,fontsize=8,pad=1)
    fig.suptitle('RIND3D future-geometry research: continuous models, sampled DISPLAY previews',fontsize=14,y=.985)
    fig.text(.5,.012,'64³ float samples -> display isosurfaces only. These meshes and component counts are not canonical geometry or topology certificates.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.02,right=.98,bottom=.04,top=.94,wspace=.02,hspace=.09)
    fig.savefig(args.output/'gaussian-gallery.png',dpi=155);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for m in models:
        rows=[r for r in results if r.get('model')==m and 'quality' in r]
        x=[r['quality']['occupied_ratio_estimate'] for r in rows]
        axes[0,0].hist(x,bins=np.geomspace(.0005,.5,16),histtype='step',lw=2,label=NAMES[m],color=COLORS[m])
        axes[0,1].scatter(x,[r['quality']['components_6_connected_estimate'] for r in rows],s=24,alpha=.7,color=COLORS[m],label=m)
        rr=[r for r in rows if 'queries' in r]
        axes[1,0].scatter([r['quality']['occupied_ratio_estimate'] for r in rr],
            [1000*r['queries']['ray_seconds']/max(1,r['queries']['free_receiver_rays']) for r in rr],s=24,color=COLORS[m],alpha=.7)
        axes[1,1].scatter(x,[r['quality']['spatial_4x4x4_coverage_estimate'] for r in rows],s=24,color=COLORS[m],alpha=.7)
    axes[0,0].set(xscale='log',xlabel='Estimated occupied fraction (64³)',ylabel='Scene count');axes[0,0].legend(fontsize=7)
    axes[0,1].set(xscale='log',yscale='log',xlabel='Estimated occupied fraction',ylabel='6-connected voxel components (estimate)')
    axes[1,0].set(xscale='log',yscale='log',xlabel='Estimated occupied fraction',ylabel='Primary ms / attempted free-receiver ray')
    axes[1,1].set(xscale='log',xlabel='Estimated occupied fraction',ylabel='Occupied 4×4×4 spatial-bin fraction')
    for ax in axes.ravel():ax.grid(alpha=.2)
    fig.suptitle('Measured geometry / query tradeoffs; different shapes are not identical physics workloads')
    fig.savefig(args.output/'gaussian-distributions.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(12,11),constrained_layout=True)
    for row,ordinal in enumerate(ordinals):
        for col,m in enumerate(models):
            scene=load(args.input/f'{m}-{ordinal:03d}'/'geometry.json');n=192
            p=grid_points(32);solid=scene.approx_values(p)>=0
            z=float(np.mean(p[solid,2])) if solid.any() else .5
            y,x=np.meshgrid((np.arange(n)+.5)/n,(np.arange(n)+.5)/n,indexing='ij')
            values=scene.approx_values(np.c_[x.ravel(),y.ravel(),np.full(n*n,z)]).reshape(n,n)
            axes[row,col].imshow(values>=0,origin='lower',extent=(0,1,0,1),cmap='bone_r',interpolation='nearest')
            axes[row,col].set_title(f'{m}\ncase {ordinal} | display Z={z:.3f}',fontsize=8)
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
    fig.suptitle('Ordinary float cross-sections: visualization, not certified occupancy grids')
    fig.savefig(args.output/'gaussian-cross-sections.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
