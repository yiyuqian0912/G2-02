"""Scientific plots of measured B2 work; meshes are visualization only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage.measure import marching_cubes

from ..models import load


def shape(ax,scene,zoom=False):
    n=64;lo=np.zeros(3);hi=np.ones(3)
    if zoom:
        centers=np.array([k.center for k in scene.kernels]);lo=centers.min(0)-.025;hi=centers.max(0)+.025
    step=(hi-lo)/n;axes=[lo[j]+(np.arange(n)+.5)*step[j] for j in range(3)]
    z,y,x=np.meshgrid(axes[2],axes[1],axes[0],indexing='ij');values=scene.approx_values(np.c_[x.ravel(),y.ravel(),z.ravel()]).reshape((n,)*3)
    vertices,faces,_,_=marching_cubes(values.astype(np.float32),0)
    xyz=lo+(vertices[:,::-1]+.5)*step
    mesh=Poly3DCollection(xyz[faces],facecolor='#438e96',edgecolor='none',alpha=.9);mesh.set_rasterized(True);ax.add_collection3d(mesh)
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]));ax.set_box_aspect(hi-lo);ax.view_init(23,-58)
    ax.set_xticks([lo[0],hi[0]]);ax.set_yticks([lo[1],hi[1]]);ax.set_zticks([lo[2],hi[2]]);ax.tick_params(labelsize=6,pad=0)
    if zoom:
        # Avoid three long corner labels colliding in the small gallery panel.
        ax.set_xticks([lo[0]],[f'{lo[0]:.3f}'])
        ax.set_yticks([hi[1]],[f'{hi[1]:.3f}'])
        ax.set_zticklabels([f'{v:.3f}' for v in (lo[2],hi[2])])
        ax.set(xlabel='x',ylabel='y',zlabel='z')
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.pane.fill=False


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    m=json.loads(a.summary.read_text());names=['Python Arb','Native uncached','Cached','+ filters','+ groups','+ Taylor','Reference plain','+ tails','+ Taylor','Reference both']
    keys=['native_no_cache','native_cached','native_filters','native_groups','native_taylor','independent_plain','independent_tail','independent_taylor','independent']
    values=[m['python_total_ray_seconds']]+[m['modes'][k]['batch_wall_seconds_sum'] for k in keys]
    fig,ax=plt.subplots(figsize=(12,5));ax.bar(np.arange(len(values)),np.array(values)*1000/m['matched_rays'],color=['#a8adb4']+['#46999b']*5+['#b68a58']*4)
    ax.set(yscale='log',ylabel='Mean batch ms / ray',title='Identical frozen B2 geometry and 6,144 certified-FREE receiver rays')
    ax.set_xticks(np.arange(len(values)),names,rotation=22,ha='right');ax.grid(axis='y',alpha=.2)
    fig.text(.5,.012,'Batch calls include Python/native transitions; geometry setup reported separately. All ordinary matched ray labels resolved and agreed.',ha='center',fontsize=8)
    fig.subplots_adjust(bottom=.24,top=.88,left=.08,right=.99);fig.savefig(a.output/'b2-query-ablation.png',dpi=155);plt.close(fig)

    row=m['field128'][0];fig,ax=plt.subplots(figsize=(8,5))
    for backend,color in [('primary','#438e96'),('independent','#b47e43')]:
        completed=[(f['resolution'],next(x for x in f['backends'] if x['backend']==backend)) for f in row['fields']]
        done=[(n,b['wall_seconds']) for n,b in completed if b['complete']]
        ax.plot([n for n,t in done],[t for n,t in done],'o-',label=backend,color=color)
        for n,b in completed:
            if not b['complete']:
                ax.scatter([n],[b['wall_seconds']],marker='x',s=100,c='#ba4652')
                ax.annotate(f"INCOMPLETE: {b['valid_voxels']/n**3:.1%} verified\n{b['wall_seconds']:.0f}s limit",(n,b['wall_seconds']),xytext=(-125,-5),textcoords='offset points',fontsize=9)
    ax.set(xscale='log',yscale='log',xlabel='Complete world sampling side N',ylabel='Field wall seconds',title='Matched case 14: complete fields and explicit incomplete gate')
    ax.set_xticks([8,32,64,128],['8','32','64','128']);ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(a.output/'b2-field-scaling.png',dpi=155);plt.close(fig)

    data=m['generation']['v2'];fig,ax=plt.subplots(figsize=(8,4.5));targets=[.001,.01,.05,.15]
    for i,t in enumerate(targets):
        values=[r['quality']['occupied_ratio_estimate'] for r in data['cases'] if r['target']==t and r['quality']]
        ax.scatter(np.full(len(values),i),values,s=34,color='#438e96');ax.scatter([i],[t],marker='_',s=280,color='#50555b')
        ax.annotate(f'{len(values)}/6 retained',(i,t),xytext=(-8 if i==3 else 8,5),textcoords='offset points',fontsize=9,ha='right' if i==3 else 'left')
    ax.set(yscale='log',ylim=(.0007,.25),ylabel='64³ occupancy estimate',title='Joint B2 calibration: retained cases, with all shortages visible')
    ax.set_xticks(range(4),['0.1% target','1%','5%','15%']);ax.grid(axis='y',alpha=.2)
    fig.text(.5,.015,'Conditional on containment/interior/source proofs; 7/24 cases not retained. Sampled occupancy is not an exact-volume certificate.',ha='center',fontsize=8)
    fig.subplots_adjust(bottom=.18,top=.88,left=.12,right=.98);fig.savefig(a.output/'b2-generation-regimes.png',dpi=155);plt.close(fig)

    rows=json.loads((a.root/'generation-v2/summary.json').read_text())['cases'];fig=plt.figure(figsize=(12,7.5));selected=[]
    for t in targets:
        candidates=[r for r in rows if r['target']==t and 'quality' in r]
        if candidates:selected.append(candidates[0])
    for i,r in enumerate(selected):
        ax=fig.add_subplot(2,3,i+1,projection='3d');scene=load(a.root/'generation-v2'/f"case-{r['ordinal']:03d}"/'geometry.json');shape(ax,scene)
        ax.set_title(f"Joint generation case {r['ordinal']}\noccupancy ~{100*r['quality']['occupied_ratio_estimate']:.2f}%; voxel components {r['quality']['components_6_connected_estimate']}",fontsize=9)
    for j,scale in enumerate((1.,.125)):
        ax=fig.add_subplot(2,3,5+j,projection='3d');scene=load(a.root/'controlled'/f'branch-{scale}.json');shape(ax,scene,zoom=scale<1)
        ax.set_title('Continuous branching proof\n'+('world view' if scale==1 else 'thin structure, zoomed world coordinates'),fontsize=9)
    fig.suptitle('RIND3D B2 research: positive Gaussian sums, not unions or truth meshes',fontsize=13)
    fig.text(.5,.018,'64³ float display meshes only. First retained seed per occupancy regime; connectivity/component counts are estimates except explicit continuous proof fixtures.',ha='center',fontsize=8)
    fig.subplots_adjust(top=.88,bottom=.08,left=.02,right=.98,wspace=.02,hspace=.2);fig.savefig(a.output/'b2-geometry-gallery.png',dpi=155);plt.close(fig)


if __name__=='__main__':main()
