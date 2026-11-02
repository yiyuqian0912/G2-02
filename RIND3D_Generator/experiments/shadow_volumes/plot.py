"""Scientific figures; rounded plotting vertices never define ground truth."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from .benchmark import ROOT
from .cases import SOURCES


def save(fig,name):
    out=ROOT/'figures';out.mkdir(exist_ok=True);fig.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)


def main():
    d=json.loads((ROOT/'diversity-statistics.json').read_text())
    fig=plt.figure(figsize=(13,14))
    for i,scene in enumerate(d['scenes']):
        ax=fig.add_subplot(4,3,i+1,projection='3d')
        for obj in scene['objects']:
            v=np.asarray(obj['vertices_for_plot']);faces=ConvexHull(v).simplices
            ax.add_collection3d(Poly3DCollection(v[faces],facecolors='#6f9ebd',edgecolors='#345873',linewidths=.35,alpha=.22))
        ax.scatter(*SOURCES[0],color='#d24c35',marker='*',s=40)
        ax.set(xlim=(0,1),ylim=(0,1),zlim=(0,1),xticks=[],yticks=[],zticks=[]);ax.set_box_aspect((1,1,1));ax.view_init(elev=20,azim=-55)
        if scene['family']=='thin_slab':ax.view_init(elev=35,azim=25)
        count=scene['obstacle_count'];ax.set_title(f"{scene['family']}  ({count} obstacle"+('s)' if count!=1 else ')'),fontsize=10)
    fig.suptitle('12 frozen geometry examples in [0,1]³; red star is one point source\nCanonical truth is lossless affine/halfspace data; rounded vertices are for display only',fontsize=11)
    fig.subplots_adjust(top=.92,hspace=.30,wspace=.04);save(fig,'geometry-diversity')
    s=json.loads((ROOT/'summary.json').read_text());fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    x=np.arange(2)
    for j,method in enumerate(('B','B_native','C')):
        vals=[]
        for name in ('box','corridor'):
            r=next(v for v in s['full'] if v['family']==name and v['method']==method)
            vals.append(r['wall_seconds'])
        bars=axes[0].bar(x+(j-1)*.25,vals,.23,label=method,color=('#888888','#ddaa33','#2879ae')[j])
        for b,v in zip(bars,vals):axes[0].text(b.get_x()+b.get_width()/2,v+.25,f'{v:.2f}',ha='center',fontsize=8)
    axes[0].set_xticks(x,['single box','corridor']);axes[0].set_ylabel('Complete 1024³ field seconds, setup separate');axes[0].legend(frameon=False,fontsize=8)
    axes[0].set_title('Same axis-aligned geometry: native corner control wins',fontsize=10)
    for name,color in (('poly8','#228833'),('poly16','#4477aa'),('poly32','#aa4499')):
        row=next(v for v in s['small'] if v['family']==name);f=next(v for v in row['rows'] if v['method']=='C' and v['resolution']==128)
        planes=sum(row['shadow']['shadow_plane_counts']);axes[1].scatter(planes,f['wall_seconds'],color=color,s=65);axes[1].annotate(name,(planes,f['wall_seconds']),xytext=(7,3),textcoords='offset points',fontsize=9)
    axes[1].set_xlabel('Stored shadow halfspaces (redundancies retained)');axes[1].set_ylabel('Complete 128³ field seconds, setup separate');axes[1].set_title('Quadratic plane growth is a measured cost',fontsize=10)
    axes[1].set_xlim(0,320);axes[1].grid(alpha=.15);fig.tight_layout();save(fig,'performance-controls')


if __name__=='__main__':main()
