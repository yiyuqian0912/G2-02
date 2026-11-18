"""Plot XY marginals of two stored, globally certified 128³ distributions."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('records',type=Path)
    ap.add_argument('output',type=Path)
    a=ap.parse_args()
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.8),layout='constrained')
    pairs=[('000004_w00_s0','Convex polyhedra · 64³','Exact · 214 cells'),
           ('000025_w04_s0','Barriers · 128³','TV bound 0.02238 · 8,800 cells')]
    arrays=[]
    for ax,(key,title,detail) in zip(axes,pairs):
        with gzip.open(a.records/'grids_heldout_raster'/(key+'.json.gz'),'rt') as f: g=json.load(f)
        c=np.array(g['cells'])
        xy=np.zeros((128,128))
        np.add.at(xy,(c//128%128,c//16384),[w/g['denominator'] for w in g['weights']])
        arrays.append(np.ma.masked_equal(xy,0))
        ax.set(title=title+'\n'+detail,xlabel='Source x',ylabel='Source y',xticks=[0,.5,1],yticks=[0,.5,1])
        ax.title.set_fontsize(11)
    norm=LogNorm(vmin=1e-8,vmax=1)
    for ax,array in zip(axes,arrays):
        artist=ax.imshow(array,origin='lower',extent=(0,1,0,1),interpolation='nearest',
                         cmap='viridis',norm=norm)
    cb=fig.colorbar(artist,ax=axes,shrink=.83,extend='min')
    cb.set_label('Probability per XY cell (summed over z)')
    fig.savefig(a.output,dpi=170)


if __name__=='__main__': main()
