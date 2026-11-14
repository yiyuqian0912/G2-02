"""Optional Matplotlib: render an actual slice of a stored observation."""
import argparse
from rind3d_reader import RINDDataset

def main():
 import matplotlib.pyplot as plt
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--index',type=int,default=0);p.add_argument('--slice',type=int,help='Local Z index; default chooses the slice with most minority-class FREE voxels');p.add_argument('--output',default='window.png');a=p.parse_args()
 with RINDDataset(a.root,split=None) as data:
  item=data[a.index];response=item['observed_response'].unpack();L=item['window_size'];scene=data.scene(item['scene_id']);solid=scene.roi(tuple(item['window_origin']),L,field='solid',packed=False)
  lit=(response&~solid).sum(axis=(1,2));shadow=(~response&~solid).sum(axis=(1,2));score=lit.clip(max=shadow);z=int(score.argmax()) if score.max() else L//2
  if a.slice is not None:z=a.slice
  if not 0<=z<L:p.error('--slice must be inside the selected window')
  fig,axes=plt.subplots(1,2,figsize=(8,4));axes[0].imshow(response[z],vmin=0,vmax=1,cmap='gray');axes[0].set_title(f'Binary response, local Z={z}');axes[1].imshow(solid[z],vmin=0,vmax=1,cmap='gray');axes[1].set_title('Solid mask');fig.tight_layout();fig.savefig(a.output,dpi=150);plt.close(fig)
if __name__=='__main__':main()
