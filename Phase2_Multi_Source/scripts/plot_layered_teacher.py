"""Plot one proposal's residuals, one-hop subtraction and global illumination."""
import argparse
from pathlib import Path
import numpy as np
from rind_phase2.data import Phase2Dataset
from rind_phase2.teacher import load_teacher_record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('teacher',type=Path);p.add_argument('--index',type=int);p.add_argument('--output',type=Path)
    a=p.parse_args();record=load_teacher_record(a.teacher)
    traces=record['metadata'].get('layer_traces')
    if not traces:p.error('This record has no layered trace')
    index=a.index if a.index is not None else int(np.argmin(record['physical_cost']))
    if not 0<=index<len(traces):p.error('index outside retained support')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ds=Phase2Dataset(record['metadata']['dataset_root']);sid=record['metadata']['scene_id'];x,y,L=map(int,record['window'])
    original=ds.get_region(sid,x,y,L);residual=original['response'].copy();obs=original['obstacle']
    steps=traces[index];fig,axes=plt.subplots(len(steps),4,figsize=(14,3.3*len(steps)),squeeze=False,layout='constrained')
    vmax=max(float(residual.max()),1e-12)
    gray=plt.get_cmap('gray').copy();gray.set_bad('.6')
    color=plt.get_cmap('viridis').copy();color.set_bad('.6')
    for row,step in zip(axes,steps):
        selected=np.zeros(residual.size,bool)
        for start,length in step['removed_runs']:selected[start:start+length]=True
        selected=selected.reshape(residual.shape)
        unit=ds.rerender(sid,record['window'],[[*step['source'][:2],1.]])
        row[0].imshow(np.ma.array(residual,mask=obs),cmap=color,vmin=0,vmax=vmax,interpolation='nearest')
        row[0].set_title(f"Step {step['step']}: residual before")
        row[1].imshow(np.ma.array(selected,mask=obs),cmap=gray,vmin=0,vmax=1,interpolation='nearest')
        row[1].set_title(f"Subtract {step['anchor_intensity']:.5g} here")
        residual[selected]-=step['anchor_intensity'];residual[np.abs(residual)<1e-12]=0
        row[2].imshow(np.ma.array(residual,mask=obs),cmap=color,vmin=0,vmax=vmax,interpolation='nearest');row[2].set_title('Residual after')
        row[3].imshow(np.ma.array(unit,mask=obs),cmap=gray,vmin=0,vmax=1,interpolation='nearest')
        row[3].set_title(f"Source at {step['source'][:2]}\nGlobal visibility, not subtraction mask")
        for ax in row:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(f"Scene {sid}, view {record['metadata']['view_id']}, proposal {index}; full-response MAE={record['physical_cost'][index]:.6g}")
    output=a.output or a.teacher.with_suffix('.layers.png');output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=160);plt.close(fig)
    print(output)

if __name__=='__main__':main()
