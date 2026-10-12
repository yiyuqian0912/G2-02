"""Phase-I style teacher/student visualization for one complete-book test record."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve();REPO=HERE.parents[3]
P1=REPO/'Phase1_Single_Source_Single_Intensity';P2=REPO/'Phase2_Multi_Source'
for item in (P1/'src',P2/'src',P2/'vendor/rind-dataset',HERE.parent):
    if str(item) not in sys.path:sys.path.insert(0,str(item))

from rind_phase2.data import Phase2Dataset
from train_combined import CombinedRecords,read_config,context


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--index',type=int,default=0)
    p.add_argument('--output',type=Path);p.add_argument('--show-truth',action='store_true')
    p.add_argument('--teacher-only',action='store_true',
                   help='write layer peeling/probability figures after prepare, without a student checkpoint')
    p.add_argument('--dense-spacing',type=int,default=0,help='0 disables dense student map; use 1 for pixel centers')
    p.add_argument('--chunk',type=int,default=4096);a=p.parse_args()
    c=read_config(a.config);dataset,root=context(c);manifest=json.loads((root/'records.json').read_text())
    records=CombinedRecords(manifest['records']['test'],dataset,root,
        c['training'].get('normalization_support','geometry'),c['teacher'].get('tolerance',1e-12))
    if not 0<=a.index<len(records):p.error('test index out of range')
    sample=records[a.index];entry=manifest['records']['test'][a.index]
    directory=a.output or (root/'visuals'/f'test_{a.index}' if a.teacher_only else
                           root/'models'/f'seed_{a.seed}'/'visuals'/f'test_{a.index}')
    directory.mkdir(parents=True,exist_ok=True)
    run_dir=root/entry['directory']
    truth=dataset.get_source_params(entry['scene_id'])[:,:2] if a.show_truth else np.empty((0,2))
    layer_visuals=plot_teacher_layers(directory,run_dir,sample,truth,
                                      int(dataset.manifest['global_size']))
    if a.teacher_only:
        metadata={'scene_id':entry['scene_id'],'view_id':entry['view_id'],'test_index':a.index,
                  'teacher_only':True,'teacher_rule':'equal-weight layer mixture',
                  'truth_used_for_inference':False,'teacher_layer_visualization':layer_visuals}
        (directory/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(directory);return
    from rind_phase1.train import load_model
    model,checkpoint=load_model(root/'models'/f'seed_{a.seed}'/'best.pt');model.eval()
    import torch
    with torch.inference_mode():
        response=torch.as_tensor(sample['response'][None],dtype=torch.float32)
        obstacle=torch.as_tensor(sample['obstacle'][None],dtype=torch.bool)
        window=torch.as_tensor(sample['window'][None],dtype=torch.float32)
        xy=torch.as_tensor(sample['candidate_xy'][None],dtype=torch.float32)
        support=torch.as_tensor(sample['support'][None],dtype=torch.bool)
        predicted=model.predict(response,obstacle,window,xy,support)['student_prob'][0].numpy()
    with np.load(run_dir/'combined.npz') as z:
        grid_index=z['grid_index'];x_axis=z['x_axis'];y_axis=z['y_axis']
    shape=(len(y_axis),len(x_axis))
    def raster(values,fill=np.nan):
        image=np.full(shape,fill,dtype=float);image.flat[grid_index]=values;return image
    teacher=raster(sample['teacher_prob']);student=raster(predicted)
    plot_matched(directory/'teacher_student.png',sample,teacher,student,truth,int(dataset.manifest['global_size']))
    metadata={'scene_id':entry['scene_id'],'view_id':entry['view_id'],'test_index':a.index,
              'checkpoint_epoch':checkpoint['epoch'],'teacher_probability_sum':float(sample['teacher_prob'].sum()),
              'student_probability_sum':float(predicted.sum()),'teacher_rule':'equal-weight layer mixture',
              'truth_used_for_inference':False,'teacher_layer_visualization':layer_visuals}
    np.savez_compressed(directory/'teacher_student.npz',teacher_prob=sample['teacher_prob'],student_prob=predicted,
        candidate_xy=sample['candidate_xy'],valid=sample['valid'],response=sample['response'],obstacle=sample['obstacle'],window=sample['window'])
    if a.dense_spacing:
        dense(directory,model,sample,dataset,entry,a.dense_spacing,a.chunk,truth)
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(directory)


def layer_peeling_arrays(layer,intensity):
    """Return the four local arrays that define one residual peel."""
    before=np.asarray(layer['residual_before'],dtype=np.float64)
    raw_mask=np.asarray(layer.get('layer_response',layer.get('full_mask')))
    if not np.isin(raw_mask,[0,1]).all():
        raise ValueError('current layer response must be binary 0/1')
    mask=raw_mask.astype(bool)
    after=np.asarray(layer['residual_after'],dtype=np.float64)
    if before.ndim!=2 or mask.shape!=before.shape or after.shape!=before.shape:
        raise ValueError('malformed saved layer peeling arrays')
    contribution=float(intensity)*mask
    if not np.allclose(after,before-contribution,rtol=0,atol=1e-10):
        raise ValueError('saved residual_after does not equal residual_before - layer contribution')
    return before,mask,contribution,after


def _load_teacher_layers(run_dir):
    report=json.loads((run_dir/'report.json').read_text());layers=[]
    for detail in report.get('layers',[]):
        path=run_dir/f"layer-{int(detail['step']):02d}.npz"
        if not path.exists():
            continue
        with np.load(path,allow_pickle=False) as archive:
            if 'residual_after' not in archive.files:
                continue
            numeric={key:archive[key].copy() for key in archive.files}
        layers.append((detail,numeric))
    return report,layers


def _draw_peeling_row(axes,arrays,obstacle,extent,title):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm,ListedColormap
    before,mask,contribution,after=arrays
    vmax=max(float(np.max(before[~obstacle],initial=0)),
             float(np.max(after[~obstacle],initial=0)),
             float(np.max(contribution,initial=0)),np.finfo(float).eps)
    residual_cmap=plt.get_cmap('viridis').copy();residual_cmap.set_bad('#496d89')
    mask_cmap=ListedColormap(['#000000','#ffffff']);mask_cmap.set_bad('#2f80ed')
    binary_norm=BoundaryNorm([-.5,.5,1.5],mask_cmap.N)
    panels=((before,'Residual before',residual_cmap,0,vmax,None),
            (mask.astype(np.uint8),'Layer response: 0 black, 1 white, obstacle blue',mask_cmap,None,None,binary_norm),
            (contribution,'Subtracted contribution',residual_cmap,0,vmax,None),
            (after,'Residual after',residual_cmap,0,vmax,None))
    for ax,(values,label,cmap,vmin,vmax_panel,norm) in zip(axes,panels):
        shown=np.ma.array(values,mask=obstacle)
        kwargs={'origin':'upper','extent':extent,'interpolation':'nearest','cmap':cmap}
        if norm is None:kwargs.update(vmin=vmin,vmax=vmax_panel)
        else:kwargs['norm']=norm
        image=ax.imshow(shown,**kwargs)
        ax.set(title=label,xlabel='World x',ylabel='World y')
        colorbar=ax.figure.colorbar(image,ax=ax,fraction=.046,pad=.04,
                                    ticks=[0,1] if norm is not None else None)
        if norm is not None:colorbar.ax.set_yticklabels(['0 absent','1 present'])
    axes[0].text(.01,.99,title,transform=axes[0].transAxes,va='top',ha='left',fontsize=8,
                 color='white',bbox={'facecolor':'black','alpha':.72,'pad':3,'edgecolor':'none'})


def plot_teacher_layers(directory,run_dir,sample,truth,world):
    """Write local peel diagnostics and every layer's full-field Phase-I mass."""
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    report,layers=_load_teacher_layers(run_dir)
    layer_dir=directory/'teacher_layers';layer_dir.mkdir(parents=True,exist_ok=True)
    x,y,L=map(int,sample['window']);extent=(x,x+L,y+L,y)
    obstacle=np.asarray(sample['obstacle'],dtype=bool)
    summaries=[];individual=[]
    for index,(detail,layer) in enumerate(layers,1):
        arrays=layer_peeling_arrays(layer,detail['intensity'])
        conflicts=len(detail.get('trace',{}).get('conflict',[]))
        title=(f"Layer {index} | a={detail['intensity']:.6g} | "
               f"min Phase-I cost={detail.get('minimum_physical_cost',float('nan')):.6g} | "
               f"exact grid matches={detail.get('exact_grid_matches','n/a')} | conflicts={conflicts}")
        fig,axes=plt.subplots(1,4,figsize=(17,4.2),layout='constrained')
        _draw_peeling_row(axes,arrays,obstacle,extent,title)
        fig.suptitle(f'Teacher residual peeling — scene {report["scene_id"]}, view {report["view_id"]}')
        path=layer_dir/f'layer_{index:02d}_peeling.png';fig.savefig(path,dpi=180);plt.close(fig)
        individual.append(str(path.relative_to(directory)))
        summaries.append({'layer':index,'intensity':float(detail['intensity']),
                          'minimum_physical_cost':detail.get('minimum_physical_cost'),
                          'exact_grid_matches':detail.get('exact_grid_matches'),
                          'conflict_regions':conflicts,
                          'peeling_consistent':True})
    overview=[]
    per_page=6
    for page,start in enumerate(range(0,len(layers),per_page),1):
        subset=layers[start:start+per_page]
        fig,axes=plt.subplots(len(subset),4,figsize=(17,3.8*len(subset)),squeeze=False,
                              layout='constrained')
        for row,(detail,layer) in enumerate(subset):
            index=start+row+1;conflicts=len(detail.get('trace',{}).get('conflict',[]))
            title=(f"L{index}: a={detail['intensity']:.6g}; "
                   f"min cost={detail.get('minimum_physical_cost',float('nan')):.4g}; "
                   f"conflicts={conflicts}")
            _draw_peeling_row(axes[row],layer_peeling_arrays(layer,detail['intensity']),
                              obstacle,extent,title)
        fig.suptitle(f'Teacher layer peeling — scene {report["scene_id"]}, view {report["view_id"]} '
                     f'— page {page}')
        path=directory/f'teacher_peeling_page_{page:02d}.png';fig.savefig(path,dpi=170);plt.close(fig)
        overview.append(path.name)
    probability_pages=[]
    with np.load(run_dir/'combined.npz',allow_pickle=False) as combined:
        grid_index=combined['grid_index'];x_axis=combined['x_axis'];y_axis=combined['y_axis']
        combined_probability=combined['probability']
    probability_maps=[('Combined teacher',combined_probability)]+[
        (f"Layer {i}: a={detail['intensity']:.6g}",layer['probability'])
        for i,(detail,layer) in enumerate(layers,1) if 'probability' in layer]
    probability_vmax=max((float(np.max(values,initial=0)) for _,values in probability_maps),
                          default=np.finfo(float).eps)
    cmap=plt.get_cmap('inferno').copy();cmap.set_bad('#a4abb5')
    for page,start in enumerate(range(0,len(probability_maps),4),1):
        subset=probability_maps[start:start+4]
        fig,axes=plt.subplots(1,len(subset),figsize=(5.3*len(subset),5.1),squeeze=False,
                              layout='constrained')
        for ax,(title,probability) in zip(axes[0],subset):
            raster=np.full((len(y_axis),len(x_axis)),np.nan,dtype=np.float64)
            raster.flat[grid_index]=probability
            image=ax.imshow(raster,origin='upper',extent=(0,world,world,0),cmap=cmap,
                            vmin=0,vmax=probability_vmax,interpolation='nearest')
            ax.add_patch(Rectangle((x,y),L,L,fill=False,edgecolor='cyan',linewidth=1.3))
            if len(truth):ax.scatter(truth[:,0],truth[:,1],marker='x',c='lime',s=35)
            ax.set(title=title,xlabel='Source world x',ylabel='Source world y',
                   xlim=(0,world),ylim=(world,0))
            fig.colorbar(image,ax=ax,label='Discrete probability mass')
        fig.suptitle('Phase-I full-field probability for each recovered teacher layer')
        path=directory/f'teacher_probability_layers_page_{page:02d}.png'
        fig.savefig(path,dpi=180);plt.close(fig);probability_pages.append(path.name)
    manifest={'status':report['status'],'layers':summaries,'individual_peeling':individual,
              'overview_pages':overview,'probability_pages':probability_pages,
              'obstacle_color':'blue','layer_response_encoding':'0=black absent, 1=white present, obstacle=blue',
              'coordinates':'world coordinates; y increases downward'}
    (directory/'teacher_layers.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def plot_matched(path,sample,teacher,student,truth,world):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    cmap=plt.get_cmap('inferno').copy();cmap.set_bad('#a4abb5')
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),layout='constrained')
    response_cmap=plt.get_cmap('viridis').copy();response_cmap.set_bad('#2f80ed')
    local=np.ma.array(sample['response'],mask=sample['obstacle'])
    local_image=axes[0].imshow(local,origin='upper',interpolation='nearest',cmap=response_cmap,
                               vmin=0,vmax=max(float(local.max()),np.finfo(float).eps))
    axes[0].set(title=f'Observed response ({sample["window"][2]} x {sample["window"][2]})\nBlue: local obstacle',xlabel='Local x',ylabel='Local y')
    fig.colorbar(local_image,ax=axes[0],label='Observed additive response')
    vmax=max(float(np.nanmax(teacher)),float(np.nanmax(student)),np.finfo(float).tiny)
    x,y,L=sample['window']
    for ax,image,title in ((axes[1],teacher,'Teacher: equal-weight layer mixture'),(axes[2],student,'Student probability')):
        im=ax.imshow(image,origin='upper',extent=(0,world,world,0),cmap=cmap,vmin=0,vmax=vmax,interpolation='nearest')
        ax.add_patch(Rectangle((x,y),L,L,fill=False,edgecolor='cyan',linewidth=1.2))
        if len(truth):ax.scatter(truth[:,0],truth[:,1],marker='x',c='lime',s=35,label='Generating sources (evaluation only)')
        ax.set(title=title,xlabel='Source world x',ylabel='Source world y',xlim=(0,world),ylim=(world,0))
        if len(truth):ax.legend(fontsize=7,loc='lower left')
        fig.colorbar(im,ax=ax,label='Discrete probability mass')
    fig.suptitle('One original observation → one full-field teacher target → one student field')
    fig.savefig(path,dpi=180);plt.close(fig)


def dense(directory,model,sample,dataset,entry,spacing,chunk,truth):
    from rind_phase1.sampling import score_grid,probability_from_energy
    from rind_phase2.physics import JointResponseEvaluator
    world=int(dataset.manifest['global_size'])
    energy,xy=score_grid(model,sample,world,spacing=spacing,chunk=chunk)
    axis=np.arange(spacing/2,world,spacing);valid=np.ones(energy.shape,bool)
    if model.normalization_support=='geometry':
        xx,yy=np.meshgrid(axis,axis,indexing='xy');points=np.column_stack((xx.ravel(),yy.ravel()))
        ev=JointResponseEvaluator(dataset,entry['scene_id'],entry['view_id']);valid=ev.positions_valid(points).reshape(energy.shape)
    else:
        x,y,L=sample['window'];xx,yy=np.meshgrid(axis,axis,indexing='xy')
        valid &= ~((xx>=x)&(xx<x+L)&(yy>=y)&(yy<y+L))
    probability=probability_from_energy(energy,valid)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm,Normalize
    from matplotlib.patches import Rectangle
    cmap=plt.get_cmap('inferno').copy();cmap.set_bad('#a4abb5');positive=probability[valid&(probability>0)]
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),layout='constrained')
    response_cmap=plt.get_cmap('viridis').copy();response_cmap.set_bad('#2f80ed')
    local=np.ma.array(sample['response'],mask=sample['obstacle'])
    local_image=axes[0].imshow(local,cmap=response_cmap,vmin=0,
                               vmax=max(float(local.max()),np.finfo(float).eps),interpolation='nearest')
    axes[0].set(title='Observed additive response\nBlue: local obstacle',xlabel='Local x',ylabel='Local y')
    fig.colorbar(local_image,ax=axes[0],label='Observed additive response');x,y,L=sample['window']
    norms=(Normalize(0,float(positive.max())),LogNorm(max(float(positive.min()),np.finfo(float).tiny),float(positive.max())))
    for ax,norm,title in zip(axes[1:],norms,('Dense student probability','Dense student probability (log)')):
        im=ax.imshow(np.ma.array(probability,mask=~valid),extent=(0,world,world,0),cmap=cmap,norm=norm,interpolation='nearest')
        ax.add_patch(Rectangle((x,y),L,L,fill=False,edgecolor='cyan'))
        if len(truth):ax.scatter(truth[:,0],truth[:,1],marker='x',c='lime',s=35)
        ax.set(title=title,xlabel='Source world x',ylabel='Source world y');fig.colorbar(im,ax=ax)
    fig.savefig(directory/'dense_student.png',dpi=180);plt.close(fig)
    np.savez_compressed(directory/'dense_student.npz',energy=energy,probability=probability,valid=valid,spacing=spacing)


if __name__=='__main__':main()
