"""Evaluate every source pixel center; never interpolate coarse candidate scores."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'src'), str(PROJECT/'vendor/rind-dataset')]
import numpy as np
import torch
from rind_phase1.data import Phase1Dataset
from rind_phase1.interfaces import validate_observation
from rind_phase1.train import load_model


def score_pixels(model, observation, world_size, chunk=4096):
    from rind_phase1.sampling import score_grid
    return score_grid(model,observation,world_size,spacing=1,chunk=chunk)[0]


def pixel_probability(energy,valid):
    from rind_phase1.sampling import probability_from_energy
    return probability_from_energy(energy,valid)


def save_visuals(output,sample,probability,valid,truth,metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm,Normalize
    from matplotlib.patches import Rectangle
    from PIL import Image
    world = len(probability)
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad('#a4abb5')
    positive = probability[valid & (probability>0)]
    low,high = float(positive.min()),float(positive.max())
    norm = Normalize(0,high)
    # Exact one-to-one raster: no dots, smoothing, or spatial interpolation.
    rgba = cmap(norm(np.ma.array(probability,mask=~valid)),bytes=True)
    Image.fromarray(rgba).save(output.with_suffix('.pixels.png'))
    fig,axes = plt.subplots(1,3,figsize=(16,5.3),layout='constrained',gridspec_kw={'width_ratios':[.8,1,1]})
    local = np.stack([sample['response']]*3,axis=-1)
    local[sample['obstacle']] = [.35,.5,.7]
    axes[0].imshow(local,origin='upper',interpolation='nearest')
    axes[0].set(title=f'Observed response ({sample["window"][2]} x {sample["window"][2]})\nBlue: local obstacle',xlabel='Local x',ylabel='Local y')
    for ax,scale,title in [(axes[1],norm,'Probability per source pixel'),
                           (axes[2],LogNorm(max(low,np.finfo(float).tiny),high if high>low else high*1.01),'Same probabilities, log color scale')]:
        im=ax.imshow(np.ma.array(probability,mask=~valid),origin='upper',extent=(0,world,world,0),
                     cmap=cmap,norm=scale,interpolation='nearest')
        x,y,size=sample['window']
        ax.add_patch(Rectangle((x,y),size,size,fill=False,edgecolor='cyan',linewidth=1))
        ax.scatter([truth[0]],[truth[1]],marker='x',color='lime',s=35,linewidths=1.5,label='True source (evaluation only)')
        ax.set(title=title,xlabel='Source world x',ylabel='Source world y')
        ax.legend(loc='lower left',fontsize=7)
        fig.colorbar(im,ax=ax,shrink=.8,label='Probability mass / 1x1 source cell')
    fig.suptitle(f'Scene {sample["scene_id"]}, view {sample["view_id"]} | {world}x{world} actual model queries | sum(p)=1\nGray: excluded support; cyan: observation window | {metadata["support"]}',fontsize=11)
    fig.savefig(output.with_suffix('.png'),dpi=180)
    plt.close(fig)
    # Standalone local viewer. The tooltip reads numeric probability, not color.
    image_b64=base64.b64encode(output.with_suffix('.pixels.png').read_bytes()).decode()
    p_b64=base64.b64encode(probability.astype('<f8').tobytes()).decode()
    valid_b64=base64.b64encode(valid.astype(np.uint8).tobytes()).decode()
    page='''<!doctype html><meta charset="utf-8"><title>Pixel probability map</title>
<style>body{font:16px system-ui;background:#111827;color:#e5e7eb;margin:24px}a{color:#67e8f9}#readout{position:sticky;top:0;background:#111827;padding:12px;z-index:2;font-family:monospace}img{image-rendering:pixelated;max-width:none;display:block}button{padding:8px 14px;margin:8px}p{max-width:950px}</style>
<h2>SCENE_TITLE</h2><p>Each image pixel is one independently scored source location (x+0.5,y+0.5). Gray pixels are excluded. Move the pointer to read the actual normalized probability. The model was trained on a coarser grid; this is dense student inference, not a dense physical teacher.</p>
<p><a href="PANEL_FILE">Observation + probability + color scales</a> · <a href="NPZ_FILE">Download numeric arrays (.npz)</a></p>
<button onclick="zoom(.5)">50%</button><button onclick="zoom(1)">100%</button><button onclick="zoom(2)">200%</button>
<div id="readout">Hover over the map to inspect a pixel.</div><img id="map" src="data:image/png;base64,IMAGE_B64">
<script>
const size=WORLD_SIZE;
function bytes(s){return Uint8Array.from(atob(s),c=>c.charCodeAt(0));}
const p=new DataView(bytes('PROB_B64').buffer),valid=bytes('VALID_B64');
const img=document.getElementById('map'),readout=document.getElementById('readout');
function zoom(z){img.style.width=(size*z)+'px';}
img.addEventListener('mousemove',e=>{const r=img.getBoundingClientRect();const x=Math.floor((e.clientX-r.left)*size/r.width),y=Math.floor((e.clientY-r.top)*size/r.height);if(x<0||y<0||x>=size||y>=size)return;const i=y*size+x;readout.textContent=`pixel [row=${y}, col=${x}] | source (${x+.5}, ${y+.5}) | p=${p.getFloat64(i*8,true).toExponential(8)} | ${valid[i]?'valid':'excluded'}`;});
zoom(1);
</script>'''
    for key,value in {'SCENE_TITLE':f'Scene {sample["scene_id"]}, view {sample["view_id"]}: {world} x {world} probability map',
                      'PANEL_FILE':output.with_suffix('.png').name,'NPZ_FILE':output.with_suffix('.npz').name,
                      'IMAGE_B64':image_b64,'WORLD_SIZE':str(world),'PROB_B64':p_b64,'VALID_B64':valid_b64}.items():
        page=page.replace(key,value)
    output.with_suffix('.html').write_text(page)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=PROJECT/'outputs/experiments/train5000')
    parser.add_argument('--variant',default='response')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--indices',type=int,nargs='+',default=[0,1,2,3],help='Indices in the saved test manifest')
    parser.add_argument('--chunk',type=int,default=4096)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--support',choices=['geometry','world'],default=None,help='Defaults to checkpoint normalization support')
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    protocol=json.loads((args.root/'protocol.json').read_text())
    dataset=Phase1Dataset(PROJECT/protocol['config']['data_root'])
    run=args.root/args.variant/f'seed_{args.seed}'
    checkpoint=run/'best.pt'
    model,_=load_model(checkpoint)
    args.support=args.support or model.normalization_support
    paths=json.loads((args.root/args.variant/'records.json').read_text())['test']
    world=int(dataset.manifest['global_size'])
    directory=run/'dense'
    directory.mkdir(parents=True,exist_ok=True)
    links=[]
    for index in args.indices:
        if not 0 <= index < len(paths): parser.error('test index out of bounds')
        with np.load(args.root/args.variant/paths[index],allow_pickle=False) as record:
            scene,view=int(record['scene_id']),int(record['view_id'])
        sample=dataset.get_observation(scene,view)
        print(f'Dense query: test index {index}, scene {scene}, view {view}, size {sample["window"][2]}',flush=True)
        energy=score_pixels(model,sample,world,args.chunk)
        valid=np.ones((world,world),dtype=bool)
        if args.support=='geometry':
            valid &= ~dataset.get_region(scene,0,0,world)['obstacle']
        x,y,size=sample['window']
        valid[y:y+size,x:x+size]=False
        probability=pixel_probability(energy,valid)
        metadata=dict(scene_id=scene,view_id=view,test_index=index,resolution=[world,world],
            pixel_centers='candidate_xy=(column+0.5,row+0.5)',probability_semantics='discrete probability per 1x1 source cell',
            support='world minus window and geometry mask (evaluation-only)' if args.support=='geometry' else 'world minus window (no hidden geometry mask)',
            normalization='one global softmax over valid pixels; no interpolation or area weighting',
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            trained_support=model.normalization_support,use_obstacle=model.use_obstacle,
            training_spacing=protocol['config']['teacher']['spacing'],valid_pixels=int(valid.sum()),
            probability_sum=float(probability.sum()),teacher_dense_available=False)
        output=directory/f'example_{index}_{args.support}'
        truth=dataset.get_scene(scene)['drivers'][0]
        np.savez_compressed(output.with_suffix('.npz'),energy=energy,student_prob=probability,valid=valid,
                            response=sample['response'],obstacle=sample['obstacle'],window=sample['window'],
                            true_source_xy=truth,metadata_json=json.dumps(metadata))
        output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
        save_visuals(output,sample,probability,valid,truth,metadata)
        links.append(f'<li><a href="{output.name}.html">Test {index}: scene {scene}, view {view}, size {size}</a> · <a href="{output.name}.png">comparison PNG</a></li>')
        print(f'Saved {output.with_suffix(".html")}',flush=True)
    (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Dense probability maps</title><h1>Dense source probability maps</h1><ul>'+''.join(links)+'</ul>')


if __name__=='__main__':main()
