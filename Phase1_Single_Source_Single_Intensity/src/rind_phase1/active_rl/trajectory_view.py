"""Self-contained evaluation-only viewer for acquired tiles and raw/min fields."""
import base64
import json
from pathlib import Path
import numpy as np


def encode(values, dtype):
    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode('ascii')


def save_view(path, row, fused, raw, patches, source, world, tile, grid, fusion, informative):
    if len(row['history']) != len(fused) or np.shape(raw) != np.shape(fused):
        raise ValueError('trajectory fields and acquisition history must align')
    metadata = dict(scene=row['scene_id'], policy=row['policy'], world=world, tile=tile, grid=grid,
                    source=np.asarray(source).tolist(), history=row['history'], curve=row['curve'],
                    tiles=row['seen_tiles'], informative=informative, fusion=fusion,
                    fused=encode(fused, '<f4'), raw=encode(raw, '<f4'), patches=encode(patches, 'u1'))
    payload = json.dumps(metadata, separators=(',', ':'), allow_nan=False)
    page = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>RIND · 逐块观测与概率融合</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#eef3f7;color:#18344b;font:15px system-ui,sans-serif}
main{max-width:1800px;margin:auto;padding:24px}h1{font-size:25px;margin:0 0 10px}.muted{color:#597083;line-height:1.7}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;background:white;padding:16px;border-radius:12px;margin:20px 0}
button{border:0;border-radius:7px;padding:9px 15px;background:#235e87;color:white;cursor:pointer}input[type=range]{flex:1;min-width:240px}
.panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}.card{background:white;border:1px solid #d3e0e9;border-radius:12px;padding:15px}
h2{font-size:18px;margin:0 0 8px}canvas{width:100%;height:auto;aspect-ratio:1;display:block;border:1px solid #d5e0e9;image-rendering:pixelated}
.caption{font-size:13px;min-height:42px;color:#60788a;line-height:1.6}.status{padding:15px;background:white;border-radius:10px;margin-top:16px;line-height:1.8}
.bar{height:9px;background:linear-gradient(to right,#000004,#420a68,#932667,#dd513a,#fca50a,#fcffa4);margin:9px 0 4px}
@media(max-width:1100px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style><main><h1 id="title"></h1><div class="muted">每次只读取一个完整小窗。灰色为未观测区域，青线为采集顺序；源位置仅用于评价标注，不参与选路。此页面可直接离线打开。</div>
<div class="controls"><button id="prev">上一步</button><button id="play">播放</button><input id="step" type="range" min="0" value="0"><button id="next">下一步</button><span id="counter"></span><label><input id="truth" type="checkbox" checked>真源标注</label><label><input id="log" type="checkbox">对数色阶</label></div>
<div class="panels">
<div class="card"><h2>累计已观测区域</h2><p class="caption">只显示当前步及之前的测量，不展示未来响应。</p><canvas id="observed" width="512" height="512"></canvas></div>
<div class="card"><h2>本次16×16响应</h2><p class="caption" id="tilelabel">黑=0，白=1。</p><canvas id="patch" width="256" height="256"></canvas></div>
<div class="card"><h2>本窗原始位置概率</h2><p class="caption">当前块单独调用冻结Phase1得到的全场预测。</p><canvas id="raw" width="512" height="512"></canvas><div class="bar"></div><small id="rawscale"></small></div>
<div class="card"><h2>累计融合位置概率</h2><p class="caption" id="fusionlabel"></p><canvas id="fused" width="512" height="512"></canvas><div class="bar"></div><small id="fusedscale"></small></div>
</div><div class="status" id="status"></div><p class="muted">两张概率图使用相同色阶范围。接近黑色只表示相对低值，不能自动解释为物理上不可能；归一化后的最小值共识分布也不是精确贝叶斯后验。</p></main>
<script id="trajectory-data" type="application/json">__PAYLOAD__</script>
<script>
const d=JSON.parse(document.getElementById('trajectory-data').textContent);
function bytes(s){return Uint8Array.from(atob(s),c=>c.charCodeAt(0))}function floats(s){const a=bytes(s);return new Float32Array(a.buffer)}
const fused=floats(d.fused),raw=floats(d.raw),patches=bytes(d.patches),N=d.grid*d.grid,byTile=new Map(d.tiles.map((x,i)=>[x,i]));
const $=x=>document.getElementById(x),slider=$('step');slider.max=d.history.length-1;
$('title').textContent=`Scene ${d.scene} · ${d.policy} · ${d.tile}×${d.tile}逐块扩展`;
$('fusionlabel').textContent=d.fusion==='minimum'?'各有效窗的原始概率逐点取最小值，再统一归一化。':'各有效窗的折扣乘积，再统一归一化。';
const palette=[[0,0,4],[66,10,104],[147,38,103],[221,81,58],[252,165,10],[252,255,164]];
function color(v){v=Math.max(0,Math.min(1,v));const z=v*(palette.length-1),i=Math.min(palette.length-2,Math.floor(z)),f=z-i;return palette[i].map((a,k)=>Math.round(a*(1-f)+palette[i+1][k]*f))}
function marker(ctx,side){if(!$('truth').checked)return;const x=d.source[0]/d.world*side,y=d.source[1]/d.world*side;ctx.strokeStyle='#13ea49';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x-6,y);ctx.lineTo(x+6,y);ctx.moveTo(x,y-6);ctx.lineTo(x,y+6);ctx.stroke()}
function drawField(id,values,max){const c=$(id),ctx=c.getContext('2d'),small=document.createElement('canvas');small.width=small.height=d.grid;const s=small.getContext('2d'),im=s.createImageData(d.grid,d.grid);for(let i=0;i<N;i++){let v=values[i]/max;if($('log').checked)v=Math.log1p(v*999)/Math.log(1000);const rgb=color(v);im.data.set([...rgb,255],4*i)}s.putImageData(im,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(small,0,0,c.width,c.height);marker(ctx,c.width)}
function patchImage(tile){const small=document.createElement('canvas');small.width=small.height=d.tile;const ctx=small.getContext('2d'),im=ctx.createImageData(d.tile,d.tile),j=byTile.get(tile);for(let i=0;i<d.tile*d.tile;i++){const v=patches[j*d.tile*d.tile+i]*255;im.data.set([v,v,v,255],4*i)}ctx.putImageData(im,0,0);return small}
const patchCanvases=new Map(d.tiles.map(t=>[t,patchImage(t)]));
function draw(){const k=+slider.value,h=d.history[k],row=d.curve[k],c=$('observed'),ctx=c.getContext('2d'),scale=c.width/d.world;ctx.fillStyle='#b9c7d4';ctx.fillRect(0,0,c.width,c.height);ctx.imageSmoothingEnabled=false;const centers=[];for(let i=0;i<=k;i++){const w=d.history[i].window;if(!w)continue;const tile=Math.floor(w[1]/d.tile)*(d.world/d.tile)+Math.floor(w[0]/d.tile);ctx.drawImage(patchCanvases.get(tile),w[0]*scale,w[1]*scale,d.tile*scale,d.tile*scale);centers.push([(w[0]+d.tile/2)*scale,(w[1]+d.tile/2)*scale])}ctx.strokeStyle='#00d9ee';ctx.lineWidth=1;ctx.beginPath();centers.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();if(h.window){const [x,y]=h.window;ctx.strokeStyle='#00d9ee';ctx.lineWidth=2;ctx.strokeRect(x*scale,y*scale,d.tile*scale,d.tile*scale)}marker(ctx,c.width);
const pc=$('patch'),pctx=pc.getContext('2d');pctx.imageSmoothingEnabled=false;pctx.fillStyle='#b9c7d4';pctx.fillRect(0,0,pc.width,pc.height);if(h.window){const tile=h.window[1]/d.tile*(d.world/d.tile)+h.window[0]/d.tile;pctx.drawImage(patchCanvases.get(tile),0,0,pc.width,pc.height)}
$('tilelabel').textContent=!h.window?'本步停止，没有新增测量。':`${h.window[0]}, ${h.window[1]}；黑=0，白=1。`+(!d.informative[k]?'全亮/全暗：计入预算，作为中性证据。':'');
const rp=raw.subarray(k*N,(k+1)*N),fp=fused.subarray(k*N,(k+1)*N);let max=0;for(let i=0;i<N;i++)max=Math.max(max,rp[i],fp[i]);max=Math.max(max,1e-30);drawField('raw',rp,max);drawField('fused',fp,max);$('rawscale').textContent=$('fusedscale').textContent=`0 → ${max.toPrecision(4)}（每个候选点的概率质量）`;
$('counter').textContent=`步骤 ${k}/${slider.max}`;$('status').textContent=`累计新增像素：${row.spent.toLocaleString()}；动作：${h.action}；M16=${(row.mass16*100).toFixed(2)}%，M32=${(row.mass32*100).toFixed(2)}%；最高概率点距真源 ${row.map_distance.toFixed(1)} px。后三项仅用于离线评价。`}
let timer=null;$('prev').onclick=()=>{slider.value=Math.max(0,+slider.value-1);draw()};$('next').onclick=()=>{slider.value=Math.min(+slider.max,+slider.value+1);draw()};$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放';return}if(+slider.value===+slider.max)slider.value=0;$('play').textContent='暂停';timer=setInterval(()=>{if(+slider.value>=+slider.max){clearInterval(timer);timer=null;$('play').textContent='播放';return}slider.value=+slider.value+1;draw()},250)};slider.oninput=draw;$('truth').onchange=draw;$('log').onchange=draw;draw();
</script></html>'''
    Path(path).write_text(page.replace('__PAYLOAD__', payload))
