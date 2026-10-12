"""Interactive browser for trained Phase-II layered combined-field experiments."""
from __future__ import annotations

import argparse
import base64
import json
from collections import OrderedDict
from pathlib import Path
import threading

import numpy as np
from flask import Flask,jsonify,render_template,request

from .data import PROJECT_ROOT,Phase2Dataset


def _read(path):return json.loads(Path(path).read_text())


def _pack(array,dtype='<f8'):
    value=np.asarray(array,dtype=dtype)
    return {'shape':list(value.shape),'dtype':dtype,
            'data':base64.b64encode(value.tobytes()).decode()}


def _distribution_metrics(q,p):
    q=np.asarray(q,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    active=q>0;mixture=.5*(q+p);p_active=p>0
    cross_entropy=float(-np.sum(q[active]*np.log(p[active])))
    entropy=float(-np.sum(q[active]*np.log(q[active])))
    js=.5*float(np.sum(q[active]*np.log(q[active]/mixture[active])))
    js+=.5*float(np.sum(p[p_active]*np.log(p[p_active]/mixture[p_active])))
    tv=.5*float(np.abs(q-p).sum())
    return {'cross_entropy':cross_entropy,'teacher_entropy':entropy,
            'forward_kl':cross_entropy-entropy,'js_divergence':js,
            'total_variation':tv,'overlap':1-tv}


class Explorer:
    def __init__(self,root):
        self.root=Path(root).resolve();self.datasets={};self.models=OrderedDict();self.lock=threading.Lock()

    def runs(self):
        result=[]
        for protocol in sorted(self.root.glob('*/protocol.json')):
            directory=protocol.parent
            if not (directory/'records.json').is_file():continue
            seeds=sorted(int(path.parent.name[5:]) for path in directory.glob('models/seed_*/best.pt')
                         if path.parent.name[5:].isdigit())
            if seeds:result.append({'name':directory.name,'seeds':seeds})
        return result

    def selection(self,args):
        run=str(args.get('run',''));seed=int(args.get('seed',0))
        choices={item['name']:item for item in self.runs()}
        if run not in choices or seed not in choices[run]['seeds']:
            raise ValueError('请选择已完成的实验和 seed')
        root=(self.root/run).resolve()
        if not root.is_relative_to(self.root):raise ValueError('实验路径越界')
        return root,seed,_read(root/'protocol.json'),_read(root/'records.json')

    def dataset(self,protocol):
        data=Path(protocol['config']['data_root'])
        if not data.is_absolute():data=PROJECT_ROOT/data
        data=data.resolve()
        if data not in self.datasets:self.datasets[data]=Phase2Dataset(data)
        return self.datasets[data]

    def model(self,checkpoint):
        checkpoint=Path(checkpoint);key=(str(checkpoint),checkpoint.stat().st_mtime_ns,checkpoint.stat().st_size)
        with self.lock:
            if key not in self.models:
                from rind_phase1.train import load_model
                self.models[key]=load_model(checkpoint)
                while len(self.models)>2:self.models.popitem(last=False)
            return self.models[key]

    def catalog(self,args):
        root,seed,protocol,manifest=self.selection(args);dataset=self.dataset(protocol)
        test_scenes=set(protocol['scene_splits']['test']);entries=[]
        for index,row in enumerate(manifest['records']['test']):
            scene=int(row['scene_id']);view=int(row['view_id'])
            if scene not in test_scenes:raise ValueError('测试记录包含非测试 scene')
            observation=dataset.get_observation(scene,view);free=observation['response'][~observation['obstacle']]
            entries.append({'index':index,'scene_id':scene,'view_id':view,
                            'size':int(row['size']),'layers':int(row['layers_recovered']),
                            'response_min':float(free.min(initial=0)),'response_max':float(free.max(initial=0))})
        checkpoint=root/'models'/f'seed_{seed}'/'best.pt'
        return {'entries':entries,'checkpoint':str(checkpoint.relative_to(self.root)),
                'checkpoint_epoch':int(self.model(checkpoint)[1]['epoch']),
                'teacher_spacing':protocol['config']['teacher']['spacing'],
                'target_semantics':protocol.get('teacher_target','saved combined layer field'),
                'note':'This is one full-field contribution-location distribution, not a joint multi-source posterior.'}

    @staticmethod
    def raster(values,grid_index,shape,fill=np.nan):
        image=np.full(shape,fill,dtype=np.float64);image.flat[grid_index]=values;return image

    def result(self,args):
        root,seed,protocol,manifest=self.selection(args);rows=manifest['records']['test']
        index=int(args.get('index',0))
        if not 0<=index<len(rows):raise ValueError('测试视窗 index 越界')
        row=rows[index];dataset=self.dataset(protocol);scene=int(row['scene_id']);view=int(row['view_id'])
        observation=dataset.get_observation(scene,view);run_dir=root/row['directory'];report=_read(run_dir/'report.json')
        if report['status']!='complete':raise ValueError('浏览器只显示完成的教师记录')
        with np.load(run_dir/'combined.npz',allow_pickle=False) as archive:
            candidate=archive['candidate_xy'];teacher=archive['probability'];valid=archive['valid'].astype(bool)
            grid_index=archive['grid_index'];x_axis=archive['x_axis'];y_axis=archive['y_axis']
        model,checkpoint=self.model(root/'models'/f'seed_{seed}'/'best.pt')
        support=valid if model.normalization_support=='geometry' else np.ones_like(valid)
        import torch
        with torch.inference_mode():
            prediction=model.predict(torch.as_tensor(observation['response'][None],dtype=torch.float32),
                torch.as_tensor(observation['obstacle'][None],dtype=torch.bool),
                torch.as_tensor(observation['window'][None],dtype=torch.float32),
                torch.as_tensor(candidate[None],dtype=torch.float32),
                torch.as_tensor(support[None],dtype=torch.bool))['student_prob'][0].numpy()
        shape=(len(y_axis),len(x_axis));truth=dataset.get_source_params(scene)
        layers=[{'step':int(layer['step']),'intensity':float(layer['intensity']),
                 'minimum_physical_cost':layer.get('minimum_physical_cost'),
                 'exact_grid_matches':layer.get('exact_grid_matches'),
                 'conflicts':len(layer.get('trace',{}).get('conflict',[]))}
                for layer in report.get('layers',[])]
        return {'metadata':{'run':root.name,'seed':seed,'index':index,'scene_id':scene,'view_id':view,
                            'window':observation['window'].tolist(),'world_size':int(dataset.manifest['global_size']),
                            'checkpoint_epoch':int(checkpoint['epoch']),'layers':layers,
                            'shadow_candidate_screening':report.get('shadow_candidate_screening'),
                            'uniform_candidate_sampling_passes':report.get('uniform_candidate_sampling_passes'),
                            'uniform_candidate_positions':report.get('uniform_candidate_positions'),
                            'layer_probability_assignments':report.get('layer_probability_assignments'),
                            'layer_candidate_score_comparisons':report.get('layer_candidate_score_comparisons'),
                            'candidate_evaluations':report.get('candidate_evaluations'),
                            'candidate_score_evaluations_without_shadow_screen':report.get('candidate_score_evaluations_without_shadow_screen'),
                            'candidate_score_evaluations_saved_by_shadow_screen':report.get('candidate_score_evaluations_saved_by_shadow_screen'),
                            'metrics':_distribution_metrics(teacher,prediction),
                            'teacher_probability_sum':float(teacher.sum()),
                            'student_probability_sum':float(prediction.sum()),
                            'semantics':protocol.get('teacher_target','saved combined layer field')+'; not a joint posterior'},
                'response':_pack(observation['response']),'obstacle':_pack(observation['obstacle'],'|u1'),
                'teacher':_pack(self.raster(teacher,grid_index,shape)),
                'student':_pack(self.raster(prediction,grid_index,shape)),
                'teacher_valid':_pack(self.raster(valid,grid_index,shape,fill=0),'|u1'),
                'student_support':_pack(self.raster(support,grid_index,shape,fill=0),'|u1'),
                'truth':_pack(truth),'grid_shape':list(shape)}

    def layer(self,args):
        root,_,protocol,manifest=self.selection(args);rows=manifest['records']['test']
        index=int(args.get('index',0));step=int(args.get('layer',1))
        if not 0<=index<len(rows):raise ValueError('测试视窗 index 越界')
        row=rows[index];run_dir=root/row['directory'];report=_read(run_dir/'report.json')
        details={int(item['step']):item for item in report.get('layers',[])}
        if step not in details:raise ValueError('层编号越界')
        with np.load(run_dir/f'layer-{step:02d}.npz',allow_pickle=False) as archive:
            before=archive['residual_before'];after=archive['residual_after']
            mask=archive['layer_response'] if 'layer_response' in archive.files else archive['full_mask']
            probability=archive['probability'];grid_index=archive['grid_index']
            shape=(len(archive['y_axis']),len(archive['x_axis']))
        intensity=float(details[step]['intensity']);contribution=intensity*mask.astype(np.float64)
        if not np.allclose(after,before-contribution,rtol=0,atol=1e-10):
            raise ValueError('保存的逐层残差不满足实际剥离等式')
        dataset=self.dataset(protocol);observation=dataset.get_observation(int(row['scene_id']),int(row['view_id']))
        return {'metadata':{'step':step,'intensity':intensity,
                            'minimum_physical_cost':details[step].get('minimum_physical_cost'),
                            'exact_grid_matches':details[step].get('exact_grid_matches'),
                            'conflicts':len(details[step].get('trace',{}).get('conflict',[]))},
                'before':_pack(before),'mask':_pack(mask,'|u1'),'contribution':_pack(contribution),
                'after':_pack(after),'obstacle':_pack(observation['obstacle'],'|u1'),
                'probability':_pack(self.raster(probability,grid_index,shape))}


def create_app(root=None):
    app=Flask(__name__);explorer=Explorer(root or PROJECT_ROOT/'outputs/experiments')
    app.extensions['explorer']=explorer
    @app.errorhandler(ValueError)
    @app.errorhandler(FileNotFoundError)
    def invalid(error):return jsonify(error=str(error)),400
    @app.get('/')
    def home():return render_template('results.html')
    @app.get('/api/runs')
    def runs():return jsonify(explorer.runs())
    @app.get('/api/catalog')
    def catalog():return jsonify(explorer.catalog(request.args))
    @app.post('/api/result')
    def result():
        payload=request.get_json()
        if not isinstance(payload,dict):raise ValueError('JSON object required')
        return jsonify(explorer.result(payload))
    @app.post('/api/layer')
    def layer():
        payload=request.get_json()
        if not isinstance(payload,dict):raise ValueError('JSON object required')
        return jsonify(explorer.layer(payload))
    return app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=PROJECT_ROOT/'outputs/experiments')
    parser.add_argument('--port',type=int,default=8768);parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.threads<1:parser.error('threads must be positive')
    import torch;torch.set_num_threads(args.threads)
    print(f'RIND Phase II results: http://127.0.0.1:{args.port}',flush=True)
    create_app(args.root).run(host='127.0.0.1',port=args.port,debug=False,threaded=True)


if __name__=='__main__':main()
