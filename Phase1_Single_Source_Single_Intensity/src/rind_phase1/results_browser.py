"""Read-only test-set explorer with actual dense inference and saved teacher samples."""
import argparse
import base64
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
import torch
from .data import PROJECT_ROOT, Phase1Dataset
from .train import load_model
from .teacher import load_teacher_record
from .predict import predict
from .sampling import score_grid, probability_from_energy, sampling_report
from .search import CandidateDomain
import math


def geometry_support(xy, types, params, count, world):
    """Vectorized closed-shape membership, checked against RIND scalar geometry."""
    x, y = np.asarray(xy, dtype=np.float64).T
    valid = (x >= 0) & (y >= 0) & (x <= world) & (y <= world)
    def cross(ax, ay, bx, by):
        return (bx-ax)*(y-ay)-(by-ay)*(x-ax)
    for typ, p in zip(types[:count], params[:count]):
        if typ in (0, 1):
            c, s = math.cos(p[4]), math.sin(p[4])
            u, v = (x-p[0])*c+(y-p[1])*s, -(x-p[0])*s+(y-p[1])*c
            inside = (np.abs(u) <= p[2]) & (np.abs(v) <= p[3]) if typ == 0 else (u/p[2])**2+(v/p[3])**2 <= 1
        elif typ == 2:
            a, b, c = cross(*p[:4]), cross(*p[2:6]), cross(p[4], p[5], p[0], p[1])
            inside = ~(((a < 0) | (b < 0) | (c < 0)) & ((a > 0) | (b > 0) | (c > 0)))
        elif typ == 3:
            n = int(round(p[0]))
            inside, boundary = np.zeros(len(x), bool), np.zeros(len(x), bool)
            vertices = p[1:1+2*n].reshape(n, 2)
            for i in range(n):
                ax, ay = vertices[i]
                bx, by = vertices[(i+1) % n]
                boundary |= (cross(ax, ay, bx, by) == 0) & (x >= min(ax, bx)) & (x <= max(ax, bx)) & (y >= min(ay, by)) & (y <= max(ay, by))
                if ay != by:
                    inside ^= ((ay > y) != (by > y)) & (x < ax+(y-ay)*(bx-ax)/(by-ay))
            inside |= boundary
        else:
            raise ValueError(f'Unknown obstacle type {typ}')
        valid &= ~inside
    return valid


def pack(array, dtype='<f8'):
    a = np.asarray(array, dtype=dtype)
    return dict(shape=list(a.shape), dtype=dtype, data=base64.b64encode(a.tobytes()).decode())


def read(path):
    return json.loads(path.read_text())


class Explorer:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rind-inference')
        self.jobs = OrderedDict()
        self.lock = threading.Lock()
        self.models = OrderedDict()
        self.catalogs = {}
        self.datasets = {}

    def runs(self):
        result = []
        for path in sorted(self.root.glob('*/protocol.json')):
            variants = {}
            for folder in sorted(path.parent.iterdir()):
                if folder.is_dir() and (folder/'records.json').is_file():
                    seeds = [int(p.parent.name[5:]) for p in folder.glob('seed_*/best.pt')
                             if p.parent.name[5:].isdigit()]
                    if seeds:
                        variants[folder.name] = sorted(seeds)
            if variants:
                result.append(dict(name=path.parent.name, variants=variants))
        return result

    def selection(self, args):
        choices = {r['name']: r for r in self.runs()}
        name, variant = args.get('run'), args.get('variant', 'response')
        seed = int(args.get('seed', 0))
        if name not in choices or variant not in choices[name]['variants'] or seed not in choices[name]['variants'][variant]:
            raise ValueError('请选择已发现的实验、方法和 seed')
        root = self.root/name
        protocol = read(root/'protocol.json')
        directory = root/variant
        checkpoint = directory/f'seed_{seed}'/'best.pt'
        manifest = read(directory/'records.json')
        paths = [(directory/p).resolve() for p in manifest['test']]
        if any(not p.is_relative_to(directory.resolve()) for p in paths):
            raise ValueError('Teacher record outside experiment directory')
        return protocol, directory, checkpoint, paths

    def dataset(self, protocol):
        path = (PROJECT_ROOT/protocol['config']['data_root']).resolve()
        if path not in self.datasets:
            self.datasets[path] = Phase1Dataset(path)
        return self.datasets[path]

    def catalog(self, args):
        protocol, directory, checkpoint, paths = self.selection(args)
        key = (str(directory), (directory/'records.json').stat().st_mtime_ns)
        if key not in self.catalogs:
            dataset = self.dataset(protocol)
            test_ids = set(protocol['scene_splits']['test'])
            entries = []
            for index, path in enumerate(paths):
                with np.load(path, allow_pickle=False) as z:
                    scene, view = int(z['scene_id']), int(z['view_id'])
                    window = z['window'].tolist()
                if scene not in test_ids:
                    raise ValueError('Non-test scene found in test manifest')
                observation = dataset.get_observation(scene, view)
                if window != observation['window'].tolist():
                    raise ValueError('Teacher window mismatch')
                response = observation['response']
                entries.append(dict(index=index, scene_id=scene, view_id=view, window=window,
                                    nonuniform=bool(np.any(response != response.flat[0])),
                                    response_mean=float(response.mean())))
            self.catalogs[key] = entries
        c = protocol['config']
        training = c['training']
        return dict(entries=self.catalogs[key], protocol_version=c.get('protocol_version', 1),
                    teacher_spacing=c['teacher']['spacing'],
                    sampling=sampling_report(int(self.dataset(protocol).manifest['global_size']), c['teacher']['spacing'], training['fourier_frequencies']),
                    trained_support=training.get('normalization_support', 'geometry'),
                    use_obstacle=training.get('use_obstacle', True), checkpoint=str(checkpoint.relative_to(self.root)))

    def submit(self, args):
        _, _, _, paths = self.selection(args)
        index = int(args.get('index', 0))
        if not 0 <= index < len(paths):
            raise ValueError('Test index out of range')
        spacing = int(args.get('spacing', 1))
        if spacing not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError('spacing must be 1, 2, 4, 8, 16, 32 or 64')
        if args.get('support', 'world') not in ('world', 'geometry') or args.get('mode', 'dense') not in ('dense', 'matched'):
            raise ValueError('Unknown normalization support or display mode')
        with self.lock:
            if sum(j['status'] in ('queued', 'running') for j in self.jobs.values()) >= 3:
                raise ValueError('已有推理任务，请等当前任务完成')
            while len(self.jobs) >= 6:
                victim = next((k for k, j in self.jobs.items() if j['status'] not in ('queued', 'running')), None)
                if victim is None:
                    break
                del self.jobs[victim]
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = dict(status='queued')
        self.executor.submit(self.execute, job_id, dict(args, index=index, spacing=spacing))
        return job_id

    def execute(self, job_id, args):
        with self.lock:
            self.jobs[job_id]['status'] = 'running'
        try:
            result, arrays = self.compute(args)
            with self.lock:
                self.jobs[job_id].update(status='done', result=result, arrays=arrays)
        except Exception as exc:
            with self.lock:
                self.jobs[job_id].update(status='error', error=str(exc))

    def compute(self, args):
        started = time.monotonic()
        protocol, directory, checkpoint, paths = self.selection(args)
        dataset = self.dataset(protocol)
        entry = self.catalog(args)['entries'][args['index']]
        teacher = load_teacher_record(paths[args['index']])
        sample = dataset.get_observation(entry['scene_id'], entry['view_id'])
        key = (str(checkpoint), checkpoint.stat().st_mtime_ns, checkpoint.stat().st_size)
        if key not in self.models:
            self.models[key] = load_model(checkpoint)
            while len(self.models) > 2:
                self.models.popitem(last=False)
        model, cp = self.models[key]
        if cp.get('scene_splits') and entry['scene_id'] not in cp['scene_splits']['test']:
            raise ValueError('Checkpoint test split mismatch')
        support_policy = args.get('support', 'world')
        mode, spacing = args.get('mode', 'dense'), args['spacing']
        world = int(dataset.manifest['global_size'])
        prediction = predict(model, teacher, sample, chunk=256, normalization_support=support_policy)
        from part_e.evaluate import evaluate_record
        metrics = evaluate_record(teacher, prediction, threshold=protocol['config']['evaluation']['compatibility_threshold'],
                                  spacing=protocol['config']['teacher']['spacing'])
        teacher_shape = (len(teacher['y_axis']), len(teacher['x_axis']))
        def raster(values, fill=np.nan):
            array = np.full(teacher_shape, fill, dtype=np.float64)
            array.flat[teacher['grid_index']] = values
            return array
        teacher_probability = raster(teacher['teacher_prob'])
        teacher_cost = raster(np.where(teacher['valid'], teacher['physical_cost'], np.nan))
        if mode == 'matched':
            energy = raster(prediction['energy'])
            support = raster(prediction['support'], fill=0).astype(bool)
            probability = raster(prediction['student_prob'], fill=0)
            spacing = protocol['config']['teacher']['spacing']
            count = len(teacher['candidate_xy'])
        else:
            energy, xy = score_grid(model, sample, world, spacing=spacing, chunk=256)
            domain_settings = teacher['metadata']['settings']['support']
            domain = CandidateDomain(tuple(sample['window']), world, domain_settings.get('quadtree_prior', False))
            support = domain.contains(xy)
            if support_policy == 'geometry':
                scene = dataset.get_scene(entry['scene_id'])
                support &= geometry_support(xy, scene['obstacle_types'], scene['obstacle_params'], int(scene['obstacle_count']), world)
            support = support.reshape(energy.shape)
            probability = probability_from_energy(energy, support)
            count = len(xy)
        truth = dataset.get_scene(entry['scene_id'])['drivers'][0, :2]
        meta = dict(run=args['run'], variant=args.get('variant', 'response'), seed=int(args.get('seed', 0)),
                    index=args['index'], scene_id=entry['scene_id'], view_id=entry['view_id'], window=sample['window'].tolist(),
                    world_size=world, source_xy=truth.tolist(), mode=mode, spacing=spacing,
                    teacher_spacing=protocol['config']['teacher']['spacing'],
                    teacher_candidates=len(teacher['candidate_xy']), teacher_valid=int(teacher['valid'].sum()),
                    model_queries=count, normalization_support=support_policy,
                    trained_support=model.normalization_support, use_obstacle=model.use_obstacle,
                    checkpoint=str(checkpoint.relative_to(self.root)), checkpoint_epoch=int(cp['epoch']),
                    probability_sum=float(probability.sum()), seconds=time.monotonic()-started,
                    metrics=metrics, sampling=self.catalog(args)['sampling'],
                    note='Metrics use exactly the saved teacher candidates. Dense scores are fresh model queries; the teacher is not interpolated. No area weighting.')
        arrays = dict(probability=probability, energy=energy, support=support, response=sample['response'],
                      window=sample['window'], source_xy=truth, teacher_candidate_xy=teacher['candidate_xy'],
                      teacher_probability=teacher['teacher_prob'], teacher_cost=teacher['physical_cost'],
                      teacher_valid=teacher['valid'], student_on_teacher=prediction['student_prob'],
                      metadata=np.array(json.dumps(meta, ensure_ascii=False, allow_nan=False)))
        result = dict(metadata=meta, student=pack(probability), support=pack(support, '|u1'),
                      response=pack(sample['response']), teacher=pack(teacher_probability), cost=pack(teacher_cost))
        return result, arrays


def create_app(root=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 8192
    explorer = Explorer(root or PROJECT_ROOT/'outputs/experiments')
    app.extensions['explorer'] = explorer

    @app.errorhandler(ValueError)
    @app.errorhandler(FileNotFoundError)
    def invalid(exc):
        return jsonify(error=str(exc)), 400

    @app.get('/')
    def home():
        return render_template('results.html')

    @app.get('/api/runs')
    def runs():
        return jsonify(explorer.runs())

    @app.get('/api/catalog')
    def catalog():
        return jsonify(explorer.catalog(request.args))

    @app.post('/api/jobs')
    def submit():
        if request.headers.get('Origin') and request.headers['Origin'] != request.host_url.rstrip('/'):
            return jsonify(error='Cross-origin inference requests are not allowed'), 403
        payload = request.get_json()
        if not isinstance(payload, dict):
            raise ValueError('JSON object required')
        return jsonify(id=explorer.submit(payload)), 202

    @app.get('/api/jobs/<job_id>')
    def job(job_id):
        with explorer.lock:
            item = explorer.jobs.get(job_id)
            if item is None:
                return jsonify(error='Result expired; rerun inference'), 404
            return jsonify({k: v for k, v in item.items() if k != 'arrays'})

    @app.get('/api/jobs/<job_id>/download')
    def download(job_id):
        with explorer.lock:
            item = explorer.jobs.get(job_id)
            if item is None or item['status'] != 'done':
                return jsonify(error='Result not available'), 404
            arrays = item['arrays']
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        buf.seek(0)
        return send_file(buf, mimetype='application/octet-stream', as_attachment=True, download_name=f'rind_{job_id[:8]}.npz')

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT/'outputs/experiments')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    torch.set_num_threads(args.threads)
    print(f'RIND results: http://127.0.0.1:{args.port}', flush=True)
    create_app(args.root).run(host='127.0.0.1', port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
