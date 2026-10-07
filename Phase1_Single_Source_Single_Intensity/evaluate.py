"""Record-driven Teacher/Student, physical, ambiguity and paired evaluations."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
from . import checks


def entropy(p):
    p = np.asarray(p)
    positive = p > 0
    return float(-np.sum(p[positive] * np.log(p[positive])))


def distributions(q, p):
    positive = q > 0
    singular = bool(np.any(p[positive] == 0))
    kl = None if singular else float(np.sum(q[positive] * np.log(q[positive] / p[positive])))
    m = (q + p) / 2
    def relative(a):
        nz = a > 0
        return float(np.sum(a[nz] * np.log(a[nz] / m[nz])))
    return {'forward_kl': kl, 'forward_kl_infinite': singular,
            'jensen_shannon': (relative(q) + relative(p)) / 2,
            'l1_distance': float(np.abs(q-p).sum()),
            'total_variation': float(np.abs(q-p).sum()/2),
            'teacher_entropy': entropy(q), 'student_entropy': entropy(p),
            'entropy_mismatch': abs(entropy(q)-entropy(p))}


def components(xy, mask, spacing):
    """Four-neighbor lattice components; no continuous topology claim."""
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('spacing must be finite and positive')
    cells = np.rint(xy / spacing - .5).astype(np.int64)
    if not np.allclose((cells+.5)*spacing, xy, atol=1e-7, rtol=0):
        raise ValueError('components require the globally anchored common uniform lattice')
    lookup = {tuple(cell): i for i, cell in enumerate(cells) if mask[i]}
    remaining = set(lookup)
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, indices = [start], []
        while stack:
            x, y = stack.pop()
            indices.append(lookup[(x, y)])
            for nxt in ((x-1,y), (x+1,y), (x,y-1), (x,y+1)):
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
        groups.append(indices)
    return groups


def evaluate_record(record, prediction=None, *, threshold, spacing=None):
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('compatibility threshold must be explicit, finite and nonnegative')
    xy, valid, cost, q = checks.teacher(record)
    low = valid & (cost <= threshold)
    metadata = record.get('metadata', {})
    result = {'scene_id': record['scene_id'], 'view_id': record['view_id'],
              'window_size': int(record['window'][2]), 'config_id': record['config_id'],
              'support_policy': metadata.get('settings', {}).get('support', {}),
              'threshold': threshold, 'valid_candidates': int(valid.sum()),
              'compatible_candidates': int(low.sum()), 'exact_candidates': int((valid & (cost == 0)).sum()),
              'teacher_entropy': entropy(q), 'effective_candidates': float(np.exp(entropy(q))),
              'teacher_expected_physical_cost': float(q[valid] @ cost[valid]),
              'teacher_compatible_mass': float(q[low].sum()),
              'teacher_only': prediction is None,
              'physical_evaluations': metadata.get('num_physics_evaluations'),
              'search_seconds': metadata.get('elapsed_seconds')}
    if spacing is not None:
        groups = components(xy, low, spacing)
        result.update(sampled_compatible_components=len(groups),
                      component_teacher_mass=[float(q[g].sum()) for g in groups],
                      component_cell_counts=[len(g) for g in groups],
                      candidate_spacing=spacing)
    if prediction is not None:
        checks.aligned(record, prediction)
        p = checks.probability(prediction['student_prob'], valid, 'student_prob')
        result.update(distributions(q, p))
        best = int(np.argmax(p))
        result.update(student_expected_physical_cost=float(p[valid] @ cost[valid]),
                      student_compatible_mass=float(p[low].sum()),
                      student_top1_physical_cost=float(cost[best]),
                      student_top1_compatible=bool(low[best]))
        if spacing is not None:
            result['component_student_mass'] = [float(p[g].sum()) for g in groups]
    return result


def aggregate(rows):
    """Scene-balanced metrics with observation-balanced metrics alongside."""
    keys = sorted({k for r in rows for k,v in r.items()
                   if type(v) in (int, float) and k not in ('scene_id','view_id','window_size')})
    metrics = {}
    for key in keys:
        values = [float(r[key]) for r in rows if type(r.get(key)) in (int, float)]
        scenes = {}
        for r in rows:
            if type(r.get(key)) in (int, float):
                scenes.setdefault(r['scene_id'], []).append(float(r[key]))
        if values:
            metrics[key] = {'observations': len(values), 'scenes': len(scenes),
                            'observation_mean': float(np.mean(values)),
                            'scene_mean': float(np.mean([np.mean(v) for v in scenes.values()]))}
    return {'observations': len(rows), 'scenes': len({r['scene_id'] for r in rows}),
            'infinite_kl_observations': sum(r.get('forward_kl_infinite', False) for r in rows),
            'metrics': metrics}


def window_summary(rows):
    return {str(size): aggregate([r for r in rows if r['window_size'] == size])
            for size in sorted({r['window_size'] for r in rows})}


def unseen_scene_check(records, splits, trained_scene_ids):
    partition = checks.scene_splits(splits)
    evaluation = {r['scene_id'] for r in records}
    trained = set(trained_scene_ids)
    if not trained or not trained <= partition['train']:
        raise ValueError('actual trained scene IDs must be nonempty and belong to train')
    if not evaluation <= partition['test'] or evaluation & trained:
        raise ValueError('evaluation must use test scenes disjoint from training')
    return {'protocol': 'held_out_scene_ids', 'verified': True,
            'evaluated_scenes': len(evaluation), 'trained_scenes': len(trained),
            'unseen_obstacle_combinations_verified': False}


def unseen_obstacle_combinations(rows, scene_signatures, trained_scene_ids):
    """Declared categorical obstacle signatures, provided by evaluation-only geometry."""
    required = {str(r['scene_id']) for r in rows} | {str(x) for x in trained_scene_ids}
    if not required <= scene_signatures.keys():
        raise ValueError('missing train/test obstacle-combination signatures')
    if any(not isinstance(scene_signatures[k], list) for k in required):
        raise ValueError('each signature must be a JSON list of obstacle type labels')
    signature = lambda k: tuple(sorted(scene_signatures[str(k)]))
    trained = {signature(k) for k in trained_scene_ids}
    novel = [r for r in rows if signature(r['scene_id']) not in trained]
    return {'definition': 'sorted multiset of obstacle type labels',
            'novel_combination_results': aggregate(novel),
            'seen_combination_results': aggregate([r for r in rows if signature(r['scene_id']) in trained])}


def edge_ablation(response_record, edge_record, *, threshold):
    checks.teacher(response_record)
    checks.teacher(edge_record)
    checks.aligned(response_record, edge_record, config=False)
    if response_record['temperature'] != edge_record['temperature']:
        raise ValueError('edge ablation must preserve temperature')
    a = response_record['metadata']['settings']
    b = edge_record['metadata']['settings']
    if a['support'] != b['support'] or a['dataset'] != b['dataset']:
        raise ValueError('edge ablation must preserve dataset and support settings')
    pa, pb = a['physics'], b['physics']
    if pa.get('beta') != 0 or pb.get('beta', 0) <= 0 or not pb.get('edge_computed'):
        raise ValueError('need response-only beta=0 and measured edge beta>0 records')
    ignored = {'beta', 'edge_computed', 'name'}
    if {k:v for k,v in pa.items() if k not in ignored} != {k:v for k,v in pb.items() if k not in ignored}:
        raise ValueError('edge ablation changes additional physical settings')
    qa, qb = response_record['teacher_prob'], edge_record['teacher_prob']
    valid = response_record['valid']
    response_cost = response_record['physical_cost']
    return {'teacher_distribution_change': distributions(qa, qb),
            'response_only': evaluate_record(response_record, threshold=threshold),
            'response_plus_edge': evaluate_record(edge_record, threshold=threshold),
            'common_response_cost_expectation': {'response_only': float(qa[valid] @ response_cost[valid]),
                                               'response_plus_edge': float(qb[valid] @ response_cost[valid])},
            'student_retraining_comparison': 'pending matched trained predictions'}


def search_comparison(reference, covered, *, threshold, adaptive_evaluations, adaptive_seconds):
    """Coverage over reference support; omitted regions count as misses.

    covered is bool [N] of achieved common-spacing locations, never a nearest-
    neighbor approximation to exploratory adaptive points.
    """
    _, valid, cost, q = checks.teacher(reference)
    covered = np.asarray(covered)
    if covered.dtype != np.dtype(bool) or covered.shape != valid.shape:
        raise ValueError('coverage must be bool on ordered reference support')
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('threshold must be finite and nonnegative')
    if type(adaptive_evaluations) is not int or adaptive_evaluations <= 0 or not np.isfinite(adaptive_seconds) or adaptive_seconds < 0:
        raise ValueError('adaptive counts/time must be explicitly measured')
    meta = reference['metadata']
    if meta['settings'].get('search_mode') != 'uniform':
        raise ValueError('reference must be completed uniform search')
    n, seconds = meta['num_physics_evaluations'], meta['elapsed_seconds']
    if n <= 0 or seconds < 0:
        raise ValueError('uniform measurement missing')
    compatible = valid & (cost <= threshold)
    return {'common_support_candidates': len(valid),
            'compatible_cell_recall': float(covered[compatible].mean()) if compatible.any() else None,
            'teacher_mass_coverage': float(q[covered].sum()),
            'uniform_physical_evaluations': n, 'adaptive_physical_evaluations': adaptive_evaluations,
            'evaluation_reduction_fraction': 1-adaptive_evaluations/n,
            'uniform_seconds': seconds, 'adaptive_seconds': adaptive_seconds,
            'probability_vector_comparison': 'not performed on unequal supports'}


def plot_record(record, prediction, output, response=None):
    """Dependency-free SVG candidate plots; invalid candidates explicitly omitted."""
    xy, valid, cost, q = checks.teacher(record)
    fields = [('Physical cost', cost), ('Teacher mass', q)]
    if prediction is not None:
        checks.aligned(record, prediction)
        fields.append(('Student mass', checks.probability(prediction['student_prob'], valid, 'student_prob')))
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{340*len(fields)}" height="380" viewBox="0 0 {340*len(fields)} 380">', '<rect width="100%" height="100%" fill="white"/>']
    support = record.get('metadata', {}).get('settings', {}).get('support', {})
    world = float(support.get('world_size', 1024))
    for panel, (title, values) in enumerate(fields):
        left = panel*340+35
        maximum, minimum = float(values[valid].max()), float(values[valid].min())
        svg.append(f'<text x="{left}" y="22" font-size="16">{html.escape(title)}</text>')
        svg.append(f'<rect x="{left}" y="40" width="280" height="280" fill="#eee" stroke="#777"/>')
        for (x,y), value in zip(xy[valid], values[valid]):
            t = (float(value)-minimum)/(maximum-minimum) if maximum>minimum else .5
            color = f'rgb({int(240*t)},70,{int(240*(1-t))})'
            svg.append(f'<circle cx="{left+280*x/world:.3f}" cy="{40+280*y/world:.3f}" r="2.5" fill="{color}"><title>{x:g},{y:g}: {value:.6g}</title></circle>')
        x,y,size = record['window']
        svg.append(f'<rect x="{left+280*x/world}" y="{40+280*y/world}" width="{280*size/world}" height="{280*size/world}" fill="none" stroke="black"/>')
        svg.append(f'<text x="{left}" y="345" font-size="12">blue={minimum:.5g}; red={maximum:.5g}; y increases down</text>')
    svg.append('</svg>')
    Path(output).write_text('\n'.join(svg), encoding='utf-8')
    if response is not None:
        image = np.asarray(response)
        if image.shape != (int(record['window'][2]),)*2 or not np.isfinite(image).all():
            raise ValueError('response image does not match window')
        # Native-resolution PGM can be opened in standard scientific image tools.
        pixels = np.rint(np.clip(image,0,1)*255).astype(np.uint8)
        Path(output).with_suffix('.observation.pgm').write_bytes(f'P5\n{pixels.shape[1]} {pixels.shape[0]}\n255\n'.encode()+pixels.tobytes())


def run_manifest(path, output):
    path, output = Path(path), Path(output)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    base = path.parent
    threshold = manifest['compatibility_threshold']
    rows, records = [], []
    output.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(manifest['observations']):
        record = checks.load_record(base/item['teacher'])
        prediction = checks.load_record(base/item['student']) if item.get('student') else None
        if manifest.get('data_kind') != 'synthetic_fixture':
            settings = record['metadata'].get('settings', {})
            if not settings.get('dataset') or not settings.get('support') or not settings.get('physics'):
                raise ValueError('real-data evaluation requires dataset/support/physics provenance')
        spacing = record.get('metadata', {}).get('settings', {}).get('support', {}).get('candidate_spacing')
        if item.get('spacing') is not None and item['spacing'] != spacing:
            raise ValueError('manifest spacing conflicts with teacher provenance')
        rows.append(evaluate_record(record, prediction, threshold=threshold, spacing=spacing))
        records.append(record)
        response = None
        if item.get('observation'):
            sample = checks.load_record(base/item['observation'])
            response = checks.observation(record, sample)
        plot_record(record, prediction, output/f'observation-{i:04}.svg', response)
    if not rows:
        raise ValueError('manifest has no observations')
    identities = [(r['scene_id'],r['view_id']) for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate observations in primary evaluation')
    report = {'data_kind': manifest.get('data_kind', 'unspecified'), 'observations': rows,
              'aggregate': aggregate(rows), 'by_window_size': window_summary(rows),
              'unseen_scene': {'verified': False, 'reason': 'split/training provenance not supplied'},
              'edge_ablation': [], 'search_comparison': []}
    if manifest.get('splits'):
        report['unseen_scene'] = unseen_scene_check(records, manifest['splits'], manifest['trained_scene_ids'])
        if manifest.get('scene_obstacle_signatures'):
            report['obstacle_combinations'] = unseen_obstacle_combinations(rows, manifest['scene_obstacle_signatures'], manifest['trained_scene_ids'])
    for pair in manifest.get('edge_pairs', []):
        report['edge_ablation'].append(edge_ablation(checks.load_record(base/pair['response_only']),
            checks.load_record(base/pair['response_edge']), threshold=threshold))
    for pair in manifest.get('search_pairs', []):
        reference = checks.load_record(base/pair['reference'])
        trace = checks.load_record(base/pair['adaptive_coverage'])
        checks.aligned(reference, trace)
        metadata = trace.get('metadata', {})
        if metadata.get('search_mode') != 'adaptive' or metadata.get('complete') is not True:
            raise ValueError('adaptive coverage must explicitly declare complete adaptive search')
        report['search_comparison'].append(search_comparison(reference, trace['covered'], threshold=threshold,
            adaptive_evaluations=metadata['num_physics_evaluations'], adaptive_seconds=metadata['elapsed_seconds']))
    if manifest.get('data_kind') == 'synthetic_fixture':
        report['research_result'] = False
    else:
        report['research_result'] = 'requires independent provenance/physical checks; record metrics only'
    (output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (output/'summary.md').write_text(f"# Part E record evaluation\n\nData: {report['data_kind']}\n\n{len(rows)} observations, {report['aggregate']['scenes']} scenes.\n\nHeld-out protocol verified: {report['unseen_scene']['verified']}.\n\nEdge pairs: {len(report['edge_ablation'])}; search pairs: {len(report['search_comparison'])}.\n\nSee metrics.json and per-observation SVGs. Missing experiments have no inferred results.\n", encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run_manifest(args.manifest, args.output)
    print(f"Saved {len(report['observations'])} observation results to {args.output}; data={report['data_kind']}")


if __name__ == '__main__':
    main()
