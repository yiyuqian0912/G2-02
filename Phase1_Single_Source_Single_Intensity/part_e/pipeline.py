"""Bounded real-data end-to-end orchestration with stage-specific evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from . import checks
from .adapters import load_thomas_model, predict_thomas
from .record_io import save
from .evaluate import evaluate_record, plot_record


def execute(config, output, dataset_factory, teacher_factory, predictor=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {'full_pipeline_complete': False, 'teacher_side_complete': False,
              'evaluation_support': 'teacher_supported_hidden_geometry_mask',
              'config': config, 'stages': []}
    def stage(name, action):
        try:
            value = action()
        except (FileNotFoundError, ImportError) as exc:
            report['stages'].append({'name':name, 'status':'blocked', 'reason':str(exc)})
            raise
        except Exception as exc:
            report['stages'].append({'name':name, 'status':'failed', 'reason':str(exc)})
            raise
        report['stages'].append({'name':name, 'status':'passed'})
        return value
    try:
        dataset = stage('A_dataset', lambda: dataset_factory(config))
        sample = stage('A_observation', lambda: dataset.get_observation(config['scene_id'],config['view_id']))
        def verify_physics():
            scene = dataset.get_scene(config['scene_id'])
            truth = scene['drivers'][0]
            result = checks.physical_reconstruction(dataset,config['scene_id'],config['view_id'],truth,
                                                   backend=config.get('backend','auto'))
            references = dataset.get_candidates(config['scene_id'],config['view_id'])
            if len(references) != 10:
                raise ValueError('expected ten reference witnesses')
            for reference in references:
                checks.physical_reconstruction(dataset,config['scene_id'],config['view_id'],reference[0,:2],
                                               backend=config.get('backend','auto'))
            result['reference_witnesses_checked'] = len(references)
            return result
        report['physical_checks'] = stage('B_physical_reconstruction',verify_physics)
        record = stage('C_teacher_generation',lambda:teacher_factory(dataset,config))
        def handoff():
            checks.teacher(record)
            checks.observation(record,sample)
            save(record,output/'teacher.npz')
            save(sample,output/'observation.npz')
        stage('C_teacher_handoff',handoff)
        prediction = None
        if predictor is not None:
            prediction = stage('D_student_prediction',lambda:predictor(record,sample,config))
            def student_handoff():
                checks.aligned(record,prediction)
                checks.probability(prediction['student_prob'],checks.prediction_support(prediction,record['valid']),'student_prob')
                save(prediction,output/'student.npz')
            stage('D_student_handoff',student_handoff)
        else:
            report['stages'].append({'name':'D_student_prediction', 'status':'blocked',
                                     'reason':'No real trained checkpoint configured; Teacher-only run.'})
        def evaluation():
            metrics = evaluate_record(record,prediction,threshold=config['compatibility_threshold'],
                                      spacing=config['candidate_spacing'])
            (output/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
            plot_record(record,prediction,output/'comparison.svg',sample['response'])
            return metrics
        report['metrics'] = stage('E_evaluation_visualization',evaluation)
        report['teacher_side_complete'] = True
        report['full_pipeline_complete'] = prediction is not None
    except Exception:
        # The complete error is preserved in the responsible stage, with prior evidence retained.
        pass
    (output/'end_to_end_result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    lines = ['# Part E end-to-end checks','',
             f"Teacher-side complete: {report['teacher_side_complete']}",
             f"Full single-observation pipeline complete: {report['full_pipeline_complete']}",'',
             '| Stage | Status | Reason |','|---|---|---|']
    lines += [f"| {s['name']} | {s['status']} | {s.get('reason','').replace('|','/').replace(chr(10),' ')} |" for s in report['stages']]
    lines += ['', 'A successful single-observation handoff does not establish held-out generalization or complete all Part E experiments.']
    (output/'end_to_end_result.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def run(config, output):
    def dataset_factory(cfg):
        root = Path(cfg['data_root'])
        if not (root/'manifest.json').is_file():
            raise FileNotFoundError(f'Real installed RIND data missing: {root}')
        source = Path(cfg['teacher_source']).resolve()
        sys.path[:0] = [str(source/'src'),str(source/'vendor/rind-dataset')]
        from rind_phase1.data import Phase1Dataset
        return Phase1Dataset(root)
    def teacher_factory(dataset,cfg):
        from rind_phase1.teacher import generate_uniform_teacher
        return generate_uniform_teacher(dataset,cfg['scene_id'],cfg['view_id'],
            spacing=cfg['candidate_spacing'],temperature=cfg['temperature'],
            quadtree_prior=False,backend=cfg.get('backend','auto'),
            max_evaluations=cfg.get('max_evaluations'))
    def predictor(record,sample,cfg):
        if cfg.get('student_interface', 'phase1-v1') == 'phase1-v1':
            from rind_phase1.train import load_model
            model, checkpoint = load_model(cfg['checkpoint'])
        elif cfg['student_interface'] == 'legacy-relative-v0':
            model, checkpoint = load_thomas_model(cfg['student_source'], cfg['checkpoint'],
                trusted_checkpoint=cfg.get('trusted_checkpoint', False))
        else:
            raise ValueError('unknown student_interface')
        if checkpoint.get('global_step',0) <= 0:
            raise ValueError('checkpoint has no evidence of training steps')
        if getattr(model, 'interface_version', None) == 'phase1-v1':
            from rind_phase1.predict import predict
            result = predict(model, record, sample)
        else:
            result = predict_thomas(model, record, sample)
        result['metadata'] = {'checkpoint_sha256':hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest(),
                              'checkpoint_config':checkpoint.get('model_config', checkpoint.get('config')),
                              'global_step':checkpoint['global_step'],
                              'data_fingerprints':checkpoint.get('data_fingerprints')}
        return result
    return execute(config,output,dataset_factory,teacher_factory,predictor if config.get('checkpoint') else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    for key in ('teacher_source','student_source','data_root','checkpoint'):
        if config.get(key):
            config[key] = str((args.config.parent/config[key]).resolve())
    report = run(config,args.output)
    print(f"Teacher-side complete={report['teacher_side_complete']}; full pipeline complete={report['full_pipeline_complete']}. Report: {args.output}")
    if not report['full_pipeline_complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
