"""Score an existing C Teacher record with an existing D checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from . import checks
from .adapters import load_thomas_model, predict_thomas
from .record_io import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-source', type=Path, required=True,
                        help='C branch Phase I project root (src/ and vendor/ beneath it)')
    parser.add_argument('--student-source', type=Path, required=True, help='D student_baseline directory')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--trusted-checkpoint', action='store_true',
                        help='Allow D full checkpoint containing serialized RNG objects, only for trusted local files')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.teacher_source.resolve()/'src'),
                    str(args.teacher_source.resolve()/'vendor/rind-dataset')]
    from rind_phase1.data import Phase1Dataset
    record = checks.load_record(args.teacher)
    dataset = Phase1Dataset(args.data_root)
    observation = dataset.get_observation(record['scene_id'],record['view_id'])
    model, checkpoint = load_thomas_model(args.student_source,args.checkpoint,
                                         trusted_checkpoint=args.trusted_checkpoint)
    prediction = predict_thomas(model,record,observation)
    prediction['metadata'] = {'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                              'checkpoint_config':checkpoint.get('config'),
                              'checkpoint_epoch':checkpoint.get('epoch'),
                              'data_fingerprints':checkpoint.get('data_fingerprints'),
                              'coordinate_normalization':'(world_xy-window_origin)/window_size',
                              'evaluation_support':'teacher_supported_hidden_geometry_mask'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save(prediction,args.output)
    save(observation,args.output.with_suffix('.observation.npz'))
    print(f'Saved prediction and matched observation: {args.output}')


if __name__ == '__main__':
    main()
