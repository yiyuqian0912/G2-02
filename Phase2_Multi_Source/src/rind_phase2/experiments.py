"""Phase II experiment preflight; data/joint-teacher tools work independently."""
import argparse
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/experiment_multisource.json')
    parser.add_argument('--check', action='store_true', help='Inspect draft; not a training-readiness guarantee')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.check:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        print('Data and joint-teacher code imported; training is NOT ready.')
        print('Pending: experiment split, candidate/count prior settings, source-set model, training and student evaluation.')
        return 0
    parser.error('Phase II student training is not implemented. Use --check, or the working rind-multi-teacher CLI. No checkpoint was written.')


if __name__ == '__main__':
    main()
