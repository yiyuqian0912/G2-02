"""Resume a prepared experiment after a reviewed runtime-code update.

This entry point deliberately reads the experiment's frozen protocol instead
of rebuilding it with current source hashes.  It is intended for bug fixes to
the training runtime after teacher targets have already been fully rendered;
the original config, split, target index and run signature remain authoritative.
"""
import argparse
import json
from pathlib import Path

from .data import PROJECT_ROOT, Phase1Dataset
from .experiments import evaluation, summarize, train
from .train import DEFAULT_SHARPENED_OBJECTIVE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('train','evaluate','summarize'))
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--variant',default='response')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--run-label')
    parser.add_argument('--entropy-match-weight',type=float)
    parser.add_argument('--ranking-loss-weight',type=float)
    parser.add_argument('--ranking-positive-mass',type=float)
    parser.add_argument('--ranking-negative-fraction',type=float)
    parser.add_argument('--ranking-margin',type=float)
    parser.add_argument('--ranking-softmin-temperature',type=float)
    args=parser.parse_args()
    root=args.output_root.expanduser().resolve()
    protocol_path=root/'protocol.json'
    if not protocol_path.is_file():
        parser.error(f'prepared protocol is missing: {protocol_path}')
    protocol=json.loads(protocol_path.read_text())
    c=protocol['config']
    if args.variant not in c['variants']:
        parser.error(f'unknown variant: {args.variant}')
    raw=Path(c['data_root']).expanduser()
    data_root=raw if raw.is_absolute() else PROJECT_ROOT/raw
    dataset=Phase1Dataset(data_root.resolve())
    seeds=c['training']['seeds'] if args.seed is None else [args.seed]
    if not set(seeds)<=set(c['training']['seeds']):
        parser.error('seed must be declared in the frozen protocol')
    print(f'Reusing frozen protocol and prepared teachers: {protocol_path}',flush=True)
    if args.stage=='train':
        supplied={
            'entropy_match_weight':args.entropy_match_weight,
            'ranking_loss_weight':args.ranking_loss_weight,
            'ranking_positive_mass':args.ranking_positive_mass,
            'ranking_negative_fraction':args.ranking_negative_fraction,
            'ranking_margin':args.ranking_margin,
            'ranking_softmin_temperature':args.ranking_softmin_temperature,
        }
        objective=(None if all(value is None for value in supplied.values()) else
                   {key:(DEFAULT_SHARPENED_OBJECTIVE[key] if value is None else value)
                    for key,value in supplied.items()})
        if objective is not None and args.run_label is None:
            parser.error('--run-label is required for a noncanonical student objective')
        for seed in seeds:
            train(c,root,dataset,protocol,args.variant,seed,args.run_label,objective)
    elif args.stage=='evaluate':
        for seed in seeds:evaluation(c,root,dataset,protocol,args.variant,seed,args.run_label)
    else:
        summarize(c,root,args.variant)


if __name__=='__main__':main()
