from __future__ import annotations
import argparse,json
from pathlib import Path
import yaml
from .window_contract import validate_window_config

def load_config(path):
    return validate_config(yaml.safe_load(Path(path).read_text()))

def validate_config(cfg):
    from .window_contract import WINDOW_SIDES
    for key in ('pilot_side_counts','production_side_counts'):
        counts=cfg.get(key)
        if isinstance(counts,dict) and set(counts)==set(map(str,WINDOW_SIDES)):
            cfg[key]={int(k):v for k,v in counts.items()}
    if cfg.get('format_version')!=2:raise ValueError('New generation requires format_version: 2; v1 datasets remain readable')
    from .profiles import profile_for,LITE
    profile=profile_for(cfg)
    if cfg.get('max_candidates')!=profile.max_candidates or cfg['production_windows']!=profile.windows or cfg['resolution']!=1024:raise ValueError('Profile candidate/window/resolution contract mismatch')
    if cfg['min_source_separation']!=32/1024:raise ValueError('Source separation must remain 32/1024')
    if cfg.get('include_true_source') is not True:raise ValueError('The true source must be retained')
    if cfg.get('source_precision')!='float64' or cfg.get('response_storage')!='RINDPack-v1':raise ValueError('Explicit float64 source and lossless RINDPack are required')
    if cfg['candidate_domain'] not in ('world_free','world_outside_window'):raise ValueError('Unknown source domain')
    validate_window_config(cfg)
    from .stored_windows import selection_settings
    selection_settings(cfg.get('window_selection'))
    if cfg.get('information_block_side',16)!=16:raise ValueError('The current compact index requires information_block_side: 16')
    if cfg.get('field_acceleration','baseline') not in ('baseline','certified_function'):raise ValueError('Unknown field_acceleration mode')
    if 'field_acceleration_settings' in cfg:
        from .field_acceleration import settings
        settings(cfg['field_acceleration_settings'])
    if cfg['max_window_attempts']<profile.windows or cfg['global_proposals']<0:raise ValueError('Invalid proposal budgets')
    for key in ('search_batch_size','candidate_pool','verification_seconds','verification_points','window_search_seconds','window_search_points','pilot_source_attempts'):
        if key=='window_search_seconds' and profile.name==LITE and cfg.get('ambiguity',{}).get('enabled') is False and cfg[key]==0:continue
        if cfg[key]<=0:raise ValueError('Invalid budget: '+key)
    if profile.name==LITE:
        from .lite_generators import FAMILIES
        from .rendering.backends import BACKENDS
        if not cfg.get('families') or any(f not in FAMILIES for f in cfg['families']):raise ValueError('Unsupported Lite family')
        if cfg.get('renderer_backend') not in BACKENDS:raise ValueError('Explicit renderer_backend required')
        if cfg.get('profile_version')!='0.1' or cfg.get('driver_clearance_hex')!=(2**-26).hex():raise ValueError('Lite version/driver clearance contract mismatch')
        if 'quality_certificate' in cfg:
            from .stratified_windows import settings
            settings(cfg.get('observation_sampling'))
            if cfg.get('geometry_mixture') not in ('original-v01','fast-v1'):raise ValueError('Unknown Core geometry mixture')
            if cfg.get('ambiguity',{}).get('enabled') is not False or cfg['window_search_seconds']!=0 or cfg['candidate_domain']!='world_outside_window':raise ValueError('Core requires explicit zero optional search and outside-window sources')
            if cfg['geometry_mixture']=='fast-v1':
                from .fast_geometry import FAMILIES as FAST_FAMILIES
                if any(f not in FAST_FAMILIES for f in cfg['families']):raise ValueError('Family outside declared fast mixture')
    return cfg

def main(argv=None):
    import sys
    values=list(sys.argv[1:] if argv is None else argv)
    if values and values[0]=='campaign':
        from .campaign_cli import main as campaign_main
        return campaign_main(values[1:])
    parser=argparse.ArgumentParser(prog='rind3d');sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('pilot');p.add_argument('--config',required=True);p.add_argument('--scenes',type=int,default=8);p.add_argument('--windows-per-scene',type=int,default=32);p.add_argument('--output',default='outputs/pilot')
    p=sub.add_parser('generate');p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--scenes',type=int,default=8);p.add_argument('--resume',action='store_true');p.add_argument('--pilot-report',default='outputs/pilot/report.json')
    p=sub.add_parser('verify');p.add_argument('--root',required=True);p.add_argument('--scene');p.add_argument('--full-index-check',action='store_true');p.add_argument('--physical',action='store_true')
    p=sub.add_parser('benchmark');p.add_argument('--config',required=True);p.add_argument('--compare-codecs',action='store_true');p.add_argument('--compare-backends',action='store_true');p.add_argument('--output',default='outputs/benchmark');p.add_argument('--full-1024',action='store_true')
    p=sub.add_parser('stats');p.add_argument('--root',required=True);p.add_argument('--by-family',action='store_true');p.add_argument('--by-window-size',action='store_true')
    p=sub.add_parser('export-zarr');p.add_argument('--response',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('build-info-index');p.add_argument('--response',required=True);p.add_argument('--solid',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('build-count-index');p.add_argument('--response',required=True);p.add_argument('--solid',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('window-counts');p.add_argument('--scene',required=True);p.add_argument('--origin',nargs=3,type=int,required=True);p.add_argument('--side',type=int,choices=(64,128,256),required=True);p.add_argument('--index')
    p=sub.add_parser('select-windows');p.add_argument('--scene',required=True);p.add_argument('--output',required=True);p.add_argument('--config',required=True)
    p.add_argument('--count',type=int,default=4096);p.add_argument('--index');p.add_argument('--diagnostic',action='store_true');p.add_argument('--allow-synthetic',action='store_true')
    p=sub.add_parser('certify-lite');p.add_argument('--output',default='outputs/lite-v01-production/scientific-certification.json');p.add_argument('--cpu-only',action='store_true')
    p=sub.add_parser('lite-pilot');p.add_argument('--config',default='configs/lite-v0.1.yaml');p.add_argument('--output',required=True);p.add_argument('--scenes',type=int,choices=(10,100),default=10);p.add_argument('--resume',action='store_true');p.add_argument('--pilot-report')
    p=sub.add_parser('core-pilot');p.add_argument('--config',default='configs/lite-core-research-v1.yaml');p.add_argument('--output',required=True);p.add_argument('--scenes',type=int,choices=(10,100,300),default=10);p.add_argument('--resume',action='store_true');p.add_argument('--pilot-report');p.add_argument('--verification-workers',type=int,choices=(1,2,4),default=2);p.add_argument('--sequential',action='store_true');p.add_argument('--verifier',choices=('original','silhouette'),default='silhouette')
    p=sub.add_parser('enrich');p.add_argument('--scene',required=True);p.add_argument('--windows',nargs='+',type=int,required=True);p.add_argument('--config',default='configs/ambiguity-enrichment-research-v1.yaml');p.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='enrich':
            from .enrichment_cli import run
            print(json.dumps(run(args.scene,args.windows,args.config,args.output),indent=2));return 0
        if args.command in ('build-count-index','window-counts'):
            from .pack import RINDPack
            from .counts import CountIndex
            if args.command=='build-count-index':yp,mp=args.response,args.solid
            else:yp,mp=Path(args.scene)/'response.rindb',Path(args.scene)/'solid.rindb'
            with RINDPack(yp,verify_file=True) as y,RINDPack(mp,verify_file=True) as m:
                index=CountIndex.load(args.index,y,m) if args.command=='window-counts' and args.index else CountIndex.build(y,m)
                if args.command=='build-count-index':result=dict(index.build_report,persistent_bytes=index.save(args.output))
                else:
                    c=index.query(tuple(args.origin),args.side);result={'lit_free':c.lit,'shadow_free':c.shadow,'solid':c.solid,'lit_ratio':c.lit_ratio,'informative':c.informative,'ratio_bin':c.ratio_bin if c.informative else None}
                print(json.dumps(result,indent=2));return 0
        if args.command=='core-pilot':
            from .core_research import run
            r=run(load_config(args.config),args.output,args.scenes,workers=args.verification_workers,resume=args.resume,pilot_report=args.pilot_report,sequential=args.sequential,verifier=args.verifier)
            print(json.dumps({'complete':r['complete'],'ready_scenes':r['ready_scenes'],'report':str(Path(args.output)/'report.json')},indent=2));return 0 if r['complete'] else 2
        if args.command=='certify-lite':
            from .lite_certification import certify
            Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            result=certify(args.output,cuda=not args.cpu_only)
            print(json.dumps({'certificate':args.output,'complete':result['complete'],'seconds':result['seconds']},indent=2));return 0
        if args.command=='lite-pilot':
            from .lite_pipeline import generate_lite
            cfg=load_config(args.config)
            if 'quality_certificate' in cfg:raise ValueError('Use core-pilot for the explicitly versioned Core research configurations')
            result=generate_lite(cfg,args.output,args.scenes,pilot_report=args.pilot_report,resume=args.resume)
            print(json.dumps({'report':str(Path(args.output)/'report.json'),'complete':result['complete'],'ready_scenes':result['ready_scenes'],'failures':result['failures']},indent=2));return 0 if result['complete'] else 2
        if args.command=='build-info-index':
            from .selection_cli import build_index
            print(json.dumps(build_index(args.response,args.solid,args.output),indent=2));return 0
        if args.command=='select-windows':
            from .selection_cli import select_scene
            report=select_scene(args.scene,args.output,load_config(args.config),args.count,index_path=args.index,
                                diagnostic=args.diagnostic,allow_synthetic=args.allow_synthetic)
            print(json.dumps(report,indent=2));return 0 if report['complete'] else 2
        if args.command=='export-zarr':
            from .zarr_store import export_zarr
            print(export_zarr(args.response,args.output));return 0
        if args.command=='pilot':
            from .pilot import run_pilot
            r=run_pilot(load_config(args.config),args.output,args.scenes,args.windows_per_scene)
            print(json.dumps({'report':str(Path(args.output)/'report.json'),'publication_feasible':r['publication_feasible'],'search_feasible':r['search_feasible'],'summary':r['summary'],'by_window_side':r['by_window_side']},indent=2))
            return 0 if r['publication_feasible'] else 2
        if args.command=='generate':
            from .pipeline import generate_dataset
            cfg=load_config(args.config)
            if 'quality_certificate' in cfg:raise ValueError('Use core-pilot for the explicitly versioned Core research configurations')
            generate_dataset(cfg,args.output,args.scenes,args.pilot_report,resume=args.resume);return 0
        if args.command=='benchmark':
            from .benchmark import benchmark
            print(json.dumps(benchmark(load_config(args.config),args.output,args.compare_codecs,args.compare_backends,args.full_1024),indent=2));return 0
        if args.command=='verify':
            from .pipeline import verify_dataset
            print(json.dumps(verify_dataset(args.root,args.scene,full_index=args.full_index_check,physical=args.physical),indent=2));return 0
        if args.command=='stats':
            from .data import dataset_stats
            print(json.dumps(dataset_stats(args.root),indent=2));return 0
    except (ValueError,RuntimeError,OSError) as e:
        parser.exit(2,f'{type(e).__name__}: {e}\n')

if __name__=='__main__':raise SystemExit(main())
