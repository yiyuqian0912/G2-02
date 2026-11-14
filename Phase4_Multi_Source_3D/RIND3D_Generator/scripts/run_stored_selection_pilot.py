"""Controlled selection-only pilot over immutable complete stored fields.

Every frozen source is inventoried. Incomplete fields stay BLOCKED, not replaced.
A/B/C quota ports and D use the same exact stored truth. No source equivalence
verifier is changed, no READY scene is produced, and no mass production occurs.
"""
import argparse,json,os,resource,subprocess,time
from pathlib import Path
from collections import Counter
import numpy as np
from rind3d.pack import RINDPack,atomic_json
from rind3d.information import InformationIndex
from rind3d.stored_windows import select_stored_windows,selection_settings
from rind3d.window_contract import balanced_quotas
from rind3d.windows import Window,excludes
from rind3d.cli import load_config
from stored_selection_baselines import scan,guided

ROOT=Path('outputs/stored-window-selection');BENCH=ROOT/'benchmark'
METHODS=('A_scan','B_stored','C_stored','D_uniform','D_soft')
TARGETS=(32,128,512,4096)


def validate_rows(rows,case):
    source=np.asarray(case['source']);seen=set();tasks=[]
    with RINDPack(case['response']) as r,RINDPack(case['solid']) as m:
        for row in rows:
            w=Window(tuple(row['origin']),row['side'],witnesses=tuple(map(tuple,row['witnesses'])))
            if w.record() in seen or not excludes(w,source,1024):raise AssertionError('Duplicate or source-containing benchmark window')
            if min(w.origin)<0 or max(w.origin)+w.size>1024:raise AssertionError('Bounds')
            seen.add(w.record())
            for point,value in zip(w.witnesses,(True,False)):
                if not all(o<=v<o+w.size for o,v in zip(w.origin,point)):raise AssertionError('Witness outside window')
                tasks.append((point,value))
        for point,value in sorted(tasks,key=lambda x:tuple(v//64 for v in x[0])):
            if m.read_point(*point) or r.read_point(*point)!=value:raise AssertionError('Independent stored witness failed')
    return {'exact_valid_windows':len(rows),'independent_stored_point_checks':len(tasks),'duplicates':0,'oracle_calls':0}


def prepare_inputs():
    BENCH.mkdir(exist_ok=True);cases=[];blocked=[]
    for path in sorted((ROOT/'fields').glob('*/preparation.json')):
        d=json.loads(path.read_text());case={'id':path.parent.name,'seed':d['seed'],'scene':d['scene'],'source_rank':d['source_rank'],
            'source':d['source'],'families':d['families'],'geometry_hash':d['geometry_hash'],'synthetic':False,
            'preparation_wall':d['wall_seconds'],'preparation_process':d['process_seconds'],'ready_capable':d.get('ready_capable')}
        if d['status']!='COMPLETE_STORED_FIELD':blocked.append({**case,'status':d['status'],'reason':d.get('reason'),'failure_type':d.get('failure_type')});continue
        case.update(response=d['response'],solid=d['solid']);cases.append(case)
    for name,scene,synthetic in (('box-fixture','outputs/box1024',False),('rare-stress',str(ROOT/'rare1024'),True)):
        p=Path(scene);cases.append({'id':name,'seed':20261031,'scene':0,'source_rank':0,'source':np.fromfile(p/'driver.f64',dtype='<f8').tolist(),
            'families':['synthetic-rare'] if synthetic else ['box-fixture'],'synthetic':synthetic,'response':str(p/'response.rindb'),'solid':str(p/'solid.rindb'),
            'preparation_wall':0,'preparation_process':0,'ready_capable':False if synthetic else True})
    for case in cases:
        root=BENCH/case['id'];root.mkdir(exist_ok=True)
        with RINDPack(case['response']) as r,RINDPack(case['solid']) as m:
            if r.shape!=(1024,)*3:raise ValueError('A partial/small field cannot enter this pilot')
            index=InformationIndex.build(r,m);size=index.save(root/'information.rindi')
            case['index_build']={**index.build_report,'persistent_bytes':size};case['index_path']=str(root/'information.rindi')
        atomic_json(root/'case.json',case)
    atomic_json(BENCH/'cohort.json',{'cases':cases,'blocked':blocked,'frozen_source_cases':48,'targets':TARGETS,'methods':METHODS,
        'source_resampling':False,'is_dataset':False,'publication_attempted':False})
    return cases


def one(case_id,method,target):
    case=json.loads((BENCH/case_id/'case.json').read_text());out=BENCH/case_id/f'{method}-{target}';out.mkdir(exist_ok=False)
    cfg=load_config('configs/pilot-extended.yaml');settings=selection_settings(cfg['window_selection']);source=np.asarray(case['source'])
    started=time.perf_counter();cpu=time.process_time();rows=[];extra={}
    with RINDPack(case['response'],cache_bricks=16) as r,RINDPack(case['solid'],cache_bricks=16) as m:
        if method=='A_scan':rows,extra=scan(r,m,source,target,case['seed'],case['scene'],settings['uniform_max_proposals_per_side'],out)
        elif method in ('B_stored','C_stored'):
            b_settings=json.loads(Path('configs/experimental-boundary-guided.json').read_text())
            rows,extra=guided(r,m,source,target,case['seed'],case['scene'],b_settings,method,out)
        else:
            index=InformationIndex.load(case['index_path'],r,m)
            settings.update(control_per_side=0)
            if method=='D_uniform':settings.update(selection_mode='uniform',guided_fallback=False)
            result=select_stored_windows(index,source,target,seed=case['seed'],scene=case['scene'],settings=settings,output=out)
            rows=result.provenance;extra=result.report
        reader={'decoded_bricks':r.decoded_bricks+m.decoded_bricks,'compressed_bytes_read':r.bytes_read+m.bytes_read,
                'packed_cache_limit_bytes':2*16*32768}
    validation=validate_rows(rows,case);counts=dict(Counter(row['side'] for row in rows));complete=all(counts.get(s,0)==n for s,n in balanced_quotas(target).items())
    report={'case':case_id,'method':method,'target':target,'complete':complete,'status':'COMPLETE' if complete else 'INCOMPLETE',
        'side_counts':counts,'selected':len(rows),'validation':validation,'wall_seconds':time.perf_counter()-started,'process_seconds':time.process_time()-cpu,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'reader':reader,'details':extra,'oracle_calls':0,
        'index_build_wall_charged':case['index_build']['wall_seconds'] if method.startswith('D_') else 0,
        'index_build_process_charged':case['index_build']['process_seconds'] if method.startswith('D_') else 0,
        'source_resampled':False,'is_dataset':False,'production_ready':False}
    report['end_to_end_wall']=report['wall_seconds']+report['index_build_wall_charged'];report['end_to_end_process']=report['process_seconds']+report['index_build_process_charged']
    (out/'selected.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows));atomic_json(out/'result.json',report)
    print(case_id,method,target,report['status'],len(rows),report['end_to_end_wall'],flush=True)


def proc_usage(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().split();hz=os.sysconf('SC_CLK_TCK')
        status=Path(f'/proc/{pid}/status').read_text().splitlines();peak=next((int(s.split()[1])*1024 for s in status if s.startswith('VmHWM:')),None)
        return {'process_seconds':(int(stat[13])+int(stat[14]))/hz,'peak_rss_bytes':peak}
    except (FileNotFoundError,ProcessLookupError):return {}


def run(workers,budget,case_filter=None):
    cohort=json.loads((BENCH/'cohort.json').read_text());cases=[c for c in cohort['cases'] if case_filter is None or c['id']==case_filter]
    jobs=[]
    for c in cases:
        for target in TARGETS:
            for method in METHODS:
                out=BENCH/c['id']/f'{method}-{target}'
                if (out/'result.json').exists():continue
                jobs.append((c,method,target))
    running=[];done=[];started=time.time()
    while jobs or running:
        while jobs and len(running)<workers:
            case,method,target=jobs.pop(0);name=f"{case['id']}-{method}-{target}";log=(BENCH/(name+'.log')).open('w')
            args=['.venv/bin/python',__file__,'--case',case['id'],'--method',method,'--target',str(target)]
            process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            running.append((case,method,target,process,log,time.time()))
        for item in running[:]:
            case,method,target,process,log,start=item;code=process.poll();out=BENCH/case['id']/f'{method}-{target}';timed=False;usage={}
            if code is None and time.time()-start>budget:
                usage=proc_usage(process.pid);process.kill();code=process.wait();timed=True
            elif code is None:continue
            log.close();running.remove(item)
            if not (out/'result.json').exists():
                out.mkdir(exist_ok=True);partial=out/'partial-selected.jsonl';rows=[]
                if partial.exists():
                    for line in partial.read_text().splitlines():
                        try:row=json.loads(line)
                        except ValueError:continue
                        if method in ('B_stored','C_stored') and row.get('method')!=method:continue
                        rows.append(row)
                valid=validate_rows(rows,case);charge=case['index_build']['wall_seconds'] if method.startswith('D_') else 0
                report={'case':case['id'],'method':method,'target':target,'complete':False,'status':'WALL_BUDGET_INCOMPLETE' if timed else 'ERROR',
                    'exit_code':code,'selected':len(rows),'side_counts':dict(Counter(r['side'] for r in rows)),'validation':valid,
                    'wall_seconds':time.time()-start,'end_to_end_wall':time.time()-start+charge,'index_build_wall_charged':charge,
                    **usage,'oracle_calls':0,'source_resampled':False,'production_ready':False,'timeout_is_physical_rejection':False}
                (out/'selected.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows));atomic_json(out/'result.json',report)
            done.append({'case':case['id'],'method':method,'target':target,'exit_code':code,'watchdog':timed});print(done[-1],flush=True)
        atomic_json(BENCH/'progress.json',{'elapsed_seconds':time.time()-started,'done':done,'running':[f'{c["id"]}-{m}-{t}' for c,m,t,*_ in running],'remaining':len(jobs),'wall_budget_per_job':budget})
        if running or jobs:time.sleep(.5)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--case');p.add_argument('--method',choices=METHODS);p.add_argument('--target',type=int,choices=TARGETS)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seconds',type=float,default=60);p.add_argument('--only-case');a=p.parse_args()
    if a.prepare:prepare_inputs()
    elif a.case:one(a.case,a.method,a.target)
    else:run(a.workers,a.seconds,a.only_case)


if __name__=='__main__':main()
