# Legacy four-side diagnostic; archived results/source snapshots are unchanged.
"""Offline A/B accounting and exact integer-origin/multi-scale audits."""
from __future__ import annotations
import argparse,csv,json
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
from rind3d.pack import atomic_json
from rind3d.legacy.window_contract import WINDOW_SIDES
from boundary_experiment_support import plain
from experimental_boundary_guided import OriginBox,pair_region,origin_options


def union_rectangles(rectangles):
    """Exact integer point count via a continuous union of [lo,hi+1) cells."""
    events=[]
    for lo,hi in rectangles:
        events.extend(((lo[0],1,lo[1],hi[1]+1),(hi[0]+1,-1,lo[1],hi[1]+1)))
    events.sort();active=Counter();last=None;area=0
    for x,sign,a,b in events:
        if last is not None and x>last:
            end=None;length=0
            for left,right in sorted(k for k,v in active.items() if v>0):
                if end is None or left>end:length+=right-left;end=right
                elif right>end:length+=right-end;end=right
            area+=(x-last)*length
        active[(a,b)]+=sign;last=x
    return int(area)


def union_origin_count(boxes):
    """Exact distinct-origin count; unlike summed pair volumes, no overlap inflation."""
    edges=sorted({v for b in boxes for v in (b.lower[0],b.upper[0]+1)})
    total=0
    for left,right in zip(edges[:-1],edges[1:]):
        rectangles=[(b.lower[1:],b.upper[1:]) for b in boxes if b.lower[0]<=left<=b.upper[0]]
        total+=(right-left)*union_rectangles(rectangles)
    return int(total)


def common_center_regions(pair,source,resolution):
    """All integer centres supporting four exactly co-centred nested windows."""
    common=None
    for side in WINDOW_SIDES:
        boxes=[OriginBox(tuple(v+side//2 for v in b.lower),tuple(v+side//2 for v in b.upper))
               for b in pair_region(pair,source,side,resolution)]
        if common is None:common=boxes;continue
        next_boxes=[]
        for a in common:
            for b in boxes:
                lo=tuple(max(x,y) for x,y in zip(a.lower,b.lower));hi=tuple(min(x,y) for x,y in zip(a.upper,b.upper))
                box=OriginBox(lo,hi)
                if box.count:next_boxes.append(box)
        common=next_boxes
    return common or []


def nearby_nested_chain(pair,source,resolution):
    """Construct a clipped-centre nested chain, if one of the finite base options works.

    Failure is inconclusive, not a proof that no chain exists. Every returned
    origin is in the exact legal region; no random origin search is performed.
    """
    base=pair_region(pair,source,32,resolution)
    for origin in origin_options(base):
        center=np.asarray(origin)+16;previous=np.asarray(origin);previous_side=32
        chain=[{'side':32,'origin':list(origin),'center_displacement':0.}]
        for side in WINDOW_SIDES[1:]:
            options=[]
            for b in pair_region(pair,source,side,resolution):
                lo=np.maximum(b.lower,previous+previous_side-side);hi=np.minimum(b.upper,previous)
                if np.any(lo>hi):continue
                o=np.clip(center-side//2,lo,hi).astype(int)
                distance=float(np.linalg.norm(o+side//2-center));options.append((distance,tuple(o)))
            if not options:break
            distance,o=min(options);chain.append({'side':side,'origin':list(o),'center_displacement':distance})
            previous=np.asarray(o);previous_side=side
        if len(chain)==4:return chain
    return None


def diversity(rows):
    result={}
    for side in WINDOW_SIDES:
        part=[r for r in rows if r['side']==side]
        origins=np.asarray([r['origin'] for r in part],int).reshape(-1,3)
        distances=[];ious=[]
        for a,b in combinations(origins,2):
            distances.append(float(np.linalg.norm(a-b)))
            intersection=int(np.prod(np.maximum(0,side-np.abs(a-b))))
            ious.append(intersection/(2*side**3-intersection))
        result[str(side)]={'windows':len(part),'unique_pairs':len({r['pair_id'] for r in part}),
                           'unique_buckets':len({tuple(r['bucket']) for r in part}),
                           'minimum_center_distance':min(distances,default=None),'maximum_iou':max(ious,default=None),
                           'center_bbox_span':(np.ptp(origins,axis=0).tolist() if len(origins) else None),
                           'pair_reuse':dict(Counter(r['pair_id'] for r in part))}
    return result


def distribution(values):
    v=list(values)
    return {'n':len(v),'min':min(v,default=None),'median':float(np.median(v)) if v else None,
            'p95':float(np.percentile(v,95)) if v else None,'max':max(v,default=None)}


def add_calls(*parts):
    out=Counter()
    for p in parts:out.update(p)
    return dict(out)


def scaled(values,factor):return {k:v*factor for k,v in values.items()}


def analyze_run(directory):
    meta=json.loads((directory/'run.json').read_text())
    if not meta['final_complete']:raise ValueError('Run not yet final: '+str(directory))
    rows=[json.loads(line) for line in (directory/'proposals.jsonl').read_text().splitlines()]
    method=meta['method'];meter=meta['meter'];independent=meta['independent_meter']
    metrics=[];analysis={}
    if method=='guided':
        pairs=json.loads((directory/'pairs.json').read_text());lookup={r['id']:r for r in pairs}
        analysis['diversity']=diversity(rows)
        analysis['discovery']={k:meta['sampling'][k] for k in ('exact_points','labels','global_points','refinement_points','opposite_pairs','wall_seconds','process_seconds','refinement_attempts','cache_hits')}
        labels=meta['sampling']['labels']
        analysis['discovery']['all_combinatorial_opposite_pairs']=labels.get('free_lit',0)*labels.get('free_shadow',0)
        analysis['pair_distances']=distribution(r['distance_voxels'] for r in pairs)
        analysis['pair_max_axis_distances']=distribution(r['max_axis_voxels'] for r in pairs)
        analysis['selected32_pair_distances']=distribution(lookup[r['pair_id']]['distance_voxels'] for r in rows if r['side']==32)
        analysis['selected32_pair_axis_distances']=distribution(lookup[r['pair_id']]['max_axis_voxels'] for r in rows if r['side']==32)
        analysis['origins']={};analysis['multiscale']={}
        selected32={r['pair_id'] for r in rows if r['side']==32}
        for side in WINDOW_SIDES:
            selected_pairs={r['pair_id'] for r in rows if r['side']==side}
            boxes=[b for pairid in selected_pairs for b in pair_region(lookup[pairid],meta['source'],side,1024)]
            analysis['origins'][str(side)]={
                'feasible_pairs':sum(pair['feasible_origins'][str(side)]>0 for pair in pairs),
                'sum_all_indexed_pair_counts_including_overlap':sum(pair['feasible_origins'][str(side)] for pair in pairs),
                'distinct_origins_union_of_selected_pairs':union_origin_count(boxes),
                'actually_proposed':sum(r['side']==side for r in rows),
                'analytic_origin_options_considered':meta['sampling']['by_side'][str(side)]['analytic_origin_options_considered']}
        multiscale=[]
        for pair in pairs:
            if pair['max_axis_voxels']>=32:continue
            common=common_center_regions(pair,meta['source'],1024)
            if common:
                center=origin_options(common)[0]
                multiscale.append({'pair_id':pair['id'],'selected_for_side32':pair['id'] in selected32,
                                   'common_centers':sum(b.count for b in common),'example_center':list(center),
                                   'windows':[{'side':side,'origin':[v-side//2 for v in center]} for side in WINDOW_SIDES],
                                   'positive':pair['positive'],'negative':pair['negative']})
        selected_pairs32=[lookup[pairid] for pairid in sorted(selected32)]
        nearby=[{'pair_id':p['id'],'chain':nearby_nested_chain(p,meta['source'],1024)} for p in selected_pairs32]
        analysis['multiscale']={'indexed_pairs_with_exact_nested_chain':len(multiscale),
                               'selected32_pairs_with_exact_nested_chain':sum(x['selected_for_side32'] for x in multiscale),
                               'selected32_pairs_with_constructed_nearby_nested_chain':sum(x['chain'] is not None for x in nearby),
                               'nearby_examples':nearby,
                               'examples':sorted(multiscale,key=lambda x:(not x['selected_for_side32'],x['pair_id']))[:3]}
        discovery_stages=('witness_discovery_global','witness_discovery_refinement')
        shared_calls=add_calls(*(meter['calls'].get(stage,{}) for stage in discovery_stages))
        discovery_time={k:sum(meter['timings'].get(stage,{}).get(k,0) for stage in discovery_stages) for k in ('wall','process')}
        construction=meter['timings'].get('window_construction',{})
        analysis['cost_decomposition']={'witness_discovery':discovery_time,'window_construction':construction}
    else:
        shared_calls={};discovery_time={'wall':0.,'process':0.};construction={'wall':0.,'process':0.}
        analysis['cost_decomposition']={'uniform_window_first':meter['timings']['uniform_sampling']}
    for side in WINDOW_SIDES:
        part=[r for r in rows if r['side']==side]
        accepted=[r for r in part if r['accepted']]
        final_stage=f'final_validation:{side}';candidate_stage=f'candidate_validation:{side}'
        final_time=independent['timings'].get(final_stage,{});candidate_time=meter['timings'].get(candidate_stage,{})
        if method=='uniform':
            proposal_calls=Counter()
            for row in part:
                for key,info in row['reference_calls'].items():
                    proposal_calls['oracle_calls']+=info['calls']
                    field='classify_box' if key.startswith('certificate.') else key.split('.',1)[-1]
                    proposal_calls[field]+=info['calls']
                proposal_calls['api_certificate']+=int('certificate' in row['stage_seconds'])
            wall=sum(r['elapsed_seconds'] for r in part);cpu=sum(r['process_seconds'] for r in part)
            pairkeys=[tuple(tuple(p) for p in r['witnesses']) for r in meta['windows'] if r['side']==side]
            buckets={tuple(((np.asarray(p)+np.asarray(q)+1)/2//128).astype(int)) for p,q in pairkeys}
            unique_pairs=len(set(pairkeys));unique_buckets=len(buckets)
        else:
            proposal_calls=scaled(shared_calls,.25)
            wall=(discovery_time['wall']+construction.get('wall',0))/4
            cpu=(discovery_time['process']+construction.get('process',0))/4
            unique_pairs=len({r['pair_id'] for r in accepted});unique_buckets=len({tuple(r['bucket']) for r in accepted})
        total_calls=add_calls(proposal_calls,independent['calls'].get(final_stage,{}),meter['calls'].get(candidate_stage,{}))
        total_wall=wall+final_time.get('wall',0)+candidate_time.get('wall',0)
        total_cpu=cpu+final_time.get('process',0)+candidate_time.get('process',0)
        n=len(accepted)
        metric={'run':directory.name,'method':method,'source':meta['source_rank'],'stream':meta['stream'],'side':side,
                'attempted_windows':len(part),'accepted_windows':n,'acceptance_rate':n/len(part) if part else None,
                'unique_witness_pairs_used':unique_pairs,'unique_spatial_buckets_used':unique_buckets,
                'oracle_calls':total_calls.get('oracle_calls',0),'certificate_calls':total_calls.get('api_certificate',0),
                'response_ray_calls':total_calls.get('segment_intersect',0),'membership_calls':total_calls.get('classify_points',0),
                'box_calls':total_calls.get('classify_box',0),'proposal_oracle_calls':proposal_calls.get('oracle_calls',0),
                'allocated_wall_seconds':total_wall,'allocated_process_seconds':total_cpu,
                'seconds_per_accepted_window':total_wall/n if n else None,'queries_per_accepted_window':total_calls.get('oracle_calls',0)/n if n else None,
                'shared_witness_discovery_wall_allocated':discovery_time['wall']/4,
                'shared_window_construction_wall_allocated':construction.get('wall',0)/4,
                'final_validation_wall':final_time.get('wall',0),'candidate_validation_wall':candidate_time.get('wall',0),
                'common_setup_wall_allocated':(meta['setup']['wall']+meta['independent_oracle_setup']['wall'])/4,
                'common_setup_oracle_calls_allocated':meter['calls'].get('source_pool_setup',{}).get('oracle_calls',0)/4}
        metrics.append(metric)
    analysis['cost_decomposition'].update(
        final_validation={k:sum(v.get(k,0) for stage,v in independent['timings'].items() if stage.startswith('final_validation:')) for k in ('wall','process')},
        candidate_source_validation={k:sum(v.get(k,0) for stage,v in meter['timings'].items() if stage.startswith('candidate_validation:')) for k in ('wall','process')},
        common_setup={'wall':meta['setup']['wall']+meta['independent_oracle_setup']['wall'],
                      'process':meta['setup']['process']+meta['independent_oracle_setup']['process']},
        record_validation=meter['timings'].get('record_validation',{}))
    analysis.update(run=directory.name,source=meta['source_rank'],stream=meta['stream'],method=method,validation=meta['validation'])
    return analysis,metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='outputs/boundary-guided-ab');p.add_argument('--partial',action='store_true')
    a=p.parse_args();root=Path(a.root);results=[];metrics=[];pending=[]
    for d in sorted(root.glob('*source-*-stream-*')):
        if not d.is_dir() or d.name.startswith('smoke-'):continue
        r=json.loads((d/'run.json').read_text())
        if not r['final_complete']:
            pending.append(d.name)
            if a.partial:continue
            raise ValueError('Pending: '+d.name)
        analysis,table=analyze_run(d);results.append(analysis);metrics.extend(table)
    summary={'complete':not pending,'runs':results,'metrics':metrics,'pending':pending,
             'allocation':'Guided witness discovery and construction are shared by four equal quotas and charged 1/4 per side. Independent/final candidate work charged directly. Common setup shown separately, not hidden.'}
    prefix='progress' if a.partial else 'summary'
    atomic_json(root/(prefix+'.json'),plain(summary))
    if metrics:
        with (root/(prefix+'-metrics.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    print(json.dumps({'completed_runs':len(results),'pending':pending},indent=2))


if __name__=='__main__':main()
