# Legacy four-side diagnostic/report reproduction only; not the current dataset contract.
"""Cohort accounting, paired origin effects, support-aware distributions.

No data from failed windows or unqueried voxels is promoted to exact truth.
All optional mixtures are empirical marginal simulations, not production code.
"""
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path
import argparse,csv,json,math
import numpy as np
from rind3d.pack import atomic_json
from experimental_boundary_guided import pair_region
from full_origin_selector import pair_origin_candidates
from boundary_experiment_support import plain
from analyze_boundary_ab import common_center_regions,nearby_nested_chain
from origin_multiscale_analysis import actual_anchor_metrics

SIDES=(32,64,128,256)
FEATURE_RANGES={'midpoint_face_min_relative':(0,.5),'witness_face_min_relative':(0,.5),
                'source_window_distance':(0,math.sqrt(3)),'source_center_distance':(0,math.sqrt(3)),
                'window_world_boundary_clearance':(0,.5),'lit_fraction':(0,1),
                'shadow_fraction':(0,1),'solid_fraction':(0,1),'free_fraction':(0,1),
                'pair_distance_relative':(0,math.sqrt(3))}


def desc(values):
    values=[float(x) for x in values if x is not None and np.isfinite(x)]
    return {'n':len(values),'mean':float(np.mean(values)) if values else None,
            'median':float(np.median(values)) if values else None,
            'q25':float(np.quantile(values,.25)) if values else None,
            'q75':float(np.quantile(values,.75)) if values else None,
            'min':min(values,default=None),'max':max(values,default=None)}


def delta(after,before):
    result={}
    for section in ('timings','calls'):
        result[section]={}
        for stage,values in after[section].items():
            result[section][stage]={k:v-before.get(section,{}).get(stage,{}).get(k,0) for k,v in values.items()}
    return result


def sum_stages(report,section,stages):
    out=Counter()
    for stage in stages:out.update(report[section].get(stage,{}))
    return dict(out)


def method_metrics(case,method,source_dir):
    meta=case['methods'][method];d=delta(meta['meter_after'],meta['meter_before'])
    independent=delta(meta['independent_after'],meta['independent_before'])
    bd=delta(case['methods']['B']['meter_after'],case['methods']['B']['meter_before'])
    shared=('witness_discovery_global','witness_discovery_refinement','window_construction')
    proposal=sum_stages(bd,'calls',shared) if method=='C' else {}
    actual=Counter();
    for values in (*d['calls'].values(),*independent['calls'].values()):actual.update(values)
    charged=actual+Counter(proposal)
    extra=case['methods']['B']['proposal_wall'] if method=='C' else 0
    extra_cpu=case['methods']['B']['proposal_process'] if method=='C' else 0
    stage_times={}
    for label,stages,report in (
        ('witness_discovery',('witness_discovery_global','witness_discovery_refinement'),bd if method=='C' else d),
        ('construction',('window_construction',),bd if method=='C' else d),
        ('independent_witness_validation',tuple(k for k in independent['timings'] if k.startswith('final_validation:')),independent),
        ('candidate_validation',tuple(k for k in d['timings'] if k.startswith('candidate_validation:')),d)):
        stage_times[label]=sum_stages(report,'timings',stages)
    if method=='C':
        for key,val in d['timings'].get('C_origin_selection',{}).items():
            stage_times['construction'][key]=stage_times['construction'].get(key,0)+val
    stage_times['selector_C_only']=d['timings'].get('C_origin_selection',{})
    stage_times['uniform_sampling']=d['timings'].get('A_uniform_sampling',{})
    quota_times={side:{'wall':meta['proposal_wall']+extra+value['validation_wall'],
                       'process':meta['proposal_process']+extra_cpu+value['validation_process']}
                 for side,value in meta['validation']['validation_quota_times'].items()}
    side32=quota_times.get('32',{}).get('wall')
    total=meta['method_execution_wall']+extra;total_cpu=meta['method_execution_process']+extra_cpu
    result={'case':f"{case['seed']}-{case['scene']}-{case['source_rank']}",
            'family':' + '.join(case['families']),'seed':case['seed'],'scene':case['scene'],'source_rank':case['source_rank'],
            'method':method,'windows':meta['validation']['window_count'],'proposed_windows':meta['validation']['proposed_window_count'],
            'side_counts':meta['validation']['side_counts'],'scene_quota_complete':meta['validation']['pilot_quota_complete'],
            'sampling_fatal':meta['sampling_fatal'],'validation_fatal':meta['validation']['fatal_validation_failure'],
            'center_octants':meta['validation']['placement']['occupied_octants'],
            'wall_cold_equivalent':total,'process_cold_equivalent':total_cpu,
            'wall_actually_executed':meta['method_execution_wall'],'shared_B_proposal_wall_charged':extra,
            'time_to_eight_side32_windows':side32,
            'time_to_eight_windows_by_side':quota_times,
            'time_to_32_window_scene_quota':total if meta['validation']['pilot_quota_complete'] else None,
            'actual_calls':dict(actual),'cold_equivalent_calls':dict(charged),'stages':stage_times,
            'common_case_setup_wall':case['case_setup_wall']+case['scene_setup_share_wall'],
            'common_case_setup_process':case['case_setup_process']+case['scene_setup_share_process'],
            'production_ready':False,'geometry_ready_capable':meta['validation']['geometry_ready_capable']}
    return result


def group_distribution(rows):
    summaries={key:desc(r.get(key) for r in rows) for key in FEATURE_RANGES}
    n=len(rows);known=[r.get('center_near_solid_r32') for r in rows]
    near=sum(x is True for x in known);far=sum(x is False for x in known);unknown=n-near-far
    per_case=defaultdict(list)
    for r in rows:per_case[r['case']].append(r)
    spacing=[];iou=[]
    for part in per_case.values():
        for a,b in combinations(part,2):
            shift=np.abs(np.asarray(a['origin'])-b['origin']);side=a['side']
            spacing.append(float(np.linalg.norm(shift)))
            intersection=int(np.prod(np.maximum(0,side-shift)))
            iou.append(intersection/(2*side**3-intersection))
    buckets=Counter(tuple(r['center_bucket128']) for r in rows)
    p=np.asarray(list(buckets.values()),dtype=float);p=p/p.sum() if len(p) else p
    return {'windows':n,'source_cases':len(per_case),'geometry_clusters':len({(r['seed'],r['scene']) for r in rows}),
            'geometry_seeds':sorted({r['seed'] for r in rows}),'features':summaries,
            'midpoint_within_5pct_face':sum(r['midpoint_face_min_relative']<.05 for r in rows)/n if n else None,
            'center_mean_xyz':np.mean([r['center_world'] for r in rows],axis=0).tolist() if n else None,
            'positive_relative_mean':np.mean([r['positive_relative'] for r in rows],axis=0).tolist() if n else None,
            'negative_relative_mean':np.mean([r['negative_relative'] for r in rows],axis=0).tolist() if n else None,
            'midpoint_relative_mean':np.mean([r['midpoint_relative'] for r in rows],axis=0).tolist() if n else None,
            'center_bucket_count':len(buckets),'center_bucket_entropy_bits':float(-(p*np.log2(p)).sum()) if len(p) else None,
            'pairwise_center_distance_voxels':desc(spacing),'pairwise_iou':desc(iou),
            'center_near_solid_r32':{'near_certified':near,'far_certified':far,'unresolved':unknown,
                'near_fraction_lower_bound':near/n if n else None,'near_fraction_upper_bound':(near+unknown)/n if n else None},
            'composition_kinds':dict(Counter(r.get('composition_kind','PENDING') for r in rows)),
            'composition_status':dict(Counter(r.get('composition_status','PENDING') for r in rows)),
            'local_leaf_aabb_touch_count':desc(r.get('finite_leaf_aabb_touch_count') for r in rows),
            'local_leaf_aabb_touch_density':desc(r.get('finite_leaf_aabb_touch_density') for r in rows),
            'low_free_probe_support':sum(r.get('free_n') is not None and r['free_n']<8 for r in rows)}


def cluster_bootstrap(rows,field,seed=20261030,repeats=2000):
    by_cluster=defaultdict(list)
    for r in rows:
        if r.get(field) is not None:by_cluster[(r['seed'],r['scene'])].append(r[field])
    values=np.array([np.mean(v) for _,v in sorted(by_cluster.items())])
    if len(values)<4:return {'geometry_clusters':len(values),'mean_delta':float(values.mean()) if len(values) else None,'interval95':None,'note':'too few geometry clusters for an interval'}
    rng=np.random.default_rng(seed);draws=values[rng.integers(0,len(values),(repeats,len(values)))].mean(axis=1)
    return {'geometry_clusters':len(values),'mean_delta':float(values.mean()),'interval95':np.quantile(draws,[.025,.975]).tolist(),
            'note':'descriptive bootstrap of tested geometry clusters, not a family-population guarantee'}


def paired_origin_effect(rows,repeats):
    index={(r['case'],r['side'],r['pair_id'],r['method']):r for r in rows if r['method'] in ('B','C')}
    paired=[]
    for key,b in index.items():
        case,side,pair,method=key
        if method!='B':continue
        c=index.get((case,side,pair,'C'))
        if c is None:continue
        row={'case':case,'seed':b['seed'],'scene':b['scene'],'family':b['family'],'source_rank':b['source_rank'],'side':side,'pair_id':pair,
             'delta_midpoint_face':c['midpoint_face_min_relative']-b['midpoint_face_min_relative'],
             'delta_edge_indicator':float(c['midpoint_face_min_relative']<.05)-float(b['midpoint_face_min_relative']<.05)}
        for feature in ('source_window_distance','window_world_boundary_clearance','lit_fraction','solid_fraction'):
            row['delta_'+feature]=c.get(feature)-b.get(feature) if c.get(feature) is not None and b.get(feature) is not None else None
        paired.append(row)
    pooled={str(side):{field:cluster_bootstrap([r for r in paired if r['side']==side],field,repeats=repeats)
                        for field in ('delta_midpoint_face','delta_edge_indicator','delta_source_window_distance','delta_window_world_boundary_clearance','delta_lit_fraction','delta_solid_fraction')}
            for side in SIDES}
    return paired,pooled


def paired_spatial_coverage(rows,repeats):
    groups=defaultdict(lambda:defaultdict(dict))
    for row in rows:
        if row['method'] in ('B','C'):groups[(row['case'],row['side'])][row['method']][row['pair_id']]=row
    effects=[]
    for (case,side),methods in sorted(groups.items()):
        b=methods['B'];c=methods['C'];keys=sorted(b.keys()&c.keys())
        if len(keys)<2:continue
        statistics={}
        for method,lookup in (('B',b),('C',c)):
            distances=[];overlaps=[]
            for ka,kb in combinations(keys,2):
                shift=np.abs(np.asarray(lookup[ka]['origin'])-lookup[kb]['origin'])
                distances.append(float(np.linalg.norm(shift)))
                intersection=int(np.prod(np.maximum(0,side-shift)))
                overlaps.append(intersection/(2*side**3-intersection))
            statistics[method]={'spacing':float(np.mean(distances)),'iou':float(np.mean(overlaps)),
                                'buckets':len({tuple(lookup[k]['center_bucket128']) for k in keys})}
        r=b[keys[0]]
        effects.append({'case':case,'side':side,'seed':r['seed'],'scene':r['scene'],'family':r['family'],
                        'matched_witness_pairs':len(keys),**{f'delta_{k}':statistics['C'][k]-statistics['B'][k] for k in ('spacing','iou','buckets')}})
    summaries={str(side):{field:cluster_bootstrap([r for r in effects if r['side']==side],field,repeats=repeats)
                         for field in ('delta_spacing','delta_iou','delta_buckets')} for side in SIDES}
    return effects,summaries


def weighted_hist(rows,cases,feature,bins):
    grouping=defaultdict(list)
    for r in rows:
        if r['case'] in cases and r.get(feature) is not None:grouping[r['case']].append(r)
    case_rows={r['case']:r for r in rows if r['case'] in cases};families=defaultdict(lambda:defaultdict(list))
    for case in grouping:
        r=case_rows[case];families[r['family']][(r['seed'],r['scene'])].append(case)
    hist=np.zeros(len(bins)-1)
    for family,geometries in families.items():
        for geometry,keys in geometries.items():
            for case in keys:
                vals=[r[feature] for r in grouping[case]]
                weight=1/(len(families)*len(geometries)*len(keys)*len(vals))
                hist+=np.histogram(vals,bins=bins)[0]*weight
    return hist/hist.sum() if hist.sum() else hist


def jsd(a,b):
    m=(a+b)/2;value=0.
    for x in (a,b):
        mask=x>0;value+=float((x[mask]*np.log2(x[mask]/m[mask])).sum())/2
    return value


def compare_reference(rows,settings):
    comparisons=[];mixtures=[]
    for side in SIDES:
        part=[r for r in rows if r['side']==side]
        for feature,limits in FEATURE_RANGES.items():
            bins=np.linspace(*limits,21);by_method={m:[r for r in part if r['method']==m] for m in 'ABC'}
            counts={m:Counter(r['case'] for r in by_method[m] if r.get(feature) is not None) for m in 'ABC'}
            cases={case for case,n in counts['A'].items() if n>=settings['matched_reference_min_A_per_case'] and counts['B'][case]>0 and counts['C'][case]>0}
            # A handful of A points from one family must not receive an entire
            # equal-family weight just because other families have good support.
            family_support=defaultdict(lambda:{'n':0,'seeds':set()})
            case_family={r['case']:r['family'] for r in by_method['A']}
            for r in by_method['A']:
                if r['case'] in cases and r.get(feature) is not None:
                    family_support[r['family']]['n']+=1;family_support[r['family']]['seeds'].add(r['seed'])
            supported_families={family for family,support in family_support.items()
                                if support['n']>=settings['family_reference_min_A'] and len(support['seeds'])>=2}
            excluded_families=sorted({r['family'] for r in part}-supported_families)
            cases={case for case in cases if case_family[case] in supported_families}
            n_a=sum(counts['A'][case] for case in cases)
            geometry={(r['seed'],r['scene']) for r in by_method['A'] if r['case'] in cases}
            sufficient=n_a>=settings['pooled_reference_min_A'] and len(geometry)>=4
            h={m:weighted_hist(by_method[m],cases,feature,bins) for m in 'ABC'}
            item={'side':side,'feature':feature,'reference_windows':n_a,'common_cases':len(cases),'geometry_clusters':len(geometry),
                  'windows_by_method':{m:sum(counts[m][case] for case in cases) for m in 'ABC'},
                  'sufficient_pooled_support':sufficient,'excluded_A_cases_with_fewer_than_four':sum(n<settings['matched_reference_min_A_per_case'] for n in counts['A'].values()),
                  'supported_families':sorted(supported_families),'excluded_insufficient_family_reference':excluded_families,
                  'JS_B_to_A':jsd(h['B'],h['A']) if cases else None,'JS_C_to_A':jsd(h['C'],h['A']) if cases else None,
                  'histograms':{m:h[m].tolist() for m in 'ABC'},'bin_edges':bins.tolist(),
                  'support_cases':sorted(cases),'note':'family/geometry/source-case balanced; A>=4 per case, A>=24 and both seeds per family; finite-probe reference'}
            comparisons.append(item)
            if sufficient:
                for uniform in settings['mixture_uniform_fractions']:
                    mixture=(1-uniform)*h['C']+uniform*h['A']
                    mixtures.append({'side':side,'feature':feature,'uniform_fraction':uniform,'C_fraction':1-uniform,
                        'JS_to_empirical_A':jsd(mixture,h['A']),
                        'cdf_L1_to_A':float(np.abs(np.cumsum(mixture)-np.cumsum(h['A'])).mean()),
                        'reference_windows':n_a,'geometry_clusters':len(geometry),
                        'note':'marginal simulation reusing fitted A; no new observations or production batch sampler'})
    return comparisons,mixtures


def csv_write(path,rows):
    if not rows:path.write_text('');return
    keys=sorted({k for row in rows for k in row})
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def analyze(root,partial=False):
    settings=json.loads(Path('configs/experimental-origin-distribution.json').read_text());rows=[];metrics=[];cases=[];pending=[];quota_rows=[];multiscale=[];conflicts=[]
    calibration={}
    cp=root/'composition-calibration.json'
    if cp.exists():
        for r in json.loads(cp.read_text())['results']:
            if r['status']=='EXACT_COMPLETE':calibration[(r['seed'],r['scene'],r['source_rank'],r['method'],r['side'],tuple(r['origin']))]=r
    for seed in settings['geometry_seeds']:
        for scene in range(8):
            base=root/f'seed-{seed}-scene-{scene}'
            for rank in settings['source_ranks']:
                d=base/f'source-{rank}';p=d/'case.json'
                if not p.exists():pending.append(str(d));continue
                case=json.loads(p.read_text());case_id=f'{seed}-{scene}-{rank}'
                if not case['sampling_complete']:pending.append(case_id);continue
                if not case['audit_complete']:pending.append(case_id+':audit')
                cases.append(case)
                b_points=[]
                if (d/'B/points.jsonl').exists():b_points=[json.loads(line) for line in (d/'B/points.jsonl').read_text().splitlines()]
                labels=Counter(r['label'] for r in b_points)
                b_pairs=json.loads((d/'B/pairs.json').read_text()) if (d/'B/pairs.json').exists() else []
                b_lookup={p['id']:p for p in b_pairs}
                for method in 'ABC':
                    metric=method_metrics(case,method,d);metrics.append(metric)
                    for line in (d/method/'features.jsonl').read_text().splitlines():
                        row=json.loads(line);row.update(case=case_id,family=' + '.join(case['families']))
                        exact=calibration.get((seed,scene,rank,method,row['side'],tuple(row['origin'])))
                        if exact:
                            for key in ('lit_fraction','shadow_fraction','solid_fraction','free_fraction','composition_n','free_n','lit_free_n'):
                                row['estimator_'+key]=row.get(key);row[key]=exact[key]
                            row.update(composition_kind='exact_calibration',composition_status='COMPLETE')
                        rows.append(row)
                    meta=case['methods'][method]
                    for side in SIDES:
                        local=[p for p in b_pairs if p.get('feasible_origins',{}).get(str(side),0)>0]
                        regions=sum(len(pair_region(p,case['source'],side,1024)) for p in local)
                        volume=sum(box.count for p in local for box in pair_region(p,case['source'],side,1024))
                        accepted=[r for r in rows if r['case']==case_id and r['method']==method and r['side']==side]
                        c_state=case['methods']['C']['sampler_report']['by_side'][str(side)]
                        quota_rows.append({'case':case_id,'family':metric['family'],'seed':seed,'scene':scene,'source_rank':rank,'method':method,'side':side,
                            'accepted':len(accepted),'quota_achieved':len(accepted)==8,'scene_quota_complete':metric['scene_quota_complete'],
                            'time_to_side_quota':metric['time_to_eight_windows_by_side'].get(str(side),{}).get('wall'),
                            'process_to_side_quota':metric['time_to_eight_windows_by_side'].get(str(side),{}).get('process'),
                            'time_to_side32_quota':metric['time_to_eight_side32_windows'] if side==32 else None,
                            'exact_discovery_points':len(b_points) if method!='A' else None,'free_lit_witnesses':labels['free_lit'] if method!='A' else None,
                            'free_shadow_witnesses':labels['free_shadow'] if method!='A' else None,'discovered_usable_pairs':len(local) if method!='A' else None,
                            'discovered_legal_origin_boxes':regions if method!='A' else None,
                            'discovered_origin_count_with_pair_multiplicity':volume if method!='A' else None,
                            'selected_exact_witness_points':len({tuple(p) for r in accepted for p in r['witnesses']}),
                            'C_sampled_candidate_origins':c_state['candidate_origins'] if method=='C' else None,
                            'C_assignment_nodes':c_state['nodes'] if method=='C' else None,
                            'C_assignment_complete':c_state['complete'] if method=='C' else None,
                            'C_assignment_budget_exhausted':c_state['node_budget_exhausted'] if method=='C' else None,
                            'distinct_selected_pairs':len({r['pair_id'] for r in accepted}),
                            'witness_buckets':len({tuple(r['witness_bucket128']) for r in accepted}),
                            'center_buckets':len({tuple(r['center_bucket128']) for r in accepted})})
                    if method in 'BC':
                        accepted=[r for r in rows if r['case']==case_id and r['method']==method and r['side']==32]
                        exact=nearby=0;displacements=[];anchor_rows=[]
                        for r in accepted:
                            p=b_lookup[r['pair_id']];exact+=bool(common_center_regions(p,case['source'],1024))
                            chain=nearby_nested_chain(p,case['source'],1024)
                            if chain:
                                nearby+=1;displacements.append(max(x['center_displacement'] for x in chain))
                            anchor_rows.append({'pair_id':r['pair_id'],**actual_anchor_metrics(r['origin'],p,case['source'])})
                        multiscale.append({'case':case_id,'family':metric['family'],'seed':seed,'scene':scene,'source_rank':rank,'method':method,
                            'selected32_pairs':len(accepted),'same_center_feasible':exact,'clipped_center_nested_constructed':nearby,
                            'clipped_chain_max_displacement':desc(displacements),
                            'actual_anchor_aligned':sum(r['actual_anchor_exactly_aligned'] for r in anchor_rows),
                            'minimum_common_center_shift':desc(r['minimum_shift_to_an_all_scale_common_center_voxels'] for r in anchor_rows),
                            'actual_anchor_metrics':anchor_rows,
                            'note':'feasibility depends on pair/source, not its chosen origin; nearby construction is not a global displacement optimum'})
                c_side=case['methods']['C']['sampler_report']['by_side']
                b_proposals=[json.loads(line) for line in (d/'B/proposals.jsonl').read_text().splitlines()] if (d/'B/proposals.jsonl').exists() else []
                for side in SIDES:
                    if c_side[str(side)]['complete']:continue
                    old=[p for p in b_proposals if p['side']==side];min_sep=max(64,side/2)
                    for a,b in combinations(old,2):
                        pa,pb=b_lookup[a['pair_id']],b_lookup[b['pair_id']]
                        da,ra=pair_origin_candidates(pa,case['source'],side,1024,settings,seed,scene,settings['stream'])
                        db,rb=pair_origin_candidates(pb,case['source'],side,1024,settings,seed,scene,settings['stream'])
                        distances=((np.asarray(da)[:,None]-np.asarray(db))**2).sum(axis=-1)
                        compatible=int(np.count_nonzero(distances>=min_sep**2))
                        if compatible:continue
                        upper=max(sum(max(abs(a-d),abs(c-b))**2 for a,b,c,d in zip(x.lower,y.lower,x.upper,y.upper)) for x in ra for y in rb)
                        conflicts.append({'case':case_id,'family':' + '.join(case['families']),'side':side,'pair_a':a['pair_id'],'pair_b':b['pair_id'],
                            'sampled_compatible_combinations':0,'sampled_combinations':len(da)*len(db),
                            'maximum_region_distance':float(np.sqrt(upper)),'required_distance':min_sep,
                            'B_distance':float(np.linalg.norm(np.asarray(a['origin'])-b['origin']))})
    if pending and not partial:raise ValueError('Pending work: '+str(pending))
    families=[' + '.join(v) for v in json.loads((root/f"seed-{settings['geometry_seeds'][0]}-scene-0/scene.json").read_text())['config']['families']]
    grouped={};family_rows=[]
    for family in families:
        for side in SIDES:
            key=f'{family}|{side}';grouped[key]={}
            for method in 'ABC':
                part=[r for r in rows if r['family']==family and r['side']==side and r['method']==method]
                summary=group_distribution(part);summary['uniform_reference_adequate_descriptive']=summary['windows']>=settings['family_reference_min_A'] and len(summary['geometry_seeds'])>=2 if method=='A' else None
                grouped[key][method]=summary
                family_rows.append({'family':family,'side':side,'method':method,'windows':summary['windows'],'geometry_clusters':summary['geometry_clusters'],
                    'midpoint_face_mean':summary['features']['midpoint_face_min_relative']['mean'],'midpoint_face_median':summary['features']['midpoint_face_min_relative']['median'],
                    'midpoint_edge_rate_5pct':summary['midpoint_within_5pct_face'],'source_window_distance_mean':summary['features']['source_window_distance']['mean'],
                    'world_boundary_clearance_mean':summary['features']['window_world_boundary_clearance']['mean'],
                    'lit_fraction_estimate_mean':summary['features']['lit_fraction']['mean'],'solid_fraction_estimate_mean':summary['features']['solid_fraction']['mean'],
                    'pairwise_center_distance_mean':summary['pairwise_center_distance_voxels']['mean'],'pairwise_iou_mean':summary['pairwise_iou']['mean'],
                    'reference_support':summary['uniform_reference_adequate_descriptive'],'clearance':summary['center_near_solid_r32']})
    paired,bootstrap=paired_origin_effect(rows,settings['bootstrap_geometry_clusters'])
    coverage,coverage_bootstrap=paired_spatial_coverage(rows,settings['bootstrap_geometry_clusters'])
    references,mixtures=compare_reference(rows,settings)
    result={'complete':not pending,'source_cases':len(cases),'pending':pending,'settings':settings,'metrics':metrics,
        'family_side_distributions':grouped,'pooled_by_side':{str(side):{m:group_distribution([r for r in rows if r['side']==side and r['method']==m]) for m in 'ABC'} for side in SIDES},
        'paired_origin_bootstrap':bootstrap,'paired_spatial_coverage_bootstrap':coverage_bootstrap,
        'references':references,'mixtures':mixtures,'multiscale':multiscale,'origin_conflicts':conflicts}
    prefix='progress' if partial else 'summary'
    atomic_json(root/(prefix+'.json'),plain(result));csv_write(root/(prefix+'-features.csv'),rows)
    csv_write(root/(prefix+'-efficiency.csv'),[{**r,**{k:v for k,v in r['cold_equivalent_calls'].items()}} for r in metrics])
    csv_write(root/(prefix+'-quotas.csv'),quota_rows);csv_write(root/(prefix+'-family-side.csv'),family_rows)
    csv_write(root/(prefix+'-paired-effects.csv'),paired);csv_write(root/(prefix+'-mixtures.csv'),mixtures)
    csv_write(root/(prefix+'-paired-coverage.csv'),coverage)
    csv_write(root/(prefix+'-multiscale.csv'),multiscale);csv_write(root/(prefix+'-conflicts.csv'),conflicts)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--partial',action='store_true');a=p.parse_args()
    result=analyze(Path('outputs/origin-distribution-audit'),a.partial)
    print(json.dumps({'source_cases':result['source_cases'],'pending':len(result['pending']),'complete':result['complete']},indent=2))


if __name__=='__main__':main()
