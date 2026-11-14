"""Guard the interpretation of the isolated distribution and multi-scale audit."""
import sys
from itertools import product
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from experimental_boundary_guided import OriginBox
from origin_multiscale_analysis import minimum_box_distance,actual_anchor_metrics
from analyze_boundary_ab import common_center_regions
from analyze_origin_distribution import weighted_hist,cluster_bootstrap,compare_reference


def test_closest_integer_center_matches_brute_force():
    boxes=[OriginBox((2,3,4),(4,5,6)),OriginBox((8,1,2),(9,3,4))]
    point=(0,9,1)
    distance,closest=minimum_box_distance(point,boxes)
    brute=[q for box in boxes for q in product(*(range(a,b+1) for a,b in zip(box.lower,box.upper)))]
    expected=min((sum((a-b)**2 for a,b in zip(q,point)),q) for q in brute)
    assert closest==expected[1] and distance==np.sqrt(expected[0])
    assert minimum_box_distance(point,[])==(None,None)


def test_common_center_regions_match_direct_half_open_multiscale_check():
    pair={'positive':[140,140,140],'negative':[168,168,168]}
    source=np.array([.05]*3);resolution=300
    boxes=common_center_regions(pair,source,resolution)
    actual={q for box in boxes for q in product(*(range(a,b+1) for a,b in zip(box.lower,box.upper)))}
    expected=set()
    for center in product(range(153,157),repeat=3):
        valid=True
        for side in (32,64,128,256):
            origin=np.asarray(center)-side//2
            valid &= bool(np.all(origin>=0) and np.all(origin+side<=resolution))
            valid &= all(np.all(np.asarray(p)+.5>=origin) and np.all(np.asarray(p)+.5<origin+side) for p in (pair['positive'],pair['negative']))
            valid &= not bool(np.all(source*resolution>=origin) and np.all(source*resolution<origin+side))
        if valid:expected.add(center)
    assert actual==expected and len(actual)==64
    metrics=actual_anchor_metrics([137]*3,pair,source,resolution)
    assert metrics['actual_anchor_exactly_aligned']
    assert metrics['minimum_shift_to_an_all_scale_common_center_voxels']==0


def test_near_world_edge_nested_chain_does_not_imply_same_center_chain():
    pair={'positive':[5,40,40],'negative':[7,41,41]}
    metrics=actual_anchor_metrics([0,20,20],pair,[.99]*3,300)
    assert not metrics['any_same_center_chain_exists']
    assert not metrics['actual_anchor_exactly_aligned']
    assert metrics['minimum_shift_to_an_all_scale_common_center_voxels'] is None
    assert metrics['anchored_nested_chain_constructed']
    assert metrics['independent_scale_minimum_center_shifts']['256']>0


def test_reference_weighting_does_not_overweight_family_with_more_sources_or_windows():
    rows=[]
    for case in ('one','two','three'):
        rows.extend({'case':case,'family':'abundant','seed':1,'scene':0,'feature':0.} for _ in range(20))
    rows.append({'case':'four','family':'sparse','seed':1,'scene':1,'feature':1.})
    h=weighted_hist(rows,{'one','two','three','four'},'feature',np.array([0,.5,1.]))
    assert np.allclose(h,[.5,.5])


def test_two_geometry_seeds_are_not_reported_as_population_confidence_interval():
    rows=[{'seed':seed,'scene':0,'effect':float(seed)} for seed in (1,2) for _ in range(100)]
    report=cluster_bootstrap(rows,'effect')
    assert report['geometry_clusters']==2 and report['interval95'] is None


def test_sparse_family_cannot_gain_full_weight_from_other_families_reference_support():
    rows=[]
    for scene,family in enumerate(('supported_one','supported_two')):
        for seed in (1,2):
            for source in (0,1):
                for method in 'ABC':
                    rows.extend({'case':f'{seed}-{scene}-{source}','family':family,'seed':seed,'scene':scene,
                                 'method':method,'side':32,'midpoint_face_min_relative':.2} for _ in range(8))
    for method in 'ABC':
        rows.extend({'case':'sparse','family':'unsupported','seed':1,'scene':2,'method':method,'side':32,
                     'midpoint_face_min_relative':0.} for _ in range(4))
    settings={'matched_reference_min_A_per_case':4,'family_reference_min_A':24,'pooled_reference_min_A':40,
              'mixture_uniform_fractions':[0,.1]}
    comparisons,_=compare_reference(rows,settings)
    reference=next(r for r in comparisons if r['side']==32 and r['feature']=='midpoint_face_min_relative')
    assert reference['sufficient_pooled_support'] and reference['reference_windows']==64
    assert reference['excluded_insufficient_family_reference']==['unsupported']
    assert 'sparse' not in reference['support_cases']
