"""Audit multi-source metadata and sample the installed physical data path."""

import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from rind_dataset.geometry import is_driver_valid_mixed

from rind_phase1_multi.data import Phase1MultiDataset, SizeBucketBatchSampler, make_dataloader, split_scene_ids
from rind_phase1_multi.install_data import validate_installed
from rind_phase1_multi.physics import REFERENCE_ATOL, evaluate_sources


def audit_metadata(dataset):
    """All scenes/views, bitmap invariants, and analytic reference equivalence.

    No full-world float rasters are retained. Full physical rerendering is a
    separate sampled check; this function does not claim to rerender every pixel.
    """
    ds=dataset._base
    histogram={}
    reference_total=0
    reference_max_error=0.0
    for start in range(0,ds.num_scenes,128):
        stop=min(start+128,ds.num_scenes)
        visible=np.asarray(ds.visibility_bits[start:stop])
        obstacle=np.asarray(ds.obstacle_bits[start:stop])
        assert not np.bitwise_and(visible,obstacle[:,None,:]).any(), 'Visibility inside obstacles'
        for offset,sid in enumerate(range(start,stop)):
            scene=ds.get_scene(sid)
            count=scene['driver_count']
            histogram[count]=histogram.get(count,0)+1
            assert not visible[offset,count:].any(), 'Inactive source visibility is nonzero'
            assert not ds.driver_strengths[sid,count:].any(), 'Inactive strengths are nonzero'
            for point in scene['drivers']:
                assert is_driver_valid_mixed(point,scene['obstacle_types'],scene['obstacle_params'],
                                              scene['obstacle_count'],domain_size=ds.global_size)
            leaves=list(dataset.iter_view_leaves(sid))
            assert len(leaves)==len(scene['local_views'])
            true=np.column_stack((scene['drivers'],scene['driver_strengths']))
            nviews=len(leaves)
            for view in scene['local_views']:
                x,y,side=map(int,view)
                end=np.array([x+side,y+side])
                points=scene['drivers']
                upper=(points<end)|((end==ds.global_size)&(points<=end))
                assert not ((points>=[x,y])&upper).all(axis=1).any(), 'Source inside observation'
            number=int(ds.candidate_numbers[sid,0])
            # The current release repeats its analytic witnesses across windows.
            assert np.all(ds.candidate_numbers[sid,:nviews]==number)
            base=np.asarray(ds.view_candidates[sid,0])
            assert np.array_equal(ds.view_candidates[sid,:nviews],np.broadcast_to(base,(nviews,*base.shape)))
            assert np.array_equal(ds.candidate_counts[sid,:nviews],
                                  np.broadcast_to(ds.candidate_counts[sid,0],
                                                  (nviews,ds.candidate_counts.shape[2])))
            witnesses=dataset.get_candidates(sid,0)
            assert np.array_equal(witnesses[0],true)
            reference_total+=nviews*len(witnesses)
            expected={}
            for row in true:
                key=tuple(row[:2])
                expected[key]=expected.get(key,0.0)+float(row[2])
            for witness in witnesses:
                actual={}
                for row in witness:
                    assert np.isfinite(row).all() and 0<=row[2]<=1
                    key=tuple(row[:2])
                    actual[key]=actual.get(key,0.0)+float(row[2])
                assert actual.keys()==expected.keys(), 'Reference introduces a new source position'
                errors=[abs(actual[key]-expected[key]) for key in expected]
                error=max(errors,default=0.0)
                assert error<=REFERENCE_ATOL, 'Reference strength is not conserved'
                reference_max_error=max(reference_max_error,error)
            assert not ds.candidate_numbers[sid,nviews:].any()
            assert not ds.candidate_counts[sid,nviews:].any()
            assert not ds.view_candidates[sid,nviews:].any()
    return {'all_scenes_audited':ds.num_scenes,'all_views_audited':len(dataset),
            'reference_sets_analytically_audited':reference_total,
            'reference_strength_max_error':reference_max_error,'source_count_histogram':histogram}


def check_data(root=None,*,full_audit=False,check_torch=False,workers=0):
    ds=Phase1MultiDataset(root)
    counts=validate_installed(ds.root)
    sample=ds[0]
    assert set(sample)=={'scene_id','view_id','response','obstacle','window'}
    assert sample['response'].dtype==np.float32 and sample['obstacle'].dtype==np.dtype(bool)
    assert sample['response'].shape==sample['obstacle'].shape
    assert not sample['response'][sample['obstacle']].any()
    assert np.array_equal(pickle.loads(pickle.dumps(ds))[0]['response'],sample['response'])
    split=split_scene_ids(ds.num_scenes,ratios=(.8,.1,.1))
    partitions=[set(ids) for ids in split.values()]
    assert not (partitions[0]&partitions[1] or partitions[0]&partitions[2] or partitions[1]&partitions[2])
    assert set.union(*partitions)==set(range(ds.num_scenes))
    audit=audit_metadata(ds) if full_audit else {'all_scenes_audited':0}
    selected=[]
    raw_counts=np.asarray(ds._base.driver_counts)
    for count in np.unique(raw_counts):
        ids=np.flatnonzero(raw_counts==count)
        selected.extend([int(ids[0]),int(ids[-1])])
    selected=sorted(set(selected))
    subset=Phase1MultiDataset(ds.root,scene_ids=selected)
    batches=list(SizeBucketBatchSampler(subset,4))
    assert sorted(i for batch in batches for i in batch)==list(range(len(subset)))
    assert all(len({int(subset[i]['window'][2]) for i in batch})==1 for batch in batches)
    witnesses_checked=0
    max_reference_cost=0.0
    sizes=set()
    over_one=False
    for sid in selected:
        scene=ds.get_scene(sid)
        assert len(list(ds.iter_view_leaves(sid)))==len(scene['local_views'])
        for side in np.unique(scene['local_views'][:,2]):
            vid=int(np.flatnonzero(scene['local_views'][:,2]==side)[0])
            observed=ds.get_observation(sid,vid)
            channels=ds.get_channels(sid,vid)
            raw=ds.get_region(sid,*map(int,observed['window']))['response']
            assert np.array_equal(channels.sum(axis=0),raw)
            assert np.array_equal(raw.astype(np.float32),observed['response'])
            assert channels.shape==(scene['driver_count'],int(side),int(side))
            over_one |= bool((raw>1).any())
            sizes.add(int(side))
            for witness in ds.get_candidates(sid,vid):
                result=evaluate_sources(ds,sid,vid,witness)
                assert result['valid'] and result['physical_cost']<=REFERENCE_ATOL
                max_reference_cost=max(max_reference_cost,result['physical_cost'])
                witnesses_checked+=1
    if ds.manifest['max_drivers']>=3:
        assert over_one, 'Additive responses were clipped or sample selection needs expansion'
    if check_torch:
        for batch in make_dataloader(subset,batch_size=4,num_workers=workers):
            assert str(batch['response'].dtype)=='torch.float32'
            assert str(batch['obstacle'].dtype)=='torch.bool'
            assert batch['response'].shape==batch['obstacle'].shape
            assert bool((batch['window'][:,2]==batch['response'].shape[1]).all())
    report={**counts,**audit,'physical_scenes_sampled':selected,'physical_window_sizes':sorted(sizes),
            'references_rerendered':witnesses_checked,'max_reference_cost':max_reference_cost,
            'response_above_one_observed':over_one,'torch_batches_checked':check_torch}
    print(json.dumps(report,indent=2))
    print('PASS: multi-source installation, channels, local masks, subsets, size batches, trees, witnesses and rerendering.')
    return report


def check_teacher(root=None, *, spacing, temperature, source_counts, samples_per_count,
                  count_prior=None, seed=20260923, backend="auto"):
    """Real joint targets across generating K and L; explicit smoke-test settings.

    Generating K is read to select diagnostic coverage, never to define support.
    Every observation receives the SAME caller-declared candidate count policy.
    """
    from rind_phase1_multi.diagnostics import inspect_teacher
    from rind_phase1_multi.physics import response_disagreement
    from rind_phase1_multi.search import CandidateDomain
    from rind_phase1_multi.teacher import generate_joint_teacher

    dataset = Phase1MultiDataset(root)
    counts = np.asarray(dataset._base.driver_counts)
    selected = [int(np.flatnonzero(counts == k)[0]) for k in np.unique(counts)]
    observations = []
    references_checked = 0
    maximum_reference_cost = maximum_alignment_error = 0.0
    for sid in selected:
        views = dataset.get_scene(sid)["local_views"]
        for side in np.unique(views[:, 2]):
            vid = int(np.flatnonzero(views[:, 2] == side)[0])
            record = generate_joint_teacher(dataset, sid, vid, spacing=spacing, temperature=temperature,
                                            source_counts=source_counts, samples_per_count=samples_per_count,
                                            count_prior=count_prior, seed=seed, backend=backend)
            diagnostic = inspect_teacher(dataset, record, backend=backend)
            assert record["evaluated"].all() and record["valid"].all()
            assert np.isclose(record["teacher_prob"].sum(), 1, rtol=0, atol=1e-12)
            domain = CandidateDomain(tuple(record["window"]), int(dataset.manifest["global_size"]))
            for i, count in enumerate(record["source_counts"]):
                assert domain.contains(record["sources"][i, :int(count), :2]).all()
            raw = dataset.get_region(sid, *map(int, record["window"]))["response"]
            # Check actual position/strength/cost alignment against the uncached renderer.
            indices = sorted({0, len(record["source_counts"]) - 1,
                              int(np.argmax(record["teacher_prob"]))})
            for i in indices:
                parameters = record["sources"][i, :int(record["source_counts"][i])]
                fresh = dataset.rerender(sid, record["window"], parameters, backend=backend)
                error = abs(response_disagreement(fresh, raw) - record["physical_cost"][i])
                assert error <= REFERENCE_ATOL, "Cached cost and fresh full-set render disagree"
                maximum_alignment_error = max(maximum_alignment_error, float(error))
            references_checked += diagnostic["references_checked"]
            maximum_reference_cost = max(maximum_reference_cost, diagnostic["max_reference_cost"])
            row = {"scene_id": sid, "view_id": vid, "generating_count_diagnostic_only": int(counts[sid]),
                   "window_size": int(side), "joint_hypotheses": diagnostic["joint_hypotheses"],
                   "unique_source_renders": diagnostic["num_source_renders"],
                   "minimum_cost": diagnostic["physical_cost_quantiles"][0],
                   "teacher_entropy_nats": diagnostic["teacher_entropy_nats"],
                   "count_distribution": diagnostic["count_distribution"],
                   "config_id": record["metadata"]["config_id"]}
            observations.append(row)
            print(f"PASS joint teacher: scene={sid}, L={side}, hypotheses={row['joint_hypotheses']}, "
                  f"unique visibility renders={row['unique_source_renders']}", flush=True)
    report = {"teacher_observations_checked": len(observations), "scenes": selected,
              "window_sizes": sorted({o["window_size"] for o in observations}),
              "references_rerendered": references_checked, "max_reference_cost": maximum_reference_cost,
              "max_cached_vs_direct_cost_error": maximum_alignment_error,
              "settings": {"spacing": spacing, "temperature": temperature, "source_counts": source_counts,
                           "samples_per_count": samples_per_count, "count_prior": count_prior,
                           "seed": seed, "backend": backend},
              "reference_solutions_used_as_targets": False, "observations": observations}
    print(f"PASS: {len(observations)} completed real joint teachers, all recorded sizes, "
          f"{references_checked} separate reference checks; cached/direct maximum error={maximum_alignment_error:g}.")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--data-only',action='store_true')
    mode.add_argument('--teacher',action='store_true',help='Check real joint targets across source counts and sizes')
    parser.add_argument('--data-root')
    parser.add_argument('--full-audit',action='store_true',help='Audit metadata/bitmaps/references for every scene')
    parser.add_argument('--torch',action='store_true')
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--spacing',type=float)
    parser.add_argument('--temperature',type=float)
    parser.add_argument('--counts',type=int,nargs='+')
    parser.add_argument('--count-prior',type=float,nargs='+')
    parser.add_argument('--samples-per-count',type=int,nargs='+')
    parser.add_argument('--seed',type=int,default=20260923)
    parser.add_argument('--backend',choices=('auto','numba','reference'),default='auto')
    args=parser.parse_args()
    if args.teacher:
        if any(v is None for v in (args.spacing,args.temperature,args.counts,args.samples_per_count)):
            parser.error('--teacher requires explicit spacing, temperature, counts and samples-per-count')
        sizes=args.samples_per_count[0] if len(args.samples_per_count)==1 else args.samples_per_count
        report=check_teacher(args.data_root,spacing=args.spacing,temperature=args.temperature,
                             source_counts=args.counts,count_prior=args.count_prior,samples_per_count=sizes,
                             seed=args.seed,backend=args.backend)
    else:
        report=check_data(args.data_root,full_audit=args.full_audit,check_torch=args.torch,workers=args.workers)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
