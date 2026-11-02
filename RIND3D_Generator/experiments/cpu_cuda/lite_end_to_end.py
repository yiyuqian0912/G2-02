"""Actual cold field -> selection -> two physical replays -> loader pilot."""
import argparse
import json
from pathlib import Path
import time
import shutil
import resource
import numpy as np
from rind3d.geometry import load_geometry
from rind3d.cli import load_config
from rind3d.information import InformationIndex
from rind3d.stored_windows import select_stored_windows
from rind3d.pack import RINDPack,atomic_json,sha256_file
from rind3d.pipeline import write_candidate_files,validate_records,parquet_write
from rind3d.search import CandidateBank
from rind3d.window_contract import WINDOW_PROTOCOL,WINDOW_SIDES,balanced_quotas,require_side_counts
from rind3d import RINDDataset
from experiments.shadow_volumes.cases import SOURCES
from experiments.shadow_volumes.renderer import geometry_split
from .cpu import GeometryGroup
from .benchmark import ROOT,OLD,Control,field
from .evidence import build_evidence


def run(method):
    root=ROOT/'lite-end-to-end'/method
    if (root/'result.json').exists():return
    root.mkdir(parents=True,exist_ok=False);path=root/'scenes/scene_box_source0';path.mkdir(parents=True)
    # Only the destination directory creation is excluded from total cold time.
    start=time.perf_counter();cpu=time.process_time();stages={};source=SOURCES[0]
    t=time.perf_counter();bundle=load_geometry(OLD/'cases/box/geometry.npz');g=GeometryGroup(bundle)
    primary=g.bind(source);independent=Control(g,source,'B_native')
    runtime=gg=gpu=None
    if method=='GPU':
        from .cuda_runtime import Runtime
        from .gpu import GPUGroup
        runtime=Runtime(root/'cuda');gg=GPUGroup(g,runtime);gpu=gg.bind(source);active=gpu
    else:active=primary
    stages['geometry_device_and_source_setup_seconds']=time.perf_counter()-t
    try:
        t=time.perf_counter();generated=field(active,path/'field',1024,method)
        assert generated['complete'];stages['full_field_seconds']=time.perf_counter()-t
        t=time.perf_counter()
        for f in ('response.rindb','response.rindb.sha256','solid.rindb','solid.rindb.sha256','information.rindi','information.rindi.sha256'):
            # Rename within the new scene only. Files retain their unchanged
            # source-bound identity. No cross-source hard link is introduced.
            (path/'field'/f).rename(path/f)
        shutil.copy2(OLD/'cases/box/geometry.npz',path/'geometry.npz');source.tofile(path/'driver.f64')
        stages['field_organization_seconds']=time.perf_counter()-t
        cfg=load_config('configs/pilot.yaml')
        with RINDPack(path/'response.rindb',verify_file=True) as response,RINDPack(path/'solid.rindb',verify_file=True) as solid:
            t=time.perf_counter();index=InformationIndex.load(path/'information.rindi',response,solid)
            stages['index_open_seconds']=time.perf_counter()-t
            t=time.perf_counter();selection=select_stored_windows(index,source,256,seed=20261031,scene=0,settings=cfg['window_selection'])
            stages['window_selection_seconds']=time.perf_counter()-t;windows=selection.windows
        require_side_counts((w.size for w in windows),balanced_quotas(256))
        frozen=np.fromfile(OLD/'lite-compatibility/scenes/scene_box_source0/window_records.bin',dtype='<u2').reshape(-1,4)
        assert np.array_equal(np.asarray([w.record() for w in windows]),frozen)
        t=time.perf_counter();proof,replays=build_evidence(path,windows,primary,independent,
            primary_backend='new-cached-parametric-v1',independent_backend='unchanged-native-filter-v1',output=path/'physical-evidence.private.json')
        stages['two_physical_replays_seconds']=time.perf_counter()-t
        t=time.perf_counter();proof.validate_files(path)
        reports=[proof.verify_window(w,primary,source) for w in windows]
        stages['full_window_evidence_validation_seconds']=time.perf_counter()-t
        t=time.perf_counter();bank=CandidateBank();bank.add(source)
        verified=[{'all_grid_exact':True,'independent_validation_passed':True,
                   'search_termination_reason':'declared_zero_optional_search_budget','actual_candidates':1} for _ in windows]
        dtype,counts=write_candidate_files(path,windows,bank,[[0] for _ in windows],verified)
        parquet_write(path/'window_selection.parquet',selection.provenance);parquet_write(path/'uniform_control.parquet',selection.controls)
        meta={'scene_id':path.name,'status':'INCOMPLETE','diagnostic_only':True,'resolution':1024,'max_candidates':8,
              'profile':'RIND3D-Lite-target','window_protocol':WINDOW_PROTOCOL,'contract_version':2,
              'candidate_storage':'pool-offsets-v2','true_source_included':True,'candidate_id_dtype':dtype,
              'min_source_separation':32/1024,'numerical_policy':g.provenance,'geometry_id':g.geometry_id,
              'source_id':primary.source_id,'reason':'Isolated timing pilot; Lite publication validator and new-family acceptance not implemented',
              'sampling':selection.report,'physical_evidence':'physical-evidence.private.json'}
        validate_records(path,meta,strict=False)
        meta['hashes']={f.name:sha256_file(f) for f in path.iterdir() if f.is_file()};atomic_json(path/'verification.json',meta)
        split=geometry_split(g.geometry_id)
        parquet_write(root/'scenes.parquet',[{'scene_id':path.name,'status':'INCOMPLETE','split':split,'families':['experimental_box'],
                     'geometry_id':g.geometry_id,'source_id':primary.source_id,'verification_sha256':sha256_file(path/'verification.json')}])
        parquet_write(root/'splits.parquet',[{'scene_id':path.name,'split':split}])
        parquet_write(root/'windows.parquet',[{'scene_id':path.name,'window_id':i,'size':w.size,'equivalent_count':1} for i,w in enumerate(windows)])
        atomic_json(root/'manifest.json',{'format':'RIND3D','version':2,'diagnostic_only':True,'status':'INCOMPLETE',
                    'window_protocol':WINDOW_PROTOCOL,'window_sides':list(WINDOW_SIDES),'max_candidates':8,'split_group':'geometry_id'})
        with RINDDataset(root,split=None,require_ready=False) as ds:
            assert len(ds)==256;sample=ds[0]
            assert not set(sample)&{'candidate_sources','true_source','witnesses','source_id','physical_evidence'}
            assert ds.private_oracle().get_candidate_sources(sample['scene_id'],sample['window_id']).shape==(1,3)
        with RINDDataset(root,split=None) as ds:assert len(ds)==0
        stages['candidate_metadata_io_loader_seconds']=time.perf_counter()-t
        stages['additional_candidate_search_seconds']=0.
    finally:
        if gpu:gpu.close()
        if gg:gg.close()
        if runtime:runtime.close()
    atomic_json(root/'result.json',{'method':method,'complete_end_to_end_diagnostic':True,'production_ready':False,
                'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'stages':stages,
                'geometry_id':g.geometry_id,'source_id':primary.source_id,'equivalent_count_each':1,'max_candidates':8,
                'side_counts':balanced_quotas(256),'windows':256,'full_field':generated,'evidence_stages':replays['stages'],
                'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'strict_loader_excludes_demo':True,'default_private_labels_hidden':True,
                'scope':'One easy box source, truth only; not representative dataset production throughput'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['CPU_clip','GPU'],default='CPU_clip');a=p.parse_args();run(a.method)
