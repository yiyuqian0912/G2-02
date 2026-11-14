"""256-window Lite compatibility demonstration; NEVER publishes READY."""
import json,shutil,time
from pathlib import Path
import numpy as np
from rind3d.cli import load_config
from rind3d.geometry import Oracle,load_geometry,Budget
from rind3d.pack import RINDPack,sha256_file,atomic_json
from rind3d.pipeline import write_candidate_files,validate_records,parquet_write
from rind3d.search import CandidateBank,verify_grid_exact
from rind3d.information import InformationIndex
from rind3d.stored_windows import select_stored_windows
from rind3d.window_contract import WINDOW_SIDES,WINDOW_PROTOCOL,balanced_quotas,require_side_counts
from rind3d import RINDDataset
from .benchmark import ROOT
from .cases import SOURCES
from .renderer import SharedGeometry,geometry_split


def main():
    source=ROOT/'full/box/C-source0';root=ROOT/'lite-compatibility';path=root/'scenes/scene_box_source0';path.mkdir(parents=True,exist_ok=False)
    for name in ('response.rindb','response.rindb.sha256','solid.rindb','solid.rindb.sha256'):shutil.copy2(source/name,path/name)
    shutil.copy2(ROOT/'cases/box/geometry.npz',path/'geometry.npz');SOURCES[0].tofile(path/'driver.f64')
    cfg=load_config('configs/pilot.yaml');cfg.update(max_candidates=8)
    oracle=Oracle(load_geometry(path/'geometry.npz'));a=SharedGeometry(oracle).bind(SOURCES[0]);bank=CandidateBank();bank.add(SOURCES[0]);start=time.perf_counter()
    cpu=time.process_time();reports=[];by_size={}
    with RINDPack(path/'response.rindb',verify_file=True) as response,RINDPack(path/'solid.rindb',verify_file=True) as solid:
        t=time.perf_counter();index=InformationIndex.build(response,solid);index.save(path/'information.rindi');index_seconds=time.perf_counter()-t
        t=time.perf_counter();selection=select_stored_windows(index,SOURCES[0],256,seed=20261031,scene=0,settings=cfg['window_selection']);selection_seconds=time.perf_counter()-t
        windows=selection.windows;require_side_counts((w.size for w in windows),balanced_quotas(256))
        parquet_write(path/'window_selection.parquet',selection.provenance);parquet_write(path/'uniform_control.parquet',selection.controls)
        for i,w in enumerate(windows):
            observed=response.read_window(*w.origin,w.size);t=time.perf_counter()
            # Different implementations: new shadow proof then ORIGINAL
            # canonical convex region filter/fallback, both whole-window.
            for active in (a,oracle):
                result=verify_grid_exact(active,SOURCES[0],SOURCES[0],w,1024,Budget(60,100_000_000),observed=observed)
                assert result['status']=='grid_exact' and result['covered_voxels']==w.size**3
            reports.append({'all_grid_exact':True,'independent_validation_passed':True})
            by_size.setdefault(w.size,[]).append(time.perf_counter()-t)
            if i%16==0:atomic_json(root/'progress.json',{'verified_windows':i+1,'seconds':time.perf_counter()-start})
    dtype,counts=write_candidate_files(path,windows,bank,[[0] for _ in windows],reports)
    meta={'scene_id':path.name,'status':'INCOMPLETE','diagnostic_only':True,'resolution':1024,'max_candidates':8,'profile':'RIND3D-Lite-target',
        'window_protocol':WINDOW_PROTOCOL,'contract_version':2,'candidate_storage':'pool-offsets-v2','true_source_included':True,'candidate_id_dtype':dtype,
        'min_source_separation':32/1024,'numerical_policy':{},'geometry_id':a.shared.geometry_id,'source_id':a.source_id,
        'reason':'Isolated Lite compatibility demo; current production publication validator remains full-profile only',
        'sampling':selection.report,'elapsed_seconds':time.perf_counter()-start}
    validate_records(path,meta,strict=False);meta['hashes']={p.name:sha256_file(p) for p in path.iterdir() if p.is_file()};atomic_json(path/'verification.json',meta)
    split=geometry_split(a.shared.geometry_id)
    row={'scene_id':path.name,'status':'INCOMPLETE','split':split,'families':['experimental_box'],'geometry_id':a.shared.geometry_id,'source_id':a.source_id,'verification_sha256':sha256_file(path/'verification.json')}
    parquet_write(root/'scenes.parquet',[row]);parquet_write(root/'splits.parquet',[{'scene_id':path.name,'split':split}])
    parquet_write(root/'windows.parquet',[{'scene_id':path.name,'window_id':i,'size':w.size,'equivalent_count':1} for i,w in enumerate(windows)])
    atomic_json(root/'manifest.json',{'format':'RIND3D','version':2,'diagnostic_only':True,'window_protocol':WINDOW_PROTOCOL,'window_sides':list(WINDOW_SIDES),'max_candidates':8,'split_group':'geometry_id','status':'INCOMPLETE'})
    with RINDDataset(root,split=None,require_ready=False,response_format='bool') as ds:
        assert len(ds)==256
        sample=ds[0];assert not set(sample)&{'candidate_sources','true_source','witnesses','source_id'}
        assert ds.private_oracle().get_candidate_sources(sample['scene_id'],sample['window_id']).shape==(1,3)
    with RINDDataset(root,split=None) as ds:assert len(ds)==0
    atomic_json(root/'result.json',{'complete_compatibility_check':True,'publication_ready':False,'windows':256,'size_counts':balanced_quotas(256),'actual_candidates_each':1,'max_candidates':8,
        'index_seconds':index_seconds,'selection_seconds':selection_seconds,'verification_by_side_seconds':by_size,'total_seconds':time.perf_counter()-start,'process_seconds':time.process_time()-cpu,'default_private_labels_hidden':True,'strict_loader_excludes_demo':True,'independent_verifier':'original canonical convex filter+CGAL fallback','production_profile_blocker':meta['reason']})


if __name__=='__main__':main()
