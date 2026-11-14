"""Independent end-to-end diagnostic scene; never impersonates production READY."""
import json,multiprocessing,pickle
from pathlib import Path
import numpy as np
import pytest
from rind3d.benchmark import fixture_box
from rind3d.data import RINDDataset,dataset_stats,family_matches
from rind3d.geometry import Oracle,save_geometry
from rind3d.pack import atomic_json,sha256_file,PackedWindow,RINDPack
from rind3d.pipeline import parquet_write,write_candidate_files,validate_records,split_for
from rind3d.search import CandidateBank
from rind3d.stream import stream_response
from rind3d.windows import Window,informative


def read_in_worker(dataset,index):
    ex=dataset[index]
    return ex['window_size'],ex['observed_response'].unpack().sum(),dataset._pid

_FORK_DATASET=None
def read_in_fork(index):return read_in_worker(_FORK_DATASET,index)


@pytest.fixture(scope='module')
def diagnostic(tmp_path_factory):
    root=tmp_path_factory.mktemp('diagnostic');path=root/'scenes'/'scene_000000';path.mkdir(parents=True)
    atomic_json(root/'manifest.json',{'format':'RIND3D','version':2,'diagnostic_only':True})
    bundle=fixture_box();save_geometry(path/'geometry.npz',bundle);oracle=Oracle(bundle)
    truth=np.array([65317,193993,337117],dtype=np.float64)/2**20;truth.astype('<f8').tofile(path/'driver.f64')
    cfg={'resolution':256,'generation_seconds':60,'generation_points':20_000_000,'workers':2,'queue_size':4,'leaf_side':8}
    stream_response(oracle,truth,path,cfg)
    windows=[Window((128,0,0),64),Window((128,32,16),64),Window((96,0,0),128),Window((64,0,0),128)]
    bank=CandidateBank();bank.add(truth)
    dtype,counts=write_candidate_files(path,windows,bank,[[0] for _ in windows],[{} for _ in windows])
    meta={'scene_id':path.name,'status':'INCOMPLETE','resolution':256,'candidate_storage':'pool-offsets-v2','contract_version':2,'max_candidates':64,'true_source_included':True,'candidate_id_dtype':dtype,'min_source_separation':.03125,
          'numerical_policy':{},'reason':'Small-grid API fixture; production resolution and window count are unmet'}
    validate_records(path,meta,strict=False)
    meta['hashes']={p.name:sha256_file(p) for p in path.iterdir() if p.is_file()}
    atomic_json(path/'verification.json',meta)
    parquet_write(root/'scenes.parquet',[{'scene_id':path.name,'status':'INCOMPLETE','split':'train','families':['csg','slabs'],
        'solid_fraction':.0625,'verification_sha256':sha256_file(path/'verification.json'),'bytes':sum(p.stat().st_size for p in path.iterdir())}])
    parquet_write(root/'splits.parquet',[{'scene_id':path.name,'split':'train'}])
    parquet_write(root/'windows.parquet',[{'scene_id':path.name,'window_id':i,'size':w.size,'equivalent_count':1} for i,w in enumerate(windows)])
    return root


def test_model_view_and_explicit_private_access(diagnostic):
    assert len(RINDDataset(diagnostic))==0  # strict is the public default
    with RINDDataset(diagnostic,require_ready=False) as ds:
        sample=ds[0];sid=sample['scene_id']
        assert set(sample)=={'scene_id','window_id','window_origin','window_size','structure','observed_response'}
        assert isinstance(sample['observed_response'],PackedWindow)
        assert sample['observed_response'].shape==(64,)*3 and sample['observed_response'].nbytes==32768
        assert not any(k.startswith(('scene_','physics_')) for k in sample['structure']['arrays'])
        assert all(not a.flags.writeable for a in sample['structure']['arrays'].values())
        private=ds.private_oracle();truth=private.get_true_source(sid)
        assert np.array_equal(private.get_candidate_sources(sid,0),truth[None])
        expected=private.rerender(sid,truth,sample['window_origin'],64)
        assert np.array_equal(expected,sample['observed_response'].unpack())
        assert np.array_equal(private.read_global_brick(sid,(2,0,0)),expected)
        assert len(ds._open(sid)['response'].cache)<=16
    with RINDDataset(diagnostic,require_ready=False,response_format='bool',view='candidate_conditioned') as ds:
        assert ds[0]['observed_response'].dtype==np.bool_ and ds[0]['candidate_sources'].shape==(1,3)


def test_filters_and_reweighting(diagnostic):
    assert family_matches(['csg','slabs'],['csg'],'any')
    assert family_matches(['csg','slabs'],['csg'],'all')
    assert not family_matches(['csg','slabs'],['csg'],'subset')
    assert not family_matches(['csg','slabs'],['csg'],'exact')
    assert len(RINDDataset(diagnostic,require_ready=False,include_families=['csg']))==0
    ds=RINDDataset(diagnostic,require_ready=False,include_families=['csg'],family_match='any',window_sizes=(128,))
    assert len(ds)==2 and np.all(ds.sampling_weights('family')==.5)
    assert len(RINDDataset(diagnostic,require_ready=False,exclude_families=['slabs']))==0
    assert len(RINDDataset(diagnostic,require_ready=False,occupancy_range=(.2,.7)))==0
    assert len(RINDDataset(diagnostic,require_ready=False,equivalent_count=64))==0
    assert len(RINDDataset(diagnostic,require_ready=False,split='test'))==0
    assert dataset_stats(diagnostic)['statuses']=={'INCOMPLETE':1}
    assert len({split_for(9,i) for i in range(100)})==3


@pytest.mark.parametrize('start_method',['spawn','fork'])
@pytest.mark.filterwarnings('ignore:This process.*:DeprecationWarning')
def test_two_workers_reopen_independent_readers(diagnostic,start_method):
    global _FORK_DATASET
    ds=RINDDataset(diagnostic,require_ready=False);ds[0]  # parent cache already open
    restored=pickle.loads(pickle.dumps(ds));assert not restored.cache
    ctx=multiprocessing.get_context(start_method)
    _FORK_DATASET=ds
    with ctx.Pool(2) as pool:
        values=(pool.map(read_in_fork,range(len(ds))) if start_method=='fork' else
                pool.starmap(read_in_worker,[(ds,i) for i in range(len(ds))]))
    assert [v[0] for v in values]==[64,64,128,128]
    assert all(v[2]!=ds._pid for v in values)
    ds.close()
    _FORK_DATASET=None


def test_corrupt_metadata_and_query_counts_are_detected(diagnostic):
    ds=RINDDataset(diagnostic,require_ready=False)
    sid,wid,size,count=ds.records[0];ds.records[0]=(sid,wid,size,count+1)
    with pytest.raises(ValueError,match='Queryable'):ds[0]
    ds.close()
    ds=RINDDataset(diagnostic,require_ready=False);ds.scenes[sid]['verification_sha256']='00'*32
    with pytest.raises(ValueError,match='verification hash'):ds[0]


@pytest.mark.parametrize('side',[110,96,127,192])
def test_loader_rejects_noncontract_sizes_even_when_other_sides_are_selected(diagnostic,tmp_path,side):
    import shutil
    import pyarrow.parquet as pq
    root=tmp_path/'invalid';shutil.copytree(diagnostic,root)
    rows=pq.read_table(root/'windows.parquet').to_pylist();rows[0]['size']=side
    parquet_write(root/'windows.parquet',rows)
    with pytest.raises(ValueError,match='window side'):
        RINDDataset(root,require_ready=False,window_sizes=(64,256))


def test_legacy_v1_padded_candidate_files_remain_readable(diagnostic,tmp_path):
    import shutil
    import pyarrow.parquet as pq
    root=tmp_path/'v1';shutil.copytree(diagnostic,root)
    path=root/'scenes/scene_000000'
    counts=np.fromfile(path/'candidate_counts.bin',dtype='<u2')
    table=np.full((len(counts),128),65535,dtype='<u2');table[:,0]=0;table.tofile(path/'candidate_ids.bin')
    (path/'candidate_offsets.bin').unlink()
    meta=json.loads((path/'verification.json').read_text());meta.update(contract_version=1,candidate_storage='fixed-128-v1')
    meta['hashes']={p.name:sha256_file(p) for p in path.iterdir() if p.is_file() and p.name!='verification.json'}
    atomic_json(path/'verification.json',meta)
    atomic_json(root/'manifest.json',{'format':'RIND3D','version':1})
    scenes=pq.read_table(root/'scenes.parquet').to_pylist();scenes[0]['verification_sha256']=sha256_file(path/'verification.json')
    parquet_write(root/'scenes.parquet',scenes)
    with RINDDataset(root,require_ready=False) as ds:
        sample=ds[0]
        assert ds.private_oracle().get_candidate_sources(sample['scene_id'],0).shape==(1,3)
        assert sample['observed_response'].shape==(64,64,64)


def test_legacy_side_32_requires_explicit_diagnostic_opt_in(diagnostic,tmp_path):
    import shutil
    import pyarrow.parquet as pq
    root=tmp_path/'legacy32';shutil.copytree(diagnostic,root);path=root/'scenes/scene_000000'
    data=np.fromfile(path/'window_records.bin',dtype='<u2').reshape(-1,4);data[2,3]=32;data.tofile(path/'window_records.bin')
    rows=pq.read_table(root/'windows.parquet').to_pylist();rows[2]['size']=32;parquet_write(root/'windows.parquet',rows)
    meta=json.loads((path/'verification.json').read_text());meta['hashes']['window_records.bin']=sha256_file(path/'window_records.bin');atomic_json(path/'verification.json',meta)
    scenes=pq.read_table(root/'scenes.parquet').to_pylist();scenes[0]['verification_sha256']=sha256_file(path/'verification.json');parquet_write(root/'scenes.parquet',scenes)
    with pytest.raises(ValueError,match='window side'):RINDDataset(root,require_ready=False)
    with pytest.raises(ValueError,match='require_ready'):RINDDataset(root,legacy_diagnostics=True)
    with RINDDataset(root,require_ready=False,legacy_diagnostics=True,window_sizes=(32,)) as ds:
        assert len(ds)==1 and ds[0]['observed_response'].shape==(32,)*3


def test_selection_provenance_is_private_only(diagnostic,tmp_path):
    import shutil
    import pyarrow.parquet as pq
    root=tmp_path/'private-selection';shutil.copytree(diagnostic,root);path=root/'scenes/scene_000000'
    parquet_write(path/'window_selection.parquet',[{'window_id':0,'provenance':'uniform_accepted','selection_score':.5,'witnesses':[[128,0,0],[129,0,0]]}])
    meta=json.loads((path/'verification.json').read_text());meta['hashes']['window_selection.parquet']=sha256_file(path/'window_selection.parquet');atomic_json(path/'verification.json',meta)
    scenes=pq.read_table(root/'scenes.parquet').to_pylist();scenes[0]['verification_sha256']=sha256_file(path/'verification.json');parquet_write(root/'scenes.parquet',scenes)
    with RINDDataset(root,require_ready=False) as ds:
        sample=ds[0]
        assert set(sample)=={'scene_id','window_id','window_origin','window_size','structure','observed_response'}
        private=ds.private_oracle().get_window_selection_metadata(sample['scene_id'],0)
        assert private['provenance']=='uniform_accepted' and 'witnesses' in private


@pytest.mark.parametrize('response_format',['packed','bool','torch.bool'])
def test_optional_torch_collation(diagnostic,response_format):
    torch=pytest.importorskip('torch')
    from rind3d.data import make_dataloader
    with RINDDataset(diagnostic,require_ready=False,response_format=response_format) as ds:
        loader=make_dataloader(ds,batch_size=2,num_workers=2)
        assert len(loader)>0 and loader.dataset is ds
        for batch in loader:
            assert batch['observed_response'].dtype==(torch.uint8 if response_format=='packed' else torch.bool)
            assert batch['observed_response'].shape[-1]==((batch['window_size']+7)//8 if response_format=='packed' else batch['window_size'])


def test_early_worker_shutdown_has_no_tensor_ipc_abort(diagnostic,tmp_path):
    pytest.importorskip('torch')
    import subprocess,sys,os
    script=tmp_path/'early.py'
    script.write_text('from rind3d import RINDDataset\nfrom rind3d.data import make_dataloader\n'
        'if __name__ == "__main__":\n'
        f' with RINDDataset({str(diagnostic)!r},require_ready=False,response_format="bool") as ds:\n'
        '  for batch in make_dataloader(ds,batch_size=2,num_workers=2):\n   break\n')
    result=subprocess.run([sys.executable,str(script)],capture_output=True,text=True,timeout=45,
                          env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    assert result.returncode==0 and 'terminate called' not in result.stderr and 'Exception ignored' not in result.stderr,result.stderr
