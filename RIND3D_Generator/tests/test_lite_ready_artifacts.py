"""Opt-in tests against a genuinely published production pilot, never a mock READY."""
import json,os,shutil
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pytest
from rind3d.data import RINDDataset,make_dataloader
from rind3d.pipeline import verify_dataset,parquet_write
from rind3d.pack import atomic_json

@pytest.fixture(scope='module')
def pilot():
    value=os.getenv('RIND_LITE_TEST_ROOT')
    if not value:pytest.skip('Set RIND_LITE_TEST_ROOT to a completed real Lite pilot')
    root=Path(value);report=json.loads((root/'report.json').read_text())
    assert report['complete'] and report['ready_scenes']>=10
    return root

def test_real_ready_dataset_integrity(pilot):
    rows=verify_dataset(pilot,full_index=True)
    assert len(rows)>=10 and all(r['status']=='READY_LITE' and r['window_count']==256 for r in rows)

def test_real_default_private_loader_and_arbitrary_rerender(pilot):
    with RINDDataset(pilot,split=None) as ds:
        example=ds[0];private=ds.private_oracle();sid=example['scene_id'];truth=private.get_true_source(sid)
        assert 'candidate_sources' not in example and 'selection_score' not in example
        assert any(np.array_equal(p,truth) for p in private.get_candidate_sources(sid,example['window_id']))
        origin=np.minimum(example['window_origin'],1024-79);expected=private.read_global_window(sid,origin,79)
        assert np.array_equal(private.rerender(sid,truth,origin,79),expected)
        assert private.get_window_selection_metadata(sid,example['window_id'])['provenance'] in ('uniform_accepted','guided_informative')
        batch=next(iter(make_dataloader(ds,batch_size=1,num_workers=2)))
        assert 'candidate_sources' not in batch
    with RINDDataset(pilot,split=None,view='candidate_conditioned') as ds:
        batch=next(iter(make_dataloader(ds,batch_size=1,num_workers=2)))
        assert 1<=len(batch['candidate_sources'][0])<=8

def test_profile_and_geometry_group_split_corruption(pilot,tmp_path):
    for name in ('manifest.json','scenes.parquet','splits.parquet','windows.parquet'):shutil.copy2(pilot/name,tmp_path/name)
    manifest=json.loads((tmp_path/'manifest.json').read_text());manifest['max_candidates']=64;atomic_json(tmp_path/'manifest.json',manifest)
    with pytest.raises(ValueError,match='manifest'):RINDDataset(tmp_path,split=None)
    shutil.copy2(pilot/'manifest.json',tmp_path/'manifest.json')
    rows=pq.read_table(tmp_path/'scenes.parquet').to_pylist();rows[1]['geometry_id']=rows[0]['geometry_id'];rows[1]['split']='test' if rows[0]['split']!='test' else 'train'
    parquet_write(tmp_path/'scenes.parquet',rows)
    with pytest.raises(ValueError,match='leaks'):RINDDataset(tmp_path,split=None)

def test_false_ready_and_evidence_corruption_fail_closed(pilot,tmp_path):
    for name in ('manifest.json','scenes.parquet','splits.parquet','windows.parquet'):shutil.copy2(pilot/name,tmp_path/name)
    row=pq.read_table(tmp_path/'scenes.parquet').to_pylist()[0];sid=row['scene_id'];scene=tmp_path/'scenes'/sid;scene.parent.mkdir()
    shutil.copytree(pilot/'scenes'/sid,scene)
    meta=json.loads((scene/'verification.json').read_text());meta['status']='READY';atomic_json(scene/'verification.json',meta)
    with RINDDataset(tmp_path,split=None) as ds:
        index=next(i for i,r in enumerate(ds.records) if r[0]==sid)
        with pytest.raises(ValueError,match='READY_LITE'):ds[index]
    shutil.copy2(pilot/'scenes'/sid/'verification.json',scene/'verification.json')
    with (scene/'candidate_evidence.bin').open('r+b') as file:file.write(bytes(4))
    with RINDDataset(tmp_path,split=None) as ds:
        with pytest.raises(ValueError,match='hash'):ds[index]

def test_archived_loader_does_not_require_producer_binary(pilot,monkeypatch):
    from rind3d.rendering import context
    original=context.physics_provenance
    def other_build(policy):return dict(original(policy),canonical_backend='another-compiler-build-for-portability-test')
    monkeypatch.setattr(context,'physics_provenance',other_build)
    with RINDDataset(pilot,split=None) as ds:
        assert ds[0]['window_size'] in (64,128,256)
