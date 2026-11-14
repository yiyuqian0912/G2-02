import json
from pathlib import Path
import numpy as np
import pytest
from rind.contracts import Geometry
from rind.storage import CanonicalBundle
from rind3d.geometry import Oracle,Aborted
from rind3d.stream import stream_response
from rind3d.pack import RINDPack,RINDPackWriter,HEADER,RECORD,RAW

def config(n=128):return {'resolution':n,'generation_seconds':30,'generation_points':10_000_000,'workers':2,'queue_size':4,'leaf_side':8}

def test_bounded_stream_full_bits_and_resume(tmp_path):
    b=CanonicalBundle(3);b.add_geometry(Geometry.empty(3));o=Oracle(b);s=np.array([.13,.23,.37]);cfg=config()
    result=stream_response(o,s,tmp_path,cfg)
    assert result['complete'] and result['certificate_covered_voxels']==128**3
    assert result['physical_point_calls']==0
    with RINDPack(tmp_path/'response.rindb') as r,RINDPack(tmp_path/'solid.rindb') as m:
        assert r.read_window(17,19,21,67).all()
        assert not m.read_window(17,19,21,67).any()
    first=(tmp_path/'response.rindb').read_bytes()
    (tmp_path/'response.rindb.sha256').unlink()  # crash after rename, before SHA
    result=stream_response(o,s,tmp_path,cfg,resume=True)
    assert result['bricks_computed_this_run']==0
    assert first==(tmp_path/'response.rindb').read_bytes()
    assert (tmp_path/'response.rindb.sha256').exists()
    assert not (tmp_path/'response.rindb.journal.json').exists()
    with pytest.raises(ValueError,match='identity'):
        stream_response(o,s+0.01,tmp_path,cfg,resume=True)

def test_resume_compacts_orphan_payload_to_raw_bound(tmp_path):
    p=tmp_path/'r';ident='cd'*32;policy='ab'*32
    raw=np.random.default_rng(1).integers(0,256,RAW,dtype=np.uint8).tobytes()
    writer=RINDPackWriter(p,(64,64,128),identity=ident,policy=policy)
    writer.write(1,raw);writer.checkpoint();del writer
    with (tmp_path/'r.partial').open('ab') as f:f.write(b'orphan-after-crash'*500)
    writer=RINDPackWriter(p,(64,64,128),identity=ident,policy=policy,resume=True)
    writer.write(0,raw);writer.finish()
    assert p.stat().st_size<=HEADER+2*RECORD.size+2*RAW
    with RINDPack(p) as r:assert r.verify_all()==2


def test_invalid_non_dyadic_resolution_is_explicit(tmp_path):
    b=CanonicalBundle(3);b.add_geometry(Geometry.empty(3))
    with pytest.raises(ValueError,match='power of two'):
        stream_response(Oracle(b),np.array([.1,.2,.3]),tmp_path,config(96))
