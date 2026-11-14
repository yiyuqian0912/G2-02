import numpy as np
import pytest
from rind3d.windows import sample_windows,excludes,informative,sobol_points
from rind3d.pack import RINDPackWriter,RINDPack
from rind3d.cli import load_config
from rind3d.window_contract import WINDOW_SIDES,PILOT_SIDE_COUNTS


def test_stratified_stored_response_windows_are_deterministic_and_informative(tmp_path):
    # Deliberately synthetic one-plane bitfield for the sampler, not published truth.
    p=tmp_path/'response';m=tmp_path/'solid'
    for path,solid in ((p,False),(m,True)):
        writer=RINDPackWriter(path,(512,)*3,identity='ab'*32,policy='cd'*32)
        z,y,x=np.indices((64,)*3)
        for bz in range(8):
            for by in range(8):
                for bx in range(8):
                    values=np.zeros((64,)*3,bool) if solid else (x+bx*64>=257)
                    writer.write((bz*8+by)*8+bx,np.packbits(values,bitorder='little').tobytes())
        writer.finish()
    cfg=load_config('configs/pilot.yaml')
    cfg.update(resolution=512,seed=71,max_window_attempts=4096,sampling_seconds=30,sampling_points=1_000_000)
    source=np.array([.05,.07,.09])
    with RINDPack(p) as response,RINDPack(m) as solid:
        windows,report=sample_windows(source,cfg,0,32,response=response,solid=solid)
        again,report2=sample_windows(source,cfg,0,32,response=response,solid=solid)
        assert windows==again and report==report2 and len(windows)==32
        assert report['side_counts']==PILOT_SIDE_COUNTS
        assert {w.size for w in windows}==set(WINDOW_SIDES)
        assert report['attempts']==32+sum(report['rejections'].values())
        assert len({w.record() for w in windows})==32 and report['occupied_octants']>=2
        for w in windows:
            assert excludes(w,source,512)
            assert informative(response.read_window(*w.origin,w.size),solid.read_window(*w.origin,w.size))[0]
            for position,value in zip(w.witnesses,(True,False)):
                assert response.read_point(*position)==value and not solid.read_point(*position)
        replacements,replacement_report=sample_windows(source,cfg,0,32,response=response,solid=solid,proposal_stream=603,
            exclude_records={w.record() for w in windows})
        assert len(replacements)==32 and replacement_report['side_counts']==PILOT_SIDE_COUNTS
        assert {w.record() for w in replacements}.isdisjoint(w.record() for w in windows)
    assert np.array_equal(sobol_points(1,2,3,16),sobol_points(1,2,3,16))
    assert not np.array_equal(sobol_points(1,2,3,16),sobol_points(1,2,4,16))
