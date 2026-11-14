import numpy as np
import pytest
from rind3d.geometry import Oracle,generate_geometry,Budget,Aborted
from rind3d.search import CandidateBank,separated,farthest,verify_grid_exact
from rind3d.legacy.windows import Window as LegacyWindow
from rind3d.windows import Window,informative,excludes

def test_separation_and_bank_indices():
    bank=CandidateBank();a=[.1,.2,.3]
    assert bank.add(a)==bank.add(a)==0
    assert bank.index_dtype.str=='<u2'
    assert separated([0,0,0],[.03125,0,0],.03125)
    assert not separated([0,0,0],[np.nextafter(.03125,0),0,0],.03125)
    points=np.array([[.1,.1,.1],[.101,.1,.1],[.8,.8,.8],[.1,.8,.1]])
    ids=farthest(points,128,.03125)
    assert len(ids)==3 and ids[0]==0
    for i in ids:
        for j in ids:
            if i!=j:assert separated(points[i],points[j],.03125)

def test_full_window_rejects_a_probe_only_match():
    oracle=Oracle(generate_geometry(7,0,['csg']));truth=np.array([.03,.07,.11])
    w=LegacyWindow((16,16,16),32)
    observed=np.zeros((16,)*3,dtype=bool)
    # Deliberately corrupt one unprobed voxel, even for the identical source.
    from rind3d.geometry import render_region
    observed=render_region(oracle,truth,w.origin,w.size,64)[0]
    observed[9,10,11]=~observed[9,10,11]
    result=verify_grid_exact(oracle,truth,truth,w,64,Budget(30),observed=observed)
    assert result['status']=='mismatch'

def test_empty_validity_and_source_boundary():
    w=Window((0,0,0),64)
    assert not excludes(w,[.01,.01,.01],1024)
    assert excludes(w,[64/1024,.01,.01],1024)
    y=np.ones((2,2,2),dtype=bool);m=np.zeros_like(y)
    assert not informative(y,m)[0]
    m[0,0,0]=True;y[0,0,0]=False
    assert not informative(y,m)[0]
    y[0,0,1]=False;assert informative(y,m)[0]
