import numpy as np
import pytest
from tests.test_exact_counts import write
from rind3d.pack import RINDPack
from rind3d.windows import Window
from rind3d.observation_research import transition_observations

@pytest.mark.parametrize('kind',['x','y','z','solid_barrier','rare_last_voxel'])
def test_transition_points_are_actual_free_voxels_and_deterministic(tmp_path,kind):
    z,y,x=np.indices((64,)*3);mask=np.zeros((64,)*3,dtype=bool)
    response=({'x':x,'y':y,'z':z}.get(kind,x)>=31)
    if kind=='solid_barrier':mask=(x>=30)&(x<=33);response=(x>33)&~mask
    if kind=='rare_last_voxel':response[:]=False;response[-1,-1,-1]=True
    write(tmp_path/'y',response);write(tmp_path/'m',mask)
    with RINDPack(tmp_path/'y') as a,RINDPack(tmp_path/'m') as b:
        p,labels,r=transition_observations(a,b,Window((0,0,0),64),seed=9);q,l,s=transition_observations(a,b,Window((0,0,0),64),seed=9)
    assert np.array_equal(p,q) and np.array_equal(labels,l) and r==s
    ids=np.rint(p*64-.5).astype(int)[:,::-1]
    assert np.array_equal(response[tuple(ids.T)],labels) and not np.any(mask[tuple(ids.T)])
    assert labels.any() and (~labels).any() and len({tuple(v) for v in ids})==len(ids)
    if kind=='solid_barrier':assert r['fallback_used'] and r['pairs_selected']==0
