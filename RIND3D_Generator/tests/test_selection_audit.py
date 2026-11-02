"""Check the common audit cannot count solid zeros as free shadow or bias bits."""
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import chisquare
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_stored_selection import uniform_class_point,composition,face_margin
from rind3d.pack import RINDPack
from test_information_index import write_fields


def test_common_audit_witness_is_uniform_over_set_voxels_not_bytes():
    bits=np.array([[[1,3,24]]],dtype=np.uint8);rng=np.random.default_rng(919)
    counts=Counter(uniform_class_point(bits,(0,0,0),rng) for _ in range(10000))
    assert set(counts)=={(0,0,0),(8,0,0),(9,0,0),(19,0,0),(20,0,0)}
    assert chisquare(list(counts.values())).pvalue>.001


def test_common_audit_counts_exact_free_response_and_uses_half_open_faces(tmp_path):
    paths=write_fields(tmp_path,(64,)*3,lambda x,y,z:((x<32)&(z>=16),z<16))
    values=[{'origin':[0,0,0],'side':64}]
    with RINDPack(paths[0]) as r,RINDPack(paths[1]) as m:
        result=composition(values,r,m,42,32);measurement=result['measurements'][0]
        assert measurement['lit_free_count']==32*64*48
        assert measurement['shadow_free_count']==32*64*48
        assert measurement['solid_count']==64*64*16
        assert measurement['lit_fraction_free']==.5 and measurement['solid_fraction']==.25
        for point,expected in zip(measurement['uniform_class_audit_witnesses'],(True,False)):
            assert not m.read_point(*point) and r.read_point(*point)==expected
    assert face_margin((63,10,10),(0,0,0),64)==0
    assert face_margin((1,10,10),(0,0,0),64)==1
