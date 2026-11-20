import numpy as np
import pytest
from scipy.ndimage import label

from selection import select, wants_alternative
from population import features,spatial_features


def row(w,frac=.1,j=0,side=64):
    return dict(scene="s",slot=3,L=side,w=w,j=j,stream="natural",fraction=frac,
                edges=12,orientation=0.,bins=3,geo=None)


def test_fixed_source_controls_and_ties():
    pair=[[row(0)],[row(1,.4)]]
    assert select(pair,"balance")==1 and select(pair,"natural")==0
    assert select([[row(0)],[row(1,.4,1)]],"balance")==0
    assert select([[row(0)],[row(1)]],"balance")==0
    assert select(pair,"geometry")==0
    with pytest.raises(ValueError):
        select([[row(0)],[row(1,side=128)]],"natural")
    pair[1][0]["stream"]="coverage"
    with pytest.raises(ValueError):
        select(pair,"balance")


def test_entire_multichannel_window_and_prior_holdback():
    a=[row(0,.2,0),row(0,.2,1)]
    b=[row(1,.01,0),row(1,.49,1)]
    assert select([a,b],"balance")==0
    flags=[wants_alternative(a,"mixture",i) for i in range(25)]
    assert len(set(flags))==2
    assert flags==[wants_alternative(a,"mixture",i) for i in range(25)]


def test_spatial_bins_require_both_free_endpoints():
    cls=np.full((64,64,64),2,np.uint8)
    cls[15,15,15]=cls[48,48,48]=0
    cls[16,15,15]=cls[49,48,48]=1
    assert spatial_features(cls)==(2,2)
    cls[16,15,15]=2
    assert spatial_features(cls)==(1,1)
    assert features(6,2,[1,0,0])["fraction"]==.25
