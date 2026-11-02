from collections import Counter
from types import SimpleNamespace
import numpy as np
import pytest
from rind3d.counts import Counts
from rind3d.stratified_windows import select,DatasetBalance,settings
from rind3d.stored_windows import select_stored_windows

class PlaneField:
    shape=(1024,)*3;global_flags=3;nbytes=1
    def __init__(self):self.counters=Counter()
    def query(self,origin,size,**kwargs):
        self.counters['queries']+=1;x,y,z=origin;lit=max(0,min(size,x+size-512));shadow=size-lit
        return SimpleNamespace(informative=lit>0 and shadow>0,flags=(1 if lit else 0)|(2 if shadow else 0),lit=(max(512,x),y,z),shadow=(min(511,x+size-1),y,z))
    def point_class(self,p):return 1 if p[0]>=512 else 2
class CountsField:
    def query(self,origin,size):
        lit=max(0,min(size,origin[0]+size-512))*size**2
        return Counts(lit,size**3-lit,0)

def test_natural_is_unchanged_and_private_annotations():
    kw=dict(seed=7,scene=3,settings={'control_per_side':0})
    old=select_stored_windows(PlaneField(),[.01]*3,32,**kw)
    new=select(PlaneField(),CountsField(),[.01]*3,seed=7,scene=3,count=32,natural_settings=kw['settings'])
    assert [w.record() for w in old.windows]==[w.record() for w in new.windows]
    assert new.report['natural_sampler_unchanged'] and all('ratio_bin' in row for row in new.provenance)

@pytest.mark.parametrize('mode',['stratified','mixed'])
def test_exact_quotas_deterministic_and_no_duplicates(mode):
    def run():return select(PlaneField(),CountsField(),[.01]*3,seed=72,scene=1,observation={'mode':mode,'control_per_side':0,'pool_multiplier':4},count=32)
    a=run();b=run();assert a.report['complete'] and len(a.windows)==32
    assert [w.record() for w in a.windows]==[w.record() for w in b.windows]
    assert len({w.record() for w in a.windows})==32
    assert a.report['side_counts']=={64:10,128:11,256:11}
    assert all(0<r['lit_ratio']<1 for r in a.provenance)

def test_dataset_soft_targets_and_explicit_shortages():
    balance=DatasetBalance();c=settings({'mode':'mixed'})
    for _ in range(1000):
        component=balance.component(c);bin=balance.bin(component,[3,4,5]);balance.record(component,4 if bin is None else bin)
    assert dict(balance.components)=={'uniform':500,'natural':300,'extreme':200}
    assert balance.bins.sum()==1000 and balance.bins[[0,1,2,6,7,8,9]].sum()==0
    with pytest.raises(ValueError):settings({'mode':'invalid'})
