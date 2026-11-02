import hashlib
from pathlib import Path
import numpy as np
import pytest
from rind3d.pack import RINDPack,RINDPackWriter
from rind3d.counts import CountIndex,Counts

def write(path,array):
    writer=RINDPackWriter(path,array.shape,identity=hashlib.sha256(b'count-test').hexdigest(),policy=hashlib.sha256(b'policy').hexdigest())
    n=array.shape[0];g=(n+63)//64
    for z in range(g):
        for y in range(g):
            for x in range(g):
                value=array[z*64:(z+1)*64,y*64:(y+1)*64,x*64:(x+1)*64];pad=np.zeros((64,)*3,dtype=bool);pad[:value.shape[0],:value.shape[1],:value.shape[2]]=value
                writer.write((z*g+y)*g+x,np.packbits(pad,bitorder='little').tobytes())
    writer.finish()

@pytest.mark.parametrize('kind',['random','uniform_lit','uniform_shadow','solid','rare'])
def test_exact_counts_against_independent_dense_roi(tmp_path,kind):
    n=128;rng=np.random.default_rng(116)
    mask=rng.random((n,)*3)<.25;response=(rng.random((n,)*3)<.4)&~mask
    if kind.startswith('uniform'):mask[:]=False;response[:]=kind=='uniform_lit'
    if kind=='solid':mask[:]=True;response[:]=False
    if kind=='rare':
        mask[:]=True;response[:]=False
        for p,label in [((0,0,0),1),((127,127,127),0),((63,64,65),1),((64,63,64),0)]:mask[p]=False;response[p]=label
    write(tmp_path/'y',response);write(tmp_path/'m',mask)
    with RINDPack(tmp_path/'y') as y,RINDPack(tmp_path/'m') as m:
        index=CountIndex.build(y,m,cache_bricks=2)
        cases=[((0,0,0),128),((0,0,0),1),((127,127,127),1),((15,17,31),64),((16,16,16),64),((63,63,63),65),((7,9,11),101)]
        for _ in range(100):
            size=int(rng.integers(1,129));cases.append((tuple(rng.integers(0,n-size+1,3)),size))
        for origin,side in cases:
            x,zv,z=map(int,origin);sl=(slice(z,z+side),slice(zv,zv+side),slice(x,x+side));a=response[sl];b=mask[sl]
            expected=Counts(int((a&~b).sum()),int((~a&~b).sum()),int(b.sum()))
            assert index.query(origin,side)==index.direct(origin,side)==expected
            if expected.informative:assert expected.ratio_bin==min(9,int(expected.lit_ratio*10))
        path=tmp_path/'count.rindc';size=index.save(path);loaded=CountIndex.load(path,y,m)
        assert size>0 and loaded.query((15,17,31),64)==index.query((15,17,31),64)
        assert index.nbytes<100_000
        with pytest.raises(ValueError):index.query((127,127,127),2)
        with pytest.raises(TypeError):index.query((.1,0,0),16)
        with pytest.raises(ValueError):Counts(1,0,0).ratio_bin

def test_count_index_corruption_and_source_identity(tmp_path):
    a=np.zeros((64,)*3,dtype=bool);write(tmp_path/'y',a);write(tmp_path/'m',a)
    with RINDPack(tmp_path/'y') as y,RINDPack(tmp_path/'m') as m:
        index=CountIndex.build(y,m);path=tmp_path/'count';index.save(path)
        with path.open('ab') as f:f.write(b'corrupt')
        with pytest.raises(ValueError,match='SHA'):CountIndex.load(path,y,m)
    b=np.ones((64,)*3,dtype=bool);write(tmp_path/'bad-y',b);write(tmp_path/'bad-m',b)
    with RINDPack(tmp_path/'bad-y') as y,RINDPack(tmp_path/'bad-m') as m:
        with pytest.raises(RuntimeError,match='Lit solid'):CountIndex.build(y,m)
