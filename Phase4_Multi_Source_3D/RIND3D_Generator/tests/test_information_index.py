"""The index is checked against independent complete packed-window scans."""
from pathlib import Path
import json
import numpy as np
import pytest
from rind3d.pack import RINDPack,RINDPackWriter
from rind3d.information import InformationIndex


def write_fields(root,shape,function):
    root=Path(root);root.mkdir(exist_ok=True);identity='ab'*32;policy='cd'*32
    writers=[RINDPackWriter(root/(name+'.rindb'),shape,identity=identity,policy=policy) for name in ('response','solid')]
    grid=tuple((n+63)//64 for n in shape);z,y,x=np.ogrid[:64,:64,:64]
    for bz in range(grid[0]):
        for by in range(grid[1]):
            for bx in range(grid[2]):
                response,solid=function(x+bx*64,y+by*64,z+bz*64)
                valid=(z+bz*64<shape[0])&(y+by*64<shape[1])&(x+bx*64<shape[2])
                arrays=[np.broadcast_to(a,(64,64,64))&valid for a in (response,solid)]
                for writer,a in zip(writers,arrays):writer.write((bz*grid[1]+by)*grid[2]+bx,np.packbits(a,bitorder='little').tobytes())
    for writer in writers:writer.finish()
    return root/'response.rindb',root/'solid.rindb'


def brute(response,solid,origin,side):
    y=response.read_window(*origin,side,packed=True).data;m=solid.read_window(*origin,side,packed=True).data
    lit=y&~m;shadow=~y&~m
    if side%8:lit[:,:,-1]&=(1<<(side%8))-1;shadow[:,:,-1]&=(1<<(side%8))-1
    return int(np.any(lit))|int(np.any(shadow))<<1


def check_queries(paths,queries):
    with RINDPack(paths[0],cache_bricks=2) as r,RINDPack(paths[1],cache_bricks=2) as m:
        index=InformationIndex.build(r,m)
        for origin,side in queries:
            expected=brute(r,m,origin,side)
            assert index.query(origin,side).flags==expected
            proof=index.query(origin,side,witnesses=True,witness_seed=13)
            assert proof.flags==expected
            for flag,p in ((1,proof.lit),(2,proof.shadow)):
                if not expected&flag:assert p is None;continue
                assert p is not None and all(o<=v<o+side for v,o in zip(p,origin))
                assert not m.read_point(*p) and r.read_point(*p)==(flag==1)
        assert len(r.cache)<=2 and len(m.cache)<=2
        return index.build_report


def test_random_unaligned_windows_all_sizes_and_solid_mask(tmp_path):
    def field(x,y,z):
        solid=(x*11+y*7+z*3)%13<4
        return ((x*17+y*5+z*11)%19<7)&~solid,solid
    paths=write_fields(tmp_path,(192,)*3,field);rng=np.random.default_rng(513);queries=[]
    for side in (1,2,7,8,9,15,16,17,31,32,63,64,65,127,128,129,191,192):
        for _ in range(12):queries.append((tuple(rng.integers(0,193-side,3)),side))
    result=check_queries(paths,queries);assert result['oracle_calls']==0


def test_boundary_voxels_half_open_and_partial_bytes(tmp_path):
    def field(x,y,z):
        lit=((x==16)&(y==16)&(z==16))|((x==63)&(y==63)&(z==63))
        solid=(x<3)&(y<3)&(z<3)
        return lit,solid
    paths=write_fields(tmp_path,(64,)*3,field)
    queries=[((0,0,0),16),((1,1,1),16),((16,16,16),1),((17,17,17),16),((48,48,48),16),
             ((49,49,49),15),((0,0,0),3),((15,16,16),1),((15,16,16),2),((63,63,63),1)]
    check_queries(paths,queries)


@pytest.mark.parametrize('kind',['lit','shadow','solid'])
def test_degenerate_scenes_and_no_solid_as_shadow(tmp_path,kind):
    paths=write_fields(tmp_path,(65,49,33),lambda x,y,z:(np.array(kind=='lit'),np.array(kind=='solid')))
    queries=[((0,0,0),1),((1,7,9),17),((16,32,48),17),((0,0,0),33)]
    check_queries(paths,queries)
    with RINDPack(paths[0]) as r,RINDPack(paths[1]) as m:
        index=InformationIndex.build(r,m);assert index.global_flags=={'lit':1,'shadow':2,'solid':0}[kind]


def test_rare_informative_region_and_aligned_leaf_boundary(tmp_path):
    paths=write_fields(tmp_path,(128,)*3,lambda x,y,z:((x==64)&(y==64)&(z==64),np.array(False)))
    queries=[((0,0,0),64),((1,1,1),64),((64,64,64),64),((64,64,64),1),((63,63,63),2)]
    check_queries(paths,queries)


def test_save_load_binding_checksums_and_aggregation(tmp_path):
    paths=write_fields(tmp_path/'fields',(128,)*3,lambda x,y,z:((x>=63)&(y<101),np.array(False)))
    with RINDPack(paths[0]) as r,RINDPack(paths[1]) as m:
        index=InformationIndex.build(r,m);p=tmp_path/'information.rindi';size=index.save(p)
        assert size<2048 and index.nbytes==585
        loaded=InformationIndex.load(p,r,m)
        assert loaded.query((7,19,31),64,witnesses=True)==index.query((7,19,31),64,witnesses=True)
        assert loaded.nbytes==index.nbytes
        data=bytearray(p.read_bytes());data[-1]^=1;p.write_bytes(data)
        with pytest.raises(ValueError,match='checksum'):InformationIndex.load(p,r,m)


@pytest.mark.parametrize('origin,side',[((-1,0,0),64),((1,0,0),64),((0,0,0),0),((0.,0,0),16),((0,0,0),True)])
def test_bad_rois_rejected(tmp_path,origin,side):
    paths=write_fields(tmp_path,(64,)*3,lambda x,y,z:(x>=32,np.array(False)))
    with RINDPack(paths[0]) as r,RINDPack(paths[1]) as m:
        index=InformationIndex.build(r,m)
        with pytest.raises((ValueError,TypeError)):index.query(origin,side)


def test_invalid_lit_solid_truth_rejected(tmp_path):
    paths=write_fields(tmp_path,(64,)*3,lambda x,y,z:(np.array(True),np.array(True)))
    with RINDPack(paths[0]) as r,RINDPack(paths[1]) as m:
        with pytest.raises(ValueError,match='solid'):InformationIndex.build(r,m)
