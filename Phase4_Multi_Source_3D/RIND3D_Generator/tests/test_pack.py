import hashlib,struct,zlib
from pathlib import Path
import numpy as np
import pytest
from rind3d.pack import RINDPack,RINDPackWriter,HEADER,RAW,RECORD
from rind3d._native import encode_brick

ID=hashlib.sha256(b'scene').hexdigest();POLICY=hashlib.sha256(b'policy').hexdigest()

def write(path,field):
    w=RINDPackWriter(path,field.shape,identity=ID,policy=POLICY)
    nz,ny,nx=(n//64 for n in field.shape)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                b=field[z*64:z*64+64,y*64:y*64+64,x*64:x*64+64]
                w.write((z*ny+y)*nx+x,np.packbits(b,bitorder='little').tobytes())
    w.finish();return path

@pytest.mark.parametrize('kind',['zero','one','checker','slab','random'])
def test_native_writer_independent_reader(tmp_path,kind):
    z,y,x=np.indices((128,)*3)
    f={'zero':x<0,'one':x>=0,'checker':(x+y+z)%2==0,'slab':(x>=37)&(x<88),'random':np.random.default_rng(1).integers(0,2,(128,)*3,dtype=np.uint8).astype(bool)}[kind]
    p=write(tmp_path/'response.rindb',f)
    assert p.stat().st_size<=HEADER+8*RECORD.size+8*RAW
    with RINDPack(p,verify_file=True,cache_bricks=2) as r:
        for origin,size in [((0,0,0),64),((17,27,31),67),((61,63,59),65),((97,99,101),7)]:
            x0,y0,z0=origin;expected=f[z0:z0+size,y0:y0+size,x0:x0+size]
            assert np.array_equal(r.read_window(*origin,size),expected)
            packed=r.read_window(*origin,size,packed=True)
            assert np.array_equal(packed.unpack(),expected)
            assert packed.nbytes==size*size*((size+7)//8)
        assert r.verify_all()==8

def test_bit_axis_and_random_access(tmp_path):
    a=np.zeros((128,)*3,dtype=bool);a[1,2,3]=True;a[71,84,99]=True;a[63,61,59]=True
    p=write(tmp_path/'r',a)
    with RINDPack(p,cache_bricks=1) as r:
        raw=r.read_brick_packed((0,0,0));i=(1*64+2)*64+3
        assert raw[i//8]&(1<<(i%8))
        assert r.decoded_bricks==1
        assert r.read_window(99,84,71,1).item()
        assert r.decoded_bricks==2
        assert len(r.cache)==1
        assert r.read_point(np.uint16(59),np.uint16(61),np.uint16(63))
        assert not r.read_point(np.uint16(59),np.uint16(61),np.uint16(31))
        with pytest.raises(TypeError):r.read_point(.5,0,0)
        with pytest.raises(ValueError):r.read_window(-1,0,0,1)
        with pytest.raises(TypeError):r.read_window(.5,0,0,1)

def test_shortest_actual_codec():
    assert encode_brick(bytes(RAW))[0]==0
    assert encode_brick(bytes([255])*RAW)[0]==1
    random=np.random.default_rng(7).integers(0,256,RAW,dtype=np.uint8).tobytes()
    codec,payload,crc,ones=encode_brick(random)
    assert codec==2 and payload==random and crc==zlib.crc32(random)
    assert len(payload)<=RAW
    assert encode_brick(bytes([0,255])*(RAW//2))[0]==3

def test_restart_and_identity(tmp_path):
    p=tmp_path/'r';w=RINDPackWriter(p,(64,64,128),identity=ID,policy=POLICY)
    w.write(1,bytes([255])*RAW);w.checkpoint()
    with pytest.raises(RuntimeError):w.finish()
    del w
    with pytest.raises((ValueError,RuntimeError),match='mismatch'):
        RINDPackWriter(p,(64,64,128),identity='ab'*32,policy=POLICY,resume=True)
    w=RINDPackWriter(p,(64,64,128),identity=ID,policy=POLICY,resume=True)
    assert w.committed=={1}
    w.write(0,bytes(RAW));w.finish()
    with RINDPack(p) as r:
        assert not r.read_window(0,0,0,64).any()
        assert r.read_window(64,0,0,64).all()

@pytest.mark.parametrize('mode',['header','axis','endian','truncated','payload','index'])
def test_corruption(tmp_path,mode):
    f=np.random.default_rng(2).integers(0,2,(64,)*3,dtype=np.uint8)
    p=write(tmp_path/'r',f);raw=bytearray(p.read_bytes())
    if mode=='header':raw[18]^=1
    elif mode=='axis':raw[34]=1;struct.pack_into('<I',raw,252,zlib.crc32(raw[:252]))
    elif mode=='endian':raw[8:10]=b'\x00\x01';struct.pack_into('<I',raw,252,zlib.crc32(raw[:252]))
    elif mode=='truncated':raw=raw[:-1]
    elif mode=='payload':raw[-1]^=1
    elif mode=='index':raw[HEADER+12]=9
    p.write_bytes(raw)
    with pytest.raises(ValueError):
        with RINDPack(p) as r:r.verify_all()
