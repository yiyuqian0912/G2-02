import numpy as np
import pytest
from rind3d.pack import RINDPackWriter,RINDPack,RAW
from rind3d.zarr_store import export_zarr,ZarrBits,create_zarr

def test_zarr_export_and_unaligned_bit_axis(tmp_path):
    p=tmp_path/'response.rindb';w=RINDPackWriter(p,(128,)*3,identity='ab'*32,policy='cd'*32)
    for i in range(8):
        rng=np.random.default_rng(i);a=rng.integers(0,2,(64,)*3,dtype=np.uint8)
        w.write(i,np.packbits(a,bitorder='little').tobytes())
    w.finish();export_zarr(p,tmp_path/'zarr')
    r=RINDPack(p);z=ZarrBits(tmp_path/'zarr')
    for origin,side in [((3,5,7),65),((13,17,19),96),((65,66,67),33)]:
        assert np.array_equal(r.read_window(*origin,side),z.read_window(*origin,side))
        assert np.array_equal(r.read_window(*origin,side,packed=True).unpack(),z.read_window(*origin,side,packed=True).unpack())
    r.close()
    store=tmp_path/'zarr';shard=next(p for p in store.rglob('*') if p.is_file() and p.name not in ('zarr.json','rind-integrity.json'))
    shard.unlink()
    with pytest.raises((ValueError,FileNotFoundError)):ZarrBits(store)

def test_incomplete_zarr_export_is_rejected(tmp_path):
    create_zarr(tmp_path/'z',(64,)*3)
    with pytest.raises(ValueError,match='Incomplete'):ZarrBits(tmp_path/'z')
