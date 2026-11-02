"""Optional Zarr v3 sharded Z-axis bitpacking baseline and lossless export."""
from pathlib import Path
import numpy as np
from .pack import PackedWindow,RINDPack,sha256_file,atomic_json
import os,json

def create_zarr(path,shape):
    import zarr
    from zarr.codecs import ZstdCodec,Crc32cCodec
    nz,ny,nx=shape
    return zarr.create_array(str(path),shape=(nx,ny,(nz+7)//8),chunks=(64,64,8),
                             shards=(256,256,32),dtype='u1',fill_value=0,zarr_format=3,
                             compressors=[ZstdCodec(level=3),Crc32cCodec()],
                             attributes={'format':'RIND3D-Zarr-packed-v1','logical_shape_zyx':list(shape),
                                         'byte_array_axes':'X Y packed-Z','bit_order':'little','complete':False})

def write_zarr_brick(array,brick_xyz,values):
    x,y,z=(v*64 for v in brick_xyz);nz,ny,nx=values.shape
    array[x:x+nx,y:y+ny,z//8:(z+nz+7)//8]=np.packbits(values.transpose(2,1,0),axis=-1,bitorder='little')

def seal_zarr(path,array,source_sha256=None):
    path=Path(path);array.attrs['complete']=True
    atomic_json(path/'rind-integrity.json',{'source_sha256':source_sha256,
        'files':{str(p.relative_to(path)):sha256_file(p) for p in path.rglob('*') if p.is_file() and p.name!='rind-integrity.json'}})

class ZarrBits:
    def __init__(self,path):
        import zarr
        self.path=Path(path);self.array=zarr.open_array(str(path),mode='r')
        attrs=dict(self.array.attrs)
        if attrs.get('format')!='RIND3D-Zarr-packed-v1' or attrs.get('bit_order')!='little' or attrs.get('byte_array_axes')!='X Y packed-Z':raise ValueError('Unsupported packed Zarr')
        if not attrs.get('complete'):raise ValueError('Incomplete packed Zarr export')
        self.shape=tuple(attrs['logical_shape_zyx'])
        if self.array.dtype!=np.dtype('u1') or self.array.chunks!=(64,64,8) or self.array.shards!=(256,256,32):raise ValueError('Unexpected Zarr dtype/chunks/shards')
        nz,ny,nx=self.shape
        if self.array.shape!=(nx,ny,(nz+7)//8):raise ValueError('Zarr byte/logical shape mismatch')
        integrity=self.path/'rind-integrity.json'
        files=json.loads(integrity.read_text())['files']
        actual={str(p.relative_to(self.path)) for p in self.path.rglob('*') if p.is_file() and p.name!='rind-integrity.json'}
        if set(files)!=actual:raise ValueError('Missing/unexpected Zarr shard or metadata')
        for name,digest in files.items():
            if name not in actual or sha256_file(self.path/name)!=digest:raise ValueError('Zarr export integrity mismatch')
    def read_window(self,x,y,z,size,*,packed=False):
        if any(not isinstance(v,(int,np.integer)) for v in (x,y,z,size)):raise TypeError('Window coordinates must be integers')
        if size<1 or min(x,y,z)<0 or any(o+size>n for o,n in zip((z,y,x),self.shape)):raise ValueError('Window out of bounds')
        raw=np.asarray(self.array[x:x+size,y:y+size,z//8:(z+size+7)//8])
        values=np.unpackbits(raw,axis=-1,bitorder='little')[...,z%8:z%8+size].transpose(2,1,0).astype(bool)
        return PackedWindow(np.packbits(values,axis=-1,bitorder='little'),values.shape) if packed else values
    def close(self):pass

def export_zarr(response_path,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    temporary=output.with_name(output.name+'.partial')
    if temporary.exists():raise FileExistsError(temporary)
    with RINDPack(response_path,verify_file=True) as reader:
        array=create_zarr(temporary,reader.shape)
        nz,ny,nx=reader.shape
        # At most one 2 MiB packed outer shard, avoiding 64 shard rewrites.
        for z in range(0,nz,256):
            for y in range(0,ny,256):
                for x in range(0,nx,256):
                    sz,sy,sx=min(256,nz-z),min(256,ny-y),min(256,nx-x)
                    buffer=np.zeros((sx,sy,(sz+7)//8),dtype=np.uint8)
                    for dz in range(0,sz,64):
                        for dy in range(0,sy,64):
                            for dx in range(0,sx,64):
                                a=reader.read_global_brick(((x+dx)//64,(y+dy)//64,(z+dz)//64))
                                a=a[:min(64,sz-dz),:min(64,sy-dy),:min(64,sx-dx)]
                                bz,by,bx=a.shape
                                buffer[dx:dx+bx,dy:dy+by,dz//8:(dz+bz+7)//8]=np.packbits(a.transpose(2,1,0),axis=-1,bitorder='little')
                    array[x:x+sx,y:y+sy,z//8:(z+sz+7)//8]=buffer
        seal_zarr(temporary,array,sha256_file(response_path))
    os.replace(temporary,output)
    return str(output)
