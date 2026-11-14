"""Independent Python RINDPack v1 reader. Never imports the native decoder."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
import zlib

import numpy as np
import zstandard as zstd

HEADER, RECORD, RAW = 256, struct.Struct('<QIBBHIIII'), 32768
MAGIC = b'RINDB001'


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def atomic_text(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        f.write(value);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


@dataclass(frozen=True)
class PackedWindow:
    """ZYX bool shape; storage ZY ceil(X/8), little bits with zero row padding."""
    data: np.ndarray
    shape: tuple[int,int,int]

    def unpack(self):
        return np.unpackbits(self.data,axis=-1,bitorder='little',count=self.shape[-1]).astype(bool)

    @property
    def nbytes(self):return self.data.nbytes


class RINDPack:
    def __init__(self,path,*,cache_bricks=16,verify_file=False):
        self.path=Path(path);self.file=self.path.open('rb');self.cache_bricks=int(cache_bricks)
        if self.cache_bricks<0:raise ValueError('cache_bricks must be nonnegative')
        self.cache=OrderedDict();self.decoded_bricks=0;self.bytes_read=0
        h=self.file.read(HEADER)
        if len(h)!=HEADER or h[:8]!=MAGIC:raise ValueError('Invalid/truncated RINDPack header')
        if zlib.crc32(h[:252])!=struct.unpack_from('<I',h,252)[0]:raise ValueError('Header checksum mismatch')
        if struct.unpack_from('<HHI',h,8)!=(1,256,1):raise ValueError('Unsupported version or incomplete file')
        self.shape=struct.unpack_from('<3I',h,16)
        if any(n<1 or n>1024 for n in self.shape):raise ValueError('Invalid logical shape')
        if struct.unpack_from('<3HBBHH',h,28)!=(64,64,64,0,0,32,1):raise ValueError('Unsupported brick, axis, bit, or codec policy')
        self.grid=tuple((n+63)//64 for n in self.shape)
        n,index,data,size=struct.unpack_from('<IQQQ',h,40)
        if n!=int(np.prod(self.grid)) or index!=HEADER or data!=HEADER+n*RECORD.size or size!=self.path.stat().st_size or size>data+n*RAW:raise ValueError('Inconsistent index/file size or raw bound exceeded')
        if any(h[132:252]):raise ValueError('Nonzero reserved header bytes')
        self.identity=h[68:100].hex();self.policy=h[100:132].hex()
        raw=self.file.read(n*RECORD.size)
        if len(raw)!=n*RECORD.size:raise ValueError('Truncated index')
        self.index=list(RECORD.iter_unpack(raw));ranges=[]
        for offset,length,codec,flags,reserved,decoded_crc,stored_crc,ones,tail in self.index:
            if flags!=1 or reserved or tail or codec not in range(4) or ones>64**3:raise ValueError('Invalid index record')
            if codec in (0,1):
                if offset or length or stored_crc or ones!=(64**3 if codec else 0):raise ValueError('Invalid uniform brick')
            else:
                if not 0<length<=RAW or offset<data or offset+length>size or (codec==2 and length!=RAW):raise ValueError('Invalid payload range')
                ranges.append((offset,offset+length))
        ranges.sort()
        if any(b[0]<a[1] for a,b in zip(ranges,ranges[1:])):raise ValueError('Overlapping chunk payloads')
        if verify_file:
            expected=self.path.with_name(self.path.name+'.sha256').read_text().split()[0]
            if sha256_file(self.path)!=expected:raise ValueError('File SHA256 mismatch')

    def close(self):
        self.file.close();self.cache.clear()

    def __del__(self):
        # Also closes descriptors after a constructor rejects a corrupt header.
        file=getattr(self,'file',None)
        if file is not None:file.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

    def read_brick_packed(self,brick_xyz):
        bx,by,bz=map(int,brick_xyz);gz,gy,gx=self.grid
        if not(0<=bx<gx and 0<=by<gy and 0<=bz<gz):raise IndexError('Brick out of bounds')
        idx=(bz*gy+by)*gx+bx
        if idx in self.cache:
            raw=self.cache.pop(idx);self.cache[idx]=raw;return raw
        offset,length,codec,_,_,decoded_crc,stored_crc,ones,_=self.index[idx]
        self.file.seek(offset);stored=self.file.read(length) if length else b'';self.bytes_read+=length
        if len(stored)!=length or zlib.crc32(stored)!=stored_crc:raise ValueError(f'Brick {idx}: payload checksum mismatch')
        if codec<2:raw=bytes([255 if codec else 0])*RAW
        elif codec==2:raw=stored
        else:
            try:raw=zstd.ZstdDecompressor().decompress(stored,max_output_size=RAW)
            except zstd.ZstdError as e:raise ValueError(f'Brick {idx}: invalid Zstd') from e
        if len(raw)!=RAW or zlib.crc32(raw)!=decoded_crc:raise ValueError(f'Brick {idx}: decoded checksum mismatch')
        if int(np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).sum())!=ones:raise ValueError('Brick population count mismatch')
        self.decoded_bricks+=1
        if self.cache_bricks:
            self.cache[idx]=raw
            while len(self.cache)>self.cache_bricks:self.cache.popitem(last=False)
        return raw

    def read_global_brick(self,brick_xyz,*,packed=False):
        raw=self.read_brick_packed(brick_xyz)
        if packed:return PackedWindow(np.frombuffer(raw,dtype=np.uint8).reshape(64,64,8).copy(),(64,64,64))
        return np.unpackbits(np.frombuffer(raw,dtype=np.uint8),bitorder='little').reshape(64,64,64).astype(bool)

    def read_point(self,x,y,z):
        if any(not isinstance(v,(int,np.integer)) for v in (x,y,z)):raise TypeError('Point coordinates must be integers')
        x,y,z=map(int,(x,y,z))
        if min(x,y,z)<0 or any(v>=n for v,n in zip((z,y,x),self.shape)):raise ValueError('Point out of bounds')
        raw=self.read_brick_packed((x//64,y//64,z//64));i=((z%64)*64+y%64)*64+x%64
        return bool(raw[i//8]&(1<<(i%8)))

    def read_window(self,x0,y0,z0,size,*,packed=False):
        if any(not isinstance(v,(int,np.integer)) for v in (x0,y0,z0,size)):raise TypeError('Window coordinates must be integers')
        x0,y0,z0,size=map(int,(x0,y0,z0,size))
        if size<1 or min(x0,y0,z0)<0 or any(o+size>n for o,n in zip((z0,y0,x0),self.shape)):raise ValueError('Window outside logical shape')
        out=np.zeros((size,size,(size+7)//8),dtype=np.uint8)
        for bz in range(z0//64,(z0+size-1)//64+1):
            for by in range(y0//64,(y0+size-1)//64+1):
                for bx in range(x0//64,(x0+size-1)//64+1):
                    starts=[max(b*64,o) for b,o in zip((bz,by,bx),(z0,y0,x0))]
                    stops=[min(b*64+64,o+size) for b,o in zip((bz,by,bx),(z0,y0,x0))]
                    local=tuple(slice(a-b*64,c-b*64) for a,c,b in zip(starts,stops,(bz,by,bx)))
                    a=self.read_global_brick((bx,by,bz))[local]
                    dz,dy,dx=(a-o for a,o in zip(starts,(z0,y0,x0)))
                    shifted=np.pad(a,((0,0),(0,0),(dx%8,0)))
                    bits=np.packbits(shifted,axis=-1,bitorder='little')
                    out[dz:dz+a.shape[0],dy:dy+a.shape[1],dx//8:dx//8+bits.shape[2]]|=bits
        result=PackedWindow(out,(size,size,size))
        return result if packed else result.unpack()

    def verify_all(self):
        before=self.cache_bricks;self.cache_bricks=0;self.cache.clear()
        try:
            gz,gy,gx=self.grid
            for z in range(gz):
                for y in range(gy):
                    for x in range(gx):self.read_brick_packed((x,y,z))
        finally:self.cache_bricks=before
        return len(self.index)


class RINDPackWriter:
    def __init__(self,path,shape,*,identity,policy,resume=False,level=3):
        from ._native import PackWriter
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.identity=identity;self.policy=policy
        self.journal=self.path.with_name(self.path.name+'.journal.json')
        previous=json.loads(self.journal.read_text()) if resume and self.journal.exists() else None
        if previous and (previous['identity']!=identity or previous['policy']!=policy):raise ValueError('Resume journal identity mismatch')
        self.native=PackWriter(str(path),list(shape),bytes.fromhex(identity),bytes.fromhex(policy),resume,level)
        if previous and not set(previous['committed'])<=self.committed:raise ValueError('A previously journaled committed brick was lost/corrupted')
        self.checkpoint()

    @property
    def committed(self):return set(self.native.committed())

    def write(self,index,packed_bytes):self.native.write(int(index),bytes(packed_bytes))

    def checkpoint(self):
        self.native.checkpoint()
        atomic_json(self.journal,{'format':'RINDPack-journal-v1','identity':self.identity,'policy':self.policy,'committed':sorted(self.committed)})

    def finish(self):
        self.native.finish()
        digest=sha256_file(self.path)
        atomic_text(self.path.with_name(self.path.name+'.sha256'),digest+'  '+self.path.name+'\n')
        self.journal.unlink(missing_ok=True)
        return digest
