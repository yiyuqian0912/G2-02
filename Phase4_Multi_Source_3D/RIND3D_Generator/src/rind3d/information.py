"""Exact packed-field existence index; no geometry or visibility queries.

Each node stores two existence bits: lit FREE (1), shadowed FREE (2).
An intersecting partial node is never treated as a contained node. Partial
16-cubes are resolved from real packed bits, including first/last-byte masks.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
import hashlib,json,numbers,os,struct,time,zlib
import numpy as np
from .pack import RINDPack,RAW,sha256_file,atomic_text

MAGIC=b'RINDINF1'
BLOCK=16
_UNIFORM_CRC=(zlib.crc32(bytes(RAW)),zlib.crc32(bytes([255])*RAW))


def _integer(value):
    if isinstance(value,bool) or not isinstance(value,numbers.Integral):raise TypeError('Integer coordinates required')
    return int(value)


def _digest(reader):
    digest=sha256_file(reader.path);sidecar=reader.path.with_name(reader.path.name+'.sha256')
    if sidecar.exists() and sidecar.read_text().split()[0]!=digest:raise ValueError('Input field SHA256 mismatch')
    return digest


def _parent(child):
    shape=tuple((n+1)//2 for n in child.shape);parent=np.zeros(shape,dtype=np.uint8)
    for z in (0,1):
        for y in (0,1):
            for x in (0,1):
                a=child[z::2,y::2,x::2];parent[:a.shape[0],:a.shape[1],:a.shape[2]]|=a
    return parent


def _constant(reader,index):
    row=reader.index[index]
    if row[2]>=2:return None
    if row[5]!=_UNIFORM_CRC[row[2]]:raise ValueError('Uniform brick decoded checksum mismatch')
    return row[2]


def _two_bit_pack(values):
    a=np.zeros((len(values)+3)//4*4,dtype=np.uint8);a[:len(values)]=values
    return (a[0::4]|(a[1::4]<<2)|(a[2::4]<<4)|(a[3::4]<<6)).tobytes()


def _two_bit_unpack(payload,count):
    a=np.frombuffer(payload,dtype=np.uint8);out=np.empty(len(a)*4,dtype=np.uint8)
    for i in range(4):out[i::4]=(a>>(2*i))&3
    if np.any(out[count:]):raise ValueError('Nonzero index tail padding')
    return out[:count]


@dataclass(frozen=True)
class InformationResult:
    flags:int
    lit:tuple[int,int,int]|None=None
    shadow:tuple[int,int,int]|None=None
    @property
    def informative(self):return self.flags==3


class InformationIndex:
    def __init__(self,response,solid,levels,metadata,*,owned=False):
        if response.shape!=solid.shape or response.identity!=solid.identity or response.policy!=solid.policy:
            raise ValueError('Response/solid shape, identity or policy mismatch')
        self.response=response;self.solid=solid;self.shape=response.shape;self.levels=levels;self.metadata=metadata;self.owned=owned
        self.counters=Counter();self.build_report={}
    @property
    def nbytes(self):return sum(a.nbytes for a in self.levels)
    @property
    def global_flags(self):return int(self.levels[-1][0,0,0])
    def close(self):
        if self.owned:self.response.close();self.solid.close()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()

    @classmethod
    def build(cls,response,solid):
        """One ordered brick/index pass; immutable compressed-file hashes separate."""
        start=time.perf_counter();cpu=time.process_time();t=time.perf_counter()
        keys={'response_sha256':_digest(response),'solid_sha256':_digest(solid)}
        verify_seconds=time.perf_counter()-t
        if response.shape!=solid.shape or response.identity!=solid.identity or response.policy!=solid.policy:
            raise ValueError('Response/solid identity mismatch')
        shape=tuple((n+BLOCK-1)//BLOCK for n in response.shape);leaf=np.zeros(shape,dtype=np.uint8)
        gz,gy,gx=response.grid;uniform=0;decoded_before=response.decoded_bricks+solid.decoded_bricks
        for bz in range(gz):
            for by in range(gy):
                for bx in range(gx):
                    index=(bz*gy+by)*gx+bx;a=_constant(response,index);b=_constant(solid,index)
                    target=tuple(slice(v*4,min(v*4+4,n)) for v,n in zip((bz,by,bx),shape))
                    if a is not None and b is not None:
                        if a and b:raise ValueError('Stored response lights solid voxels')
                        leaf[target]=0 if b else 1 if a else 2;uniform+=1;continue
                    y=np.frombuffer(response.read_brick_packed((bx,by,bz)),dtype=np.uint8).reshape(64,64,8)
                    m=np.frombuffer(solid.read_brick_packed((bx,by,bz)),dtype=np.uint8).reshape(64,64,8)
                    valid=tuple(min(64,n-o*64) for n,o in zip(response.shape,(bz,by,bx)))
                    mask=None
                    if valid!=(64,64,64):
                        mask=np.zeros((64,64,8),dtype=np.uint8);zv,yv,xv=valid
                        mask[:zv,:yv,:xv//8]=255
                        if xv%8:mask[:zv,:yv,xv//8]=(1<<(xv%8))-1
                    bad=y&m
                    if mask is not None:bad=bad&mask
                    if np.any(bad):raise ValueError('Stored response lights solid voxels')
                    lit=y&~m;shadow=~y&~m
                    if mask is not None:lit&=mask;shadow&=mask
                    flags=lit.reshape(4,16,4,16,4,2).any(axis=(1,3,5)).astype(np.uint8)
                    flags|=shadow.reshape(4,16,4,16,4,2).any(axis=(1,3,5)).astype(np.uint8)<<1
                    dest=leaf[target];dest[:]=flags[:dest.shape[0],:dest.shape[1],:dest.shape[2]]
        levels=[leaf]
        while levels[-1].shape!=(1,1,1):levels.append(_parent(levels[-1]))
        metadata={'format':'RIND-information-index-v1','block_side':BLOCK,'shape_zyx':list(response.shape),
                  'identity':response.identity,'policy':response.policy,**keys,'level_shapes':[list(a.shape) for a in levels],
                  'class_bits':{'lit_free':1,'shadow_free':2},'witness_policy':'hashed-child-order, certified packed bit'}
        result=cls(response,solid,levels,metadata)
        result.build_report={'wall_seconds':time.perf_counter()-start,'process_seconds':time.process_time()-cpu,
            'input_hash_verification_seconds':verify_seconds,'ordered_bricks':gx*gy*gz,'uniform_brick_pairs_summarized':uniform,
            'decoded_bricks':response.decoded_bricks+solid.decoded_bricks-decoded_before,'index_ram_bytes':result.nbytes,
            'largest_packed_brick_bytes':RAW,'oracle_calls':0,'global_flags':result.global_flags}
        return result

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        payload=_two_bit_pack(np.concatenate([a.ravel() for a in self.levels]))
        meta=dict(self.metadata,payload_sha256=hashlib.sha256(payload).hexdigest())
        header=json.dumps(meta,sort_keys=True,separators=(',',':')).encode();tmp=path.with_name(path.name+'.tmp')
        with tmp.open('wb') as f:
            f.write(MAGIC);f.write(struct.pack('<I',len(header)));f.write(header);f.write(payload);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path);atomic_text(path.with_name(path.name+'.sha256'),sha256_file(path)+'  '+path.name+'\n')
        return path.stat().st_size

    @classmethod
    def load(cls,path,response,solid,*,verify_inputs=True):
        path=Path(path)
        sidecar=path.with_name(path.name+'.sha256')
        if sidecar.exists() and sha256_file(path)!=sidecar.read_text().split()[0]:raise ValueError('Information-index file checksum mismatch')
        with path.open('rb') as f:
            if f.read(8)!=MAGIC:raise ValueError('Invalid information-index magic')
            raw=f.read(4)
            if len(raw)!=4:raise ValueError('Truncated information-index header')
            length=struct.unpack('<I',raw)[0]
            if not 1<=length<=16384:raise ValueError('Invalid information-index header length')
            meta=json.loads(f.read(length));payload=f.read()
        if meta.get('format')!='RIND-information-index-v1' or meta.get('block_side')!=16:raise ValueError('Unsupported information index')
        if list(response.shape)!=meta['shape_zyx'] or response.identity!=meta['identity'] or response.policy!=meta['policy']:
            raise ValueError('Index belongs to different stored truth')
        if verify_inputs and (_digest(response)!=meta['response_sha256'] or _digest(solid)!=meta['solid_sha256']):raise ValueError('Stale information index')
        if hashlib.sha256(payload).hexdigest()!=meta['payload_sha256']:raise ValueError('Information-index payload checksum mismatch')
        shapes=[];shape=tuple((n+15)//16 for n in response.shape)
        while True:
            shapes.append(shape)
            if shape==(1,1,1):break
            shape=tuple((n+1)//2 for n in shape)
        if [list(s) for s in shapes]!=meta['level_shapes']:raise ValueError('Invalid index hierarchy shapes')
        counts=[int(np.prod(s)) for s in shapes];total=sum(counts)
        if len(payload)!=(total+3)//4:raise ValueError('Invalid information-index payload length')
        data=_two_bit_unpack(payload,total);levels=[];offset=0
        for shape,count in zip(shapes,counts):levels.append(data[offset:offset+count].reshape(shape));offset+=count
        for child,parent in zip(levels,levels[1:]):
            if not np.array_equal(_parent(child),parent):raise ValueError('Information-index aggregation mismatch')
        return cls(response,solid,levels,meta)

    def _leaf(self,lower,upper,missing,witnesses,seed):
        """Intersected ROI lies in one 16-subblock, hence one 64-brick."""
        self.counters['partial_or_witness_leaf_scans']+=1
        x,y,z=lower;xx,yy,zz=upper;bx,by,bz=x//64,y//64,z//64
        sl=(slice(z%64,z%64+zz-z),slice(y%64,y%64+yy-y),slice(x%64//8,(xx-bx*64+7)//8))
        a=np.frombuffer(self.response.read_brick_packed((bx,by,bz)),dtype=np.uint8).reshape(64,64,8)[sl]
        m=np.frombuffer(self.solid.read_brick_packed((bx,by,bz)),dtype=np.uint8).reshape(64,64,8)[sl]
        result=0;points={}
        for flag,values in ((1,a&~m),(2,~a&~m)):
            if not missing&flag:continue
            values=values.copy();values[:,:,0]&=(255<<(x%8))&255
            if xx%8:values[:,:,-1]&=(1<<(xx%8))-1
            indices=np.flatnonzero(values)
            if not len(indices):continue
            result|=flag
            if witnesses:
                flat=int(indices[seed%len(indices)]);iz,iy,ib=np.unravel_index(flat,values.shape)
                value=int(values[iz,iy,ib]);bits=[i for i in range(8) if value&(1<<i)];bit=bits[(seed>>9)%len(bits)]
                points[flag]=(bx*64+(x%64//8+int(ib))*8+bit,y+int(iy),z+int(iz))
        return result,points

    def query(self,origin,size,*,witnesses=False,witness_seed=0):
        origin=tuple(_integer(v) for v in origin);size=_integer(size)
        if len(origin)!=3 or size<1 or min(origin)<0 or any(a+size>n for a,n in zip(origin,self.shape[::-1])):
            raise ValueError('Window outside information-index shape')
        self.counters['queries']+=1;lo=origin;hi=tuple(v+size for v in lo);found=0;points={}
        seed=((lo[0]*73856093)^(lo[1]*19349663)^(lo[2]*83492791)^(size*2654435761)^int(witness_seed))&0xffffffff
        def visit(level,iz,iy,ix):
            nonlocal found
            self.counters['nodes_visited']+=1
            flag=int(self.levels[level][iz,iy,ix]);missing=flag&~found
            if not missing:return
            scale=BLOCK<<level;lower=(ix*scale,iy*scale,iz*scale)
            upper=tuple(min(a+scale,n) for a,n in zip(lower,self.shape[::-1]))
            if any(a>=h or b<=l for a,b,l,h in zip(lower,upper,lo,hi)):return
            contained=all(l<=a and b<=h for a,b,l,h in zip(lower,upper,lo,hi))
            if contained and not witnesses:found|=flag;return
            if level==0:
                lower=tuple(max(a,b) for a,b in zip(lower,lo));upper=tuple(min(a,b) for a,b in zip(upper,hi))
                bits,proof=self._leaf(lower,upper,missing,witnesses,seed^ix^(iy<<6)^(iz<<12));found|=bits;points.update(proof);return
            shape=self.levels[level-1].shape;rotation=(seed+level+ix+iy*3+iz*7)%8
            for k in range(8):
                n=(k+rotation)%8;z=iz*2+(n>>2);y=iy*2+((n>>1)&1);x=ix*2+(n&1)
                if z<shape[0] and y<shape[1] and x<shape[2]:visit(level-1,z,y,x)
                if found==3:break
        visit(len(self.levels)-1,0,0,0)
        return InformationResult(found,points.get(1),points.get(2))

    def point_class(self,coordinate):
        x,y,z=map(_integer,coordinate)
        if self.solid.read_point(x,y,z):return 0
        return 1 if self.response.read_point(x,y,z) else 2
