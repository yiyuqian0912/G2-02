"""Small B2-only API. No original renderer, profile or public reader is changed."""
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from ..queries import Decision
from ..models import record as geometry_record
from .build import sources


def coordinates(value):
    raw=np.asarray(value)
    if raw.dtype.kind not in 'fiu' or (raw.dtype.kind=='f' and raw.dtype.itemsize>8):
        raise TypeError('Native endpoints must be binary64-compatible coordinates; exact rational endpoints need the original rational oracle')
    return np.asarray(raw,dtype='<f8')


def extension():
    build=Path(os.environ.get('RIND_B2_NATIVE_BUILD',Path(__file__).parents[3]/'outputs/gaussian-b2-native-v1/build'))
    identity=build/'build-identity.json';paths=list(build.glob('_b2_native*.so'))
    if len(paths)!=1 or not identity.exists():raise ImportError('Build B2 with python -m experiments.gaussian_geometry.b2_native.build')
    record=json.loads(identity.read_text())
    if record['source_sha256']!=sources() or record['extension_sha256']!=hashlib.sha256(paths[0].read_bytes()).hexdigest():raise ImportError('B2 source/binary identity changed; rebuild research extension')
    name='_b2_native'
    if name in sys.modules and getattr(sys.modules[name],'__b2_binary_sha256',None)!=record['extension_sha256']:
        raise ImportError('B2 binary changed after import; restart Python to preserve the numerical build identity')
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,paths[0]);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        module.__b2_binary_sha256=record['extension_sha256']
    return sys.modules[name]


class NativeOracle:
    ready_capable=False

    def __init__(self,scene,*,independent=False,**options):
        if scene.kind!='gaussian_sum':raise ValueError('B2 only: a Gaussian sum is never replaced by B1/A/C')
        self.scene=scene;self.independent=independent;self.module=extension();self.options=self.module.Options()
        self.geometry_id=geometry_record(scene)['geometry_id']
        for key,value in options.items():
            if not hasattr(self.options,key):raise ValueError('Unknown B2 option: '+key)
            setattr(self.options,key,value)
        cls=self.module.Independent if independent else self.module.Primary
        self.native=cls([k.center.tolist() for k in scene.kernels],[k.matrix.ravel().tolist() for k in scene.kernels],
                        [k.weight for k in scene.kernels],scene.threshold.value,scene.threshold.kind=='exp')
        self.work=Counter()

    def _convert(self,raw,point=False):
        labels={0:'FREE' if point else 'MISS',1:'SOLID' if point else 'HIT',-1:'UNKNOWN',-2:'INVALID'}
        return Decision(labels[raw['status']],raw['reason'],raw['nodes'],raw['depth'])

    def points(self,points):
        rows,work=self.native.points(coordinates(points).reshape(-1,3),self.options);self.work.update(work)
        return [self._convert(x,True) for x in rows]

    def rays(self,source,receivers):
        rows,work=self.native.rays(coordinates(source),coordinates(receivers).reshape(-1,3),self.options);self.work.update(work)
        return [self._convert(x) for x in rows]

    def membership(self,p):return self.points([p])[0]

    def segment(self,s,p):return self.rays(s,[p])[0]

    def classify_box(self,lo,hi):
        row,work=self.native.box(coordinates(lo),coordinates(hi),self.options);self.work.update(work);return self._convert(row,True)

    def certificate(self,source,lo,hi):
        result,work=self.native.certificate(coordinates(source),coordinates(lo),coordinates(hi),self.options);self.work.update(work);return result

    def source_clearance(self,p,radius=2**-26):
        p=coordinates(p).reshape(3)
        if not np.isfinite(p).all() or np.any((p<=0)|(p>=1)) or not np.isfinite(radius) or radius<=0:
            return Decision('INVALID','source/radius outside declared domain')
        lo=np.nextafter(p-radius,-np.inf);hi=np.nextafter(p+radius,np.inf)
        result=self.classify_box(lo,hi)
        return Decision('FREE','entire outward epsilon cube FREE') if result.status=='FREE' else Decision('UNKNOWN','clearance not proved; not a distance rejection')

    def field(self,source,resolution,seconds=120.,*,scalar=False):
        result=self.native.field(coordinates(source).reshape(3),resolution,seconds,self.options,scalar)
        result['geometry_id']=self.geometry_id
        self.work.update(result['work'])
        for key in ('response','solid','valid'):result[key]=result[key].astype(bool)
        return result
