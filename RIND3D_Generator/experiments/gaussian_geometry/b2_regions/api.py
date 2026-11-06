"""Continuous whole-region experiment, separate from the immutable v1 API."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from ..b2_native.api import coordinates
from ..models import record
from .build import sources


def extension():
    directory=Path(os.environ.get('RIND_B2_REGIONS_BUILD',Path(__file__).parents[3]/'outputs/gaussian-b2-regions-v1/build'))
    paths=list(directory.glob('_b2_regions*.so'));identity=directory/'build-identity.json'
    if len(paths)!=1 or not identity.exists():raise ImportError('Build b2_regions first')
    row=json.loads(identity.read_text())
    if row['source_sha256']!=sources() or row['extension_sha256']!=hashlib.sha256(paths[0].read_bytes()).hexdigest():raise ImportError('Region source/binary changed; rebuild')
    name='_b2_regions'
    if name in sys.modules and getattr(sys.modules[name],'__binary_sha',None)!=row['extension_sha256']:raise ImportError('Loaded binary changed; restart Python')
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,paths[0]);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);module.__binary_sha=row['extension_sha256']
    return sys.modules[name]


class RegionOracle:
    ready_capable=False

    def __init__(self,scene,*,independent=False,**options):
        if scene.kind!='gaussian_sum':raise ValueError('Continuous B2 only')
        self.module=extension();self.options=self.module.Options();self.settings=self.module.Settings();self.independent=independent
        self.geometry_id=record(scene)['geometry_id']
        for k,v in options.items():
            target=self.settings if hasattr(self.settings,k) else self.options
            if not hasattr(target,k):raise ValueError('Unknown option '+k)
            setattr(target,k,v)
        self.native=self.module.Engine([k.center.tolist() for k in scene.kernels],[k.matrix.ravel().tolist() for k in scene.kernels],
                                      [k.weight for k in scene.kernels],scene.threshold.value,scene.threshold.kind=='exp',independent)

    def region(self,source,lo,hi):
        return self.native.region(coordinates(source).reshape(3),coordinates(lo).reshape(3),coordinates(hi).reshape(3),self.options,self.settings)

    def field(self,source,resolution,seconds=180):
        result=self.native.field(coordinates(source).reshape(3),resolution,seconds,self.options,self.settings)
        for k in ('response','solid','valid'):result[k]=result[k].astype(bool)
        result['geometry_id']=self.geometry_id
        return result

    def tile(self,source,world_resolution,origin,size,seconds=180):
        raw=np.asarray(origin)
        if raw.shape!=(3,) or raw.dtype.kind not in 'iu':raise TypeError('Integer XYZ tile origin required')
        result=self.native.tile(coordinates(source).reshape(3),world_resolution,raw.tolist(),size,seconds,self.options,self.settings)
        for k in ('response','solid','valid'):result[k]=result[k].astype(bool)
        result['geometry_id']=self.geometry_id
        return result
