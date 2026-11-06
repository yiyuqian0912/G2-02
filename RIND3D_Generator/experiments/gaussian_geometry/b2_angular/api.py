"""Small B2-only research interface. No public reader or production change."""
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
    directory=Path(os.environ.get('RIND_B2_ANGULAR_BUILD',Path(__file__).parents[3]/'outputs/gaussian-b2-angular-v1/build'))
    paths=list(directory.glob('_b2_angular*.so'));identity=directory/'build-identity.json'
    if len(paths)!=1 or not identity.exists():raise ImportError('Build b2_angular first')
    row=json.loads(identity.read_text());digest=hashlib.sha256(paths[0].read_bytes()).hexdigest()
    if row['source_sha256']!=sources() or row['extension_sha256']!=digest:raise ImportError('Angular source/binary changed; rebuild into a new version')
    name='_b2_angular'
    if name in sys.modules and getattr(sys.modules[name],'__binary_sha',None)!=digest:raise ImportError('Loaded binary changed; restart Python')
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,paths[0]);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);module.__binary_sha=digest
    return sys.modules[name]


class AngularOracle:
    ready_capable=False

    def __init__(self,scene,*,independent=False,**options):
        if scene.kind!='gaussian_sum':raise ValueError('Continuous B2 only')
        self.module=extension();self.settings=self.module.Settings();self.options=self.module.Options()
        self.geometry_id=record(scene)['geometry_id'];self.independent=independent
        for k,v in options.items():
            target=self.settings if hasattr(self.settings,k) else self.options
            if not hasattr(target,k):raise ValueError('Unknown research option '+k)
            setattr(target,k,v)
        self.native=self.module.Engine([k.center.tolist() for k in scene.kernels],[k.matrix.ravel().tolist() for k in scene.kernels],
                                      [k.weight for k in scene.kernels],scene.threshold.value,scene.threshold.kind=='exp',independent)

    def field(self,source,resolution,seconds=300):
        return self.tile(source,resolution,[0,0,0],resolution,seconds)

    def tile(self,source,world_resolution,origin,size,seconds=300):
        raw=np.asarray(origin)
        if raw.shape!=(3,) or raw.dtype.kind not in 'iu':raise TypeError('Integer XYZ tile origin required')
        result=self.native.tile(coordinates(source).reshape(3),world_resolution,raw.tolist(),size,seconds,self.options,self.settings)
        for k in ('response','solid','valid'):result[k]=result[k].astype(bool)
        result['geometry_id']=self.geometry_id
        return result
