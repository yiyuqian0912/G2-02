"""Research solver and separately implemented MISS-proof checker.

Proof checking is not the existing independent physical-renderer contract.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

from ..b2_native.api import coordinates
from ..models import record
from .build import sources

SCHEMA='rind3d-b2-miss-certificate-v1'
OPS={0:'natural',1:'increasing',2:'decreasing',3:'local_taylor',4:'stationary'}


def extension():
    directory=Path(os.environ.get('RIND_B2_MAX_CERT_BUILD',Path(__file__).resolve().parents[3]/'outputs/gaussian-b2-max-cert-v1/build'))
    files=list(directory.glob('_b2_max_cert*.so'));identity=directory/'identity.json'
    if len(files)!=1 or not identity.exists():raise ImportError('Build b2_max_cert first')
    row=json.loads(identity.read_text());digest=hashlib.sha256(files[0].read_bytes()).hexdigest()
    if row['source_SHA256']!=sources() or row['binary_SHA256']!=digest:raise ImportError('Maximum-proof source/binary changed; rebuild into a new version')
    name='_b2_max_cert'
    if name in sys.modules and sys.modules[name].__binary_SHA256!=digest:raise ImportError('Loaded library changed; restart Python')
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,files[0]);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);module.__binary_SHA256=digest
    return sys.modules[name]


def parameters(scene):
    if scene.kind!='gaussian_sum':raise ValueError('Continuous positive B2 Gaussian sum required')
    return ([k.center.tolist() for k in scene.kernels],[k.matrix.ravel().tolist() for k in scene.kernels],
            [k.weight for k in scene.kernels],scene.threshold.value,scene.threshold.kind=='exp')


def query_identity(geometry,source,receiver):
    value={'geometry_id':geometry,'source_hex':[float(x).hex() for x in source],'receiver_hex':[float(x).hex() for x in receiver]}
    return {**value,'query_id':hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()}


def dumps(certificate):return json.dumps(certificate,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate_JSON_field')
        result[key]=value
    return result


class MaximumOracle:
    ready_capable=False

    def __init__(self,scene,**settings):
        self.module=extension();self.native=self.module.Solver(*parameters(scene));self.settings=self.module.Settings()
        self.geometry_id=record(scene)['geometry_id'];self.threshold={'kind':scene.threshold.kind,'value_hex':float(scene.threshold.value).hex()}
        for k,v in settings.items():
            if not hasattr(self.settings,k):raise ValueError('Unknown maximum-proof option '+k)
            setattr(self.settings,k,v)

    def segment(self,source,receiver,*,proof=False):
        source=coordinates(source).reshape(3);receiver=coordinates(receiver).reshape(3)
        result=self.native.solve(source,receiver,self.settings,proof);leaves=result.pop('leaves')
        result['status']={0:'MISS',1:'HIT',-1:'UNKNOWN',-2:'INVALID'}[result['status']]
        result['certificate']=None
        if proof and result['status']=='MISS' and leaves:
            rows=[]
            for leaf in leaves:
                row={k:float(getattr(leaf,k)).hex() for k in ('lo','hi','upper')};row['op']=OPS[leaf.op]
                if leaf.op==4:
                    row.update(jlo=float(leaf.jlo).hex(),jhi=float(leaf.jhi).hex(),empty=leaf.empty,inner_op=OPS[leaf.inner_op])
                rows.append(row)
            result['certificate']={'schema':SCHEMA,**query_identity(self.geometry_id,source,receiver),'threshold':self.threshold,
                                   'enclosure':'outward-binary64-algebra-directed-MPFR-exp-v1','partition':rows}
        return result


class CertificateChecker:
    """Construct from the expected canonical geometry; caller supplies query.

    Identities, coverage and numerical claims are all checked. A checksum alone
    is never physical evidence. Malformed/unsupported/budget-limited proofs fail.
    """
    ready_capable=False

    def __init__(self,scene):
        self.module=extension();self.native=self.module.Checker(*parameters(scene));self.geometry_id=record(scene)['geometry_id']
        self.threshold={'kind':scene.threshold.kind,'value_hex':float(scene.threshold.value).hex()}

    def check(self,certificate,source,receiver,*,precision=128,seconds=2.):
        try:
            if isinstance(certificate,(bytes,str)):
                if len(certificate)>8*1024*1024:raise ValueError('proof_size_budget')
                certificate=json.loads(certificate,object_pairs_hook=unique_object)
            if not isinstance(certificate,dict):raise ValueError('missing_certificate')
            required={'schema','geometry_id','source_hex','receiver_hex','query_id','threshold','enclosure','partition'}
            if set(certificate)!=required:raise ValueError('unsupported_envelope_fields')
            if certificate['schema']!=SCHEMA or certificate['enclosure']!='outward-binary64-algebra-directed-MPFR-exp-v1':raise ValueError('unsupported_certificate_version')
            source=coordinates(source).reshape(3);receiver=coordinates(receiver).reshape(3)
            identity=query_identity(self.geometry_id,source,receiver)
            if any(certificate[k]!=v for k,v in identity.items()):raise ValueError('stale_geometry_or_query_identity')
            if certificate['threshold']!=self.threshold:raise ValueError('threshold_identity_mismatch')
            rows=certificate['partition']
            if not isinstance(rows,list) or not 1<=len(rows)<=8192:raise ValueError('partition_budget')
            reverse={v:k for k,v in OPS.items()};leaves=[]
            def dyadic(s):
                if not isinstance(s,str) or len(s)>64:raise ValueError('invalid_dyadic_encoding')
                value=float.fromhex(s)
                if not math.isfinite(value) or value.hex()!=s:raise ValueError('noncanonical_dyadic_encoding')
                return value
            for row in rows:
                if not isinstance(row,dict) or row.get('op') not in reverse:raise ValueError('unsupported_proof_operation')
                keys={'lo','hi','upper','op'}|({'jlo','jhi','empty','inner_op'} if row['op']=='stationary' else set())
                if set(row)!=keys:raise ValueError('unsupported_leaf_fields')
                leaf=self.module.Leaf();leaf.op=reverse[row['op']]
                for k in ('lo','hi','upper'):setattr(leaf,k,dyadic(row[k]))
                if leaf.op==4:
                    leaf.jlo=dyadic(row['jlo']);leaf.jhi=dyadic(row['jhi'])
                    if type(row['empty']) is not bool or row['inner_op'] not in ('natural','local_taylor'):raise ValueError('unsupported_stationary_evidence')
                    leaf.empty=row['empty'];leaf.inner_op=reverse[row['inner_op']]
                    if leaf.empty and (leaf.jlo!=leaf.lo or leaf.jhi!=leaf.lo or leaf.inner_op!=0):raise ValueError('noncanonical_empty_stationary_region')
                leaves.append(leaf)
            return self.native.check(source,receiver,leaves,precision,seconds)
        except (ValueError,TypeError,KeyError,OverflowError) as error:
            return {'accepted':False,'reason':'invalid_certificate:'+str(error),'work':{}}
