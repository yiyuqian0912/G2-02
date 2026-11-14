"""Build-bound fixture gate, distinct from per-scene mathematical acceptance."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from .pack import atomic_json
from .geometry import Oracle,centres,Unsupported
from .lite_generators import FAMILIES,generate
from .lite_acceptance import geometry_acceptance,sample_source,DRIVER_RADIUS
from .rendering.context import GeometryContext
from .rendering.backends import BackendFactory

VERSION='lite-scientific-fixtures-v0.1'

def scientific_identity():
    from . import _convex
    from rind.geometry import backend_build_id
    package=Path(__file__).resolve().parent
    files=list((package/'rendering').glob('*.py'))+list((package/'rendering').glob('*.cu'))
    files += [package/n for n in ('geometry.py','lite_acceptance.py','lite_generators.py','profiles.py','lite_certification.py',
              'physical_evidence.py','lite_search.py','lite_publication.py','lite_pipeline.py','window_contract.py','stored_windows.py','stored_guidance.py','information.py','pack.py')]
    values={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    values['native_binary']=hashlib.sha256(Path(_convex.__file__).read_bytes()).hexdigest()
    root=package.parents[1];native=root/'native/convex.cpp'
    if native.exists():values['native_source']=hashlib.sha256(native.read_bytes()).hexdigest()
    values['canonical_backend']=backend_build_id(3)
    return {'files':values,'sha256':hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()}

def validate_certificate(value,*,current=True):
    if value.get('protocol')!=VERSION or not value.get('complete') or set(value.get('families',[]))!=set(FAMILIES):raise ValueError('Incomplete scientific fixture certificate')
    if current and value.get('scientific_identity')!=scientific_identity():raise ValueError('Scientific implementation changed; rerun certify-lite')
    cases=value.get('cases',[])
    if len(cases)<len(FAMILIES) or any(not c.get('passed') for c in cases):raise ValueError('Family fixture suite did not pass')
    if value.get('hard_cases_passed') is not True or value.get('driver_clearance_hex')!=DRIVER_RADIUS.hex():raise ValueError('Scientific topology/clearance fixtures missing')
    return value

def small_field(source,n):
    y=np.empty((n,)*3,dtype=bool);m=np.empty_like(y)
    for z in range(0,n,64):
        for v in range(0,n,64):
            for x in range(0,n,64):
                side=min(n,64);a,b,_=source.render((x,v,z),side,n)
                y[z:z+side,v:v+side,x:x+side]=a;m[z:z+side,v:v+side,x:x+side]=b
    return y,m

def cuda_field(source,n):
    side=min(n,64);origins=[(x,v,z) for z in range(0,n,64) for v in range(0,n,64) for x in range(0,n,64)]
    y=np.empty((n,)*3,dtype=bool);m=np.empty_like(y)
    for start in range(0,len(origins),source.device.batch):
        part=origins[start:start+source.device.batch];pairs=source.packed_batch(part,side,n)
        for (x,v,z),pair in zip(part,pairs):
            for raw,out in zip(pair,(y,m)):
                a=np.unpackbits(np.frombuffer(raw,dtype=np.uint8),bitorder='little').reshape((64,)*3)
                out[z:z+side,v:v+side,x:x+side]=a[:side,:side,:side]
    return y,m

def hard_cases(factory):
    from rind.contracts import make_leaf,Geometry,RootKind,CSGNode,CSGOpcode
    from rind.storage import CanonicalBundle
    def bundle(c=[.5,.5,.5],h=[.125,.125,.125]):
        b=CanonicalBundle(3);leaf=make_leaf('affine_box_v1',3,center=c,matrix=np.diag(h));b.add_leaf(leaf);b.add_geometry(Geometry(3,RootKind.REF,leaf.content_hash));return b
    g=GeometryContext(bundle());s=np.array([.0625,.5,.5]);a=g.bind(s);b=g.bind(s,independent=True)
    points=np.array([[.375,.5,.5],[.6875,.75,.5],[.6875,.75,.75],[.6875,np.nextafter(.75,1.),.5]])
    y=a.response(s,points);other=b.response(s,points)
    oracle=g.canonical;canonical=oracle._values(oracle.reference.segment_intersect(np.repeat(s[None],len(points),axis=0),points))==0
    assert np.array_equal(y,other) and np.array_equal(y,canonical) and y.tolist()==[False,False,False,True]
    # Clearance LE/equality rejection is certified by the original service,
    # never inferred from failure of a sufficient separating-plane GT proof.
    assert g.clearance([.375-DRIVER_RADIUS,.5,.5],DRIVER_RADIUS)['status']=='LE'
    assert g.clearance([.375-2*DRIVER_RADIUS,.5,.5],DRIVER_RADIUS)['status']=='GT'
    from .lite_acceptance import verify_geometry_proof
    cross=bundle([.98,.5,.5],[.1,.1,.1]);c=GeometryContext(cross)
    try:verify_geometry_proof(cross,c.geometry_proof())
    except ValueError:pass
    else:raise AssertionError('World-face crossing was published in a strictly-contained profile')
    try:GeometryContext(bundle([1.125,.5,.5],[.125]*3))
    except Unsupported:pass
    else:raise AssertionError('Face-only world component entered native scope')
    # Shared-face union topology uses canonical fallback when no single
    # component proves interior; closed binary membership stays exact.
    union=CanonicalBundle(3);ids=[]
    for x in (.375,.625):
        leaf=make_leaf('affine_box_v1',3,center=[x,.5,.5],matrix=np.diag([.125,.125,.125]));union.add_leaf(leaf);ids.append(leaf.content_hash)
    node=CSGNode(3,CSGOpcode.UNION,tuple(ids));union.add_node(node);union.add_geometry(Geometry(3,RootKind.REF,node.content_hash))
    u=GeometryContext(union);assert u.membership([[.5,.5,.5]]).tolist()==[1]
    thin=GeometryContext(bundle([.5]*3,[.2,.2,2**-40]));thin.geometry_proof()
    q=np.array([.1,.5,.5+2**-40]);p=np.array([[.9,.5,.5+2**-40],[.9,.5,np.nextafter(.5+2**-40,1.)]])
    assert np.array_equal(thin.bind(q).response(q,p),thin.bind(q,independent=True).response(q,p))
    return True

def certify(path,*,cuda=True):
    path=Path(path)
    if path.exists():return validate_certificate(json.loads(path.read_text()))
    started=time.perf_counter();identity=scientific_identity();cases=[]
    def progress(family,stage,**extra):
        atomic_json(path.with_suffix('.progress.json'),{'complete':False,'family':family,'stage':stage,'cases':cases,'elapsed_seconds':time.perf_counter()-started,**extra})
    with BackendFactory() as factory:
        for index,family in enumerate(FAMILIES):
            progress(family,'geometry_acceptance')
            start=time.perf_counter();bundle,recipe=generate(20261101,index,family)
            if recipe['proposal_failures']:raise ValueError('Fixture recipe proposal invalid')
            context=factory.context(bundle);accepted=geometry_acceptance(context,family,20261101,index);source,_=sample_source(context,20261101,index)
            a=context.bind(source);b=context.bind(source,independent=True)
            progress(family,'original_canonical_initialization')
            reference=Oracle(bundle);points=np.asarray(accepted['point_inputs'],dtype='<f8')
            progress(family,'original_canonical_points_and_rays')
            labels=reference._values(reference.reference.classify_points(points));rays=reference._values(reference.reference.segment_intersect(np.repeat(source[None],len(points),axis=0),points))==0
            if not np.array_equal(context.membership(points),labels) or not np.array_equal(a.response(source,points),rays) or not np.array_equal(b.response(source,points),rays):raise AssertionError('Original canonical fixture comparison failed')
            starts=np.asarray(accepted['segment_starts']);ends=np.asarray(accepted['segment_ends'])
            progress(family,'original_canonical_segments')
            segments=reference._values(reference.reference.segment_intersect(starts,ends))
            if segments.tolist()!=accepted['segment_decisions']:raise AssertionError('Original canonical acceptance segments differ')
            progress(family,'original_canonical_boxes')
            boxes=[int(reference._values(reference.reference.classify_box(box))[0]) for box in accepted['box_bounds']]
            if boxes!=accepted['box_classes']:raise AssertionError('Original canonical acceptance boxes differ')
            from rind.geometry import Clearance
            progress(family,'original_canonical_clearance')
            clearance=reference._values(reference.reference.clearance_decide(source,DRIVER_RADIUS))
            if int(clearance[0])!=int(Clearance.GT):raise AssertionError('Original canonical source clearance differs')
            gpu=factory.source(bundle,source,'cuda_certified') if cuda else None;checks=[]
            try:
                for n in (32,64,128):
                    progress(family,'complete_field_comparison',resolution=n)
                    ay,am=small_field(a,n);by,bm=small_field(b,n)
                    if not np.array_equal(ay,by) or not np.array_equal(am,bm):raise AssertionError('Independent complete small fields differ')
                    if gpu:
                        gy,gm=cuda_field(gpu,n)
                        if not np.array_equal(ay,gy) or not np.array_equal(am,gm):raise AssertionError('CPU/CUDA complete small fields differ')
                    checks.append({'resolution':n,'voxels':n**3,'all_response_and_solid_bits_equal':True})
            finally:
                if gpu:gpu.close()
            cases.append({'family':family,'geometry_id':context.geometry_id,'original_canonical_points_and_rays':len(points),'original_canonical_segments':len(starts),'original_canonical_boxes':len(boxes),'original_canonical_clearance':True,
                          'checks':checks,'passed':True,'seconds':time.perf_counter()-start})
            atomic_json(path.with_suffix('.progress.json'),{'complete':False,'cases':cases})
        hard=hard_cases(factory);runtime=factory.report()
    value={'protocol':VERSION,'complete':True,'scientific_identity':identity,'families':list(FAMILIES),
           'driver_clearance_hex':DRIVER_RADIUS.hex(),'scope':'bounded regular closed convex unions strictly inside world',
           'basis':'exact structural/interior/world dual proofs + independent physical algorithms + original canonical fixtures',
           'cases':cases,'hard_cases_passed':hard,'cpu_native_certified':True,'cuda_certified':bool(cuda),
           'runtime':runtime,'seconds':time.perf_counter()-started}
    validate_certificate(value);atomic_json(path,value);return value
