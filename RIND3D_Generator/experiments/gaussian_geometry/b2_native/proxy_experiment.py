"""Optional certified continuous inner/outer BOX covers, never geometry truth.

Only independent-reference acceleration is tested. Every excluded region is
proved FREE and every inner box is proved wholly SOLID by that verifier.
Unknown/pending regions remain in the outer cover. Exact convex B then screens
segments; undecided pairs still evaluate the original Gaussian sum.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from rind.contracts import CSGNode,CSGOpcode,Geometry,RootKind,make_leaf
from rind.storage import CanonicalBundle,pack_arrays,_decode_bundle
from rind3d.rendering.context import GeometryContext
from rind3d.geometry import Unsupported

from ..models import load
from ..queries import FieldOracle
from ..study import atomic_json
from .api import NativeOracle


def rectangles(mask):
    work=mask.copy();n=len(work);boxes=[]
    for z,y,x in np.ndindex(work.shape):
        if not work[z,y,x]:continue
        a=x+1
        while a<n and work[z,y,a]:a+=1
        b=y+1
        while b<n and work[z,b,x:a].all():b+=1
        c=z+1
        while c<n and work[c,y:b,x:a].all():c+=1
        work[z:c,y:b,x:a]=False;boxes.append(([x/n,y/n,z/n],[a/n,b/n,c/n]))
    return boxes


def build_covers(scene,depth=5,max_nodes=8192):
    start=time.perf_counter()
    if depth not in (3,4,5,6):raise ValueError('Bounded diagnostic cover depths: 3–6')
    if FieldOracle(scene).containment(.025)['status']!='CERTIFIED':raise ValueError('This frozen-input experiment requires proved global containment')
    engine=NativeOracle(scene,independent=True);n=2**depth
    inner=np.zeros((n,)*3,dtype=bool);outer=np.ones_like(inner);stack=[((0,0,0),n)];count=0
    while stack and count<max_nodes:
        origin,side=stack.pop();lo=np.array(origin)/n;hi=(np.array(origin)+side)/n
        d=engine.classify_box(lo,hi);count+=1;x,y,z=origin;region=(slice(z,z+side),slice(y,y+side),slice(x,x+side))
        if d.status=='FREE':outer[region]=False
        elif d.status=='SOLID':inner[region]=True
        elif side>1:
            h=side//2
            for dz in (0,h):
                for dy in (0,h):
                    for dx in (0,h):stack.append(((x+dx,y+dy,z+dz),h))
        # A terminal UNKNOWN and every unvisited pending region remain outer.
    a,b=rectangles(inner),rectangles(outer)
    return a,b,{'depth':depth,'region_queries':count,'pending_regions':len(stack),
                'inner_volume_lower_bound':float(inner.mean()),'outer_volume_upper_bound':float(outer.mean()),
                'inner_boxes':len(a),'outer_boxes':len(b),'build_seconds':time.perf_counter()-start,
                'work':dict(engine.work),'not_a_voxel_truth_representation':True}


def proxy_context(boxes):
    if not boxes:return None
    bundle=CanonicalBundle(3);refs=[]
    for lo,hi in boxes:
        lo,hi=np.array(lo),np.array(hi)
        # All endpoints are dyadic grid coordinates; midpoint/half-width are
        # represented exactly in binary64, so the codec does not change boxes.
        leaf=make_leaf('affine_box_v1',3,center=(lo+hi)/2,matrix=np.diag((hi-lo)/2))
        bundle.add_leaf(leaf);refs.append(leaf.content_hash)
    if len(refs)>1:
        node=CSGNode(3,CSGOpcode.UNION,tuple(refs));bundle.add_node(node);root=node.content_hash
    else:root=refs[0]
    bundle.add_geometry(Geometry(3,RootKind.REF,root))
    # These are auxiliary query bounds, not a scene/publication geometry. No
    # READY authority or geometry-family acceptance is constructed here.
    return GeometryContext(_decode_bundle(pack_arrays(bundle),3))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--queries',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);rows=[]
    for ordinal in (7,14):
        case=f'gaussian_sum-{ordinal:03d}';scene=load(args.input/case/'geometry.json');queries=np.load(args.queries/case/'queries.npz')
        source=queries['source'];points=queries['receivers'];ref=NativeOracle(scene,independent=True)
        assert ref.source_clearance(source).status=='FREE'
        t=time.perf_counter();expected=ref.rays(source,points);base=time.perf_counter()-t
        for depth in (3,4):
            inner,outer,meta=build_covers(scene,depth);t=time.perf_counter()
            try:inside=proxy_context(inner);outside=proxy_context(outer)
            except Unsupported as error:
                row={'case':case,'index':meta,'status':'INCOMPLETE_PROXY_RESOURCE_CAP','reason':str(error),
                     'claim':'existing 64-component capability is preserved; this is not a physical rejection or a bypass'}
                rows.append(row);atomic_json(args.output/f'{case}-depth-{depth}.json',row);continue
            if inside is not None and inside.contains(source[None])[0]:raise AssertionError('FREE source inside certified SOLID proxy')
            inner_model=inside.bind(source,independent=True) if inside is not None else None
            source_in_outer=bool(outside is not None and outside.contains(source[None])[0])
            outer_model=outside.bind(source,independent=True) if outside is not None and not source_in_outer else None
            setup=time.perf_counter()-t;t=time.perf_counter()
            blocked=np.zeros(len(points),dtype=bool) if inner_model is None else ~inner_model.response(source,points)
            visible=np.ones(len(points),dtype=bool) if outside is None else np.zeros(len(points),dtype=bool) if outer_model is None else outer_model.response(source,points)
            if np.any(blocked&visible):raise AssertionError('Continuous bounding-set contradiction')
            unknown=~(blocked|visible);fallback=ref.rays(source,points[unknown]);labels=np.full(len(points),'UNKNOWN',dtype='<U12')
            labels[blocked]='HIT';labels[visible]='MISS';labels[unknown]=[d.status for d in fallback]
            disagreements=sum(d.certified and label in ('HIT','MISS') and label!=d.status for d,label in zip(expected,labels))
            assert not disagreements
            warm=time.perf_counter()-t
            row={'case':case,'index':meta,'reference_baseline_seconds':base,'proxy_setup_seconds':setup,
                 'status':'MATCHED_SEGMENT_SCREENING_COMPLETE',
                 'warm_screen_and_fallback_seconds':warm,'cold_total_seconds':meta['build_seconds']+setup+warm,
                 'rays':len(points),'inner_HIT':int(blocked.sum()),'outer_MISS':int(visible.sum()),'Gaussian_fallbacks':int(unknown.sum()),
                 'source_in_outer_bound':source_in_outer,'certified_disagreements':int(disagreements),
                 'claim':'only matched segment screening measured; no hybrid full-field throughput or READY claim'}
            rows.append(row);atomic_json(args.output/f'{case}-depth-{depth}.json',row)
            print(json.dumps({'case':case,'depth':depth,'bounds_boxes':[len(inner),len(outer)],'fallbacks':int(unknown.sum()),'cold':row['cold_total_seconds'],'warm':warm}),flush=True)
    atomic_json(args.output/'summary.json',{'cases':rows,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'production':False})


if __name__=='__main__':main()
