"""Exact diagnostic vertices/facet counts; never replaces canonical geometry."""
from fractions import Fraction as F
from itertools import combinations
import json,time
import numpy as np
from rind3d.geometry import load_geometry
from rind3d.pack import atomic_json
from .renderer import extract,normalize
from .benchmark import ROOT
from .cases import NAMES,SOURCES


def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def vertices(planes):
    result=set()
    for a,b,c in combinations(planes,3):
        bc=cross(b[:3],c[:3]);ca=cross(c[:3],a[:3]);ab=cross(a[:3],b[:3]);det=dot(a[:3],bc)
        if not det:continue
        p=tuple((-a[3]*bc[i]-b[3]*ca[i]-c[3]*ab[i])/det for i in range(3))
        if all(dot(r[:3],p)+r[3]<=0 for r in planes):result.add(p)
    facets=0
    for row in planes:
        points=[p for p in result if dot(row[:3],p)+row[3]==0]
        if len(points)<3:continue
        origin=points[0];u=tuple(x-y for x,y in zip(points[1],origin))
        if any(any(cross(u,tuple(x-y for x,y in zip(p,origin)))) for p in points[2:]):facets+=1
    return sorted(result),facets


def main():
    t=time.perf_counter();result=[]
    world=[]
    for axis in range(3):
        for sign in (-1,1):world.append(tuple([F(sign*int(k==axis)) for k in range(3)]+[F(-1 if sign>0 else 0)]))
    for name in NAMES:
        case=ROOT/'cases'/name;bundle=load_geometry(case/'geometry.npz');bodies,_=extract(bundle);meta=json.loads((case/'case.json').read_text());objects=[]
        for body,obj in zip(bodies,meta['objects']):
            planes=list(dict.fromkeys([*body,*world]));v,faces=vertices(planes)
            matrix=np.asarray(obj['matrix']);sv=np.linalg.svd(matrix,compute_uv=False);orientation=float(np.degrees(np.arccos(np.clip(np.abs(matrix[:,0]).max()/np.linalg.norm(matrix[:,0]),0,1))))
            objects.append({'canonical_input_planes':len(body),'actual_clipped_facets':faces,'exact_vertex_count':len(v),
                'vertices_rational':[[str(x) for x in p] for p in v],'vertices_for_plot':[[float(x) for x in p] for p in v],
                'affine_singular_values':sv.tolist(),'aspect_ratio':float(sv.max()/sv.min()),'first_axis_angle_to_nearest_world_axis_degrees':orientation,
                'source_center_distances':np.linalg.norm(SOURCES-np.asarray(obj['center']),axis=1).tolist()})
        result.append({'family':name,'geometry_id':bundle.geometries[0].content_hash.hex(),'objects':objects,'obstacle_count':len(objects)})
        atomic_json(ROOT/'diversity-statistics-progress.json',result)
    atomic_json(ROOT/'diversity-statistics.json',{'scenes':result,'method':'Exact rational triple-plane enumeration including world clipping; only plotted coordinates are rounded','elapsed_seconds':time.perf_counter()-t,'geometry_replaced_by_mesh':False})


if __name__=='__main__':main()
