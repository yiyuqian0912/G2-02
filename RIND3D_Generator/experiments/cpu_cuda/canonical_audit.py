"""Targeted original CGAL checks and an honest bounded exhaustive attempt."""
import argparse
import time
import numpy as np
from rind.contracts import make_leaf
from rind3d.geometry import Oracle,centres
from rind3d.pack import atomic_json
from experiments.shadow_volumes.difficult import bundle
from .cpu import GeometryGroup
from .benchmark import ROOT,frozen


def difficult_cases():
    normals=np.r_[np.eye(3),-np.eye(3),[[1,2**-35,0]]]
    return [
        ('near_source',bundle(make_leaf('affine_box_v1',3,center=[.5]*3,matrix=np.diag([.125]*3))),
         [np.nextafter(.375,0.),.5,.5],[[.9,.5,.5],[.1,.5,.5],[.375,.625,.5],[.375,np.nextafter(.625,1.),.5]]),
        ('nearly_coplanar',bundle(make_leaf('convex_halfspaces_v1',3,center=[.5]*3,matrix=np.diag([.125]*3),normal=normals,plane_offset=[1.]*6+[1+2**-36])),
         [.1,.3,.6],[[.625,.625,.5],[.625,.5,.5],[.625,np.nextafter(.5625,0.),.5],[.625,np.nextafter(.5625,1.),.5]]),
        ('very_thin',bundle(make_leaf('affine_box_v1',3,center=[.5]*3,matrix=np.diag([.2,.2,2**-40]))),
         [.1,.5,.5+2**-40],[[.9,.5,.5+2**-40],[.9,.5,np.nextafter(.5+2**-40,1.)],[.9,.5,.5],[.9,.5,.5+2**-30]])]


def targeted():
    from .cuda_runtime import Runtime
    from .gpu import GPUGroup
    rows=[]
    with Runtime() as runtime:
        for name,b,source,points in difficult_cases():
            s=np.asarray(source);p=np.asarray(points);start=time.perf_counter();g=GeometryGroup(b);a=g.bind(s)
            base=g.canonical;ry=base._values(base.reference.segment_intersect(np.repeat(s[None],len(p),axis=0),p))==0
            rm=base._values(base.reference.classify_points(p))!=0
            cy,cm,_=a.native.points(p);assert np.array_equal(cy,ry) and np.array_equal(cm,rm)
            gg=GPUGroup(g,runtime);gpu=gg.bind(s)
            try:
                unresolved=gpu.point_labels(p,resolve=False);y,m=gpu.point_labels(p)
                assert np.array_equal(y,ry) and np.array_equal(m,rm)
                work=gpu.report()
            finally:gpu.close();gg.close()
            rows.append({'case':name,'geometry_id':g.geometry_id,'source':s.tolist(),'receivers':p.tolist(),
                         'canonical_response':ry.tolist(),'canonical_solid':rm.tolist(),'all_equal':True,
                         'unresolved_filter_response':int(np.sum(unresolved[0]<0)),'gpu_work':work,'wall_seconds':time.perf_counter()-start})
    atomic_json(ROOT/'targeted-canonical-audit.json',{'passed':True,'cases':rows})


def exhaustive(seconds=12):
    # Explicitly bounded attempt; counts receiver mask and ray decisions, not
    # "complete" merely because a partial comparison had no differences.
    from experiments.shadow_volumes.cases import SOURCES
    g=GeometryGroup(frozen('box'));a=g.bind(SOURCES[0]);base=g.canonical
    p=centres((0,0,0),32,32);start=time.perf_counter();count=0;rows=[]
    for i in range(0,len(p),64):
        if time.perf_counter()-start>seconds:break
        points=p[i:i+64];ry=base._values(base.reference.segment_intersect(np.repeat(a.source[None],len(points),axis=0),points))==0
        rm=base._values(base.reference.classify_points(points))!=0
        cy,cm,_=a.native.points(points);assert np.array_equal(cy,ry) and np.array_equal(cm,rm)
        count+=len(points)
    atomic_json(ROOT/'exhaustive-cgal-attempt.json',{'resolution':32,'declared_seconds':seconds,'checked_voxels':count,
                'required_voxels':len(p),'complete':count==len(p),'differences':0,'wall_seconds':time.perf_counter()-start,
                'termination_reason':'complete' if count==len(p) else 'declared_time_budget','canonical_geometry_setup_seconds':g.initialization_seconds})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['targeted','exhaustive']);a=p.parse_args()
    targeted() if a.stage=='targeted' else exhaustive()
