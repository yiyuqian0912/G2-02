"""Targeted canonical CGAL audit of close sources and near-degenerate solids."""
import time
import numpy as np
from rind.contracts import make_leaf,Geometry,RootKind
from rind.storage import CanonicalBundle
from rind3d.geometry import Oracle
from rind3d.pack import atomic_json
from .benchmark import ROOT
from .renderer import SharedGeometry


def bundle(leaf):
    b=CanonicalBundle(3);b.add_leaf(leaf);b.add_geometry(Geometry(3,RootKind.REF,leaf.content_hash));return b


def main():
    records=[]
    normals=np.r_[np.eye(3),-np.eye(3),[[1,2**-35,0]]];offsets=[1.]*6+[1+2**-36]
    leaves=[('near_source',make_leaf('affine_box_v1',3,center=[.5]*3,matrix=np.diag([.125]*3))),
        ('nearly_coplanar',make_leaf('convex_halfspaces_v1',3,center=[.5]*3,matrix=np.diag([.125]*3),normal=normals,plane_offset=offsets)),
        ('very_thin',make_leaf('affine_box_v1',3,center=[.5]*3,matrix=np.diag([.2,.2,2**-40])))]
    for name,leaf in leaves:
        base=Oracle(bundle(leaf));shared=SharedGeometry(base)
        if name=='near_source':s=[np.nextafter(.375,0.),.5,.5];p=[[.9,.5,.5],[.1,.5,.5],[.375,.625,.5],[.375,np.nextafter(.625,1.),.5]]
        elif name=='nearly_coplanar':s=[.1,.3,.6];p=[[.625,.625,.5],[.625,.5,.5],[.625,np.nextafter(.5625,0.),.5],[.625,np.nextafter(.5625,1.),.5]]
        else:s=[.1,.5,.5+2**-40];p=[[.9,.5,.5+2**-40],[.9,.5,np.nextafter(.5+2**-40,1.)],[.9,.5,.5],[.9,.5,.5+2**-30]]
        s=np.array(s);p=np.array(p);t=time.perf_counter();a=shared.bind(s)
        r=base.reference.segment_intersect(np.repeat(s[None],len(p),axis=0),p);m=base.reference.classify_points(p)
        assert r.valid.all() and m.valid.all();y=a.response(s,p);solid=a.contains(p)
        assert np.array_equal(y,r.value==0) and np.array_equal(solid,m.value!=0)
        records.append({'case':name,'geometry_id':shared.geometry_id,'source':s.tolist(),'receivers':p.tolist(),'canonical_response':(r.value==0).tolist(),'canonical_solid':(m.value!=0).tolist(),'all_equal':True,'wall_seconds':time.perf_counter()-t,'fallback_work':a.report()['work']})
    atomic_json(ROOT/'difficult-canonical-audit.json',{'passed':True,'cases':records})


if __name__=='__main__':main()
