"""Independent checks for the additional fixed sources in reuse experiments."""
import time
import numpy as np
from rind3d.geometry import Oracle,centres
from rind3d.pack import atomic_json
from .benchmark import ROOT,freeze
from .renderer import SharedGeometry
from .cases import SOURCES


def main():
    result=[];start=time.perf_counter()
    for name in ('rotated_box','poly16','multiscale4'):
        _,bundle=freeze(name);base=Oracle(bundle);shared=SharedGeometry(base)
        for rank in (1,2):
            a=shared.bind(SOURCES[rank]);rows=[]
            for n in (32,64,128):
                side=min(n,64);different=checked=0
                for z in range(0,n,side):
                    for v in range(0,n,side):
                        for x in range(0,n,side):
                            origin=(x,v,z);y,m,_=a.render(SOURCES[rank],origin,side,n);ry,rm,_=a.native.points(centres(origin,side,n),True)
                            different+=int(np.sum(y.ravel()!=ry))+int(np.sum(m.ravel()!=rm));checked+=side**3
                assert not different;rows.append({'resolution':n,'checked_voxels':checked,'differences':different})
            for origin in ((0,0,0),(960,0,0),(0,960,0),(0,0,960),(960,960,0),(960,0,960),(0,960,960),(960,960,960),(448,448,448),(576,448,384)):
                y,m,_=a.render(SOURCES[rank],origin,64,1024);ry,rm,_=a.native.points(centres(origin,64,1024),True)
                assert np.array_equal(y.ravel(),ry) and np.array_equal(m.ravel(),rm)
            p=np.random.default_rng(100+rank).uniform(.05,.95,(8,3));ref=base.reference.segment_intersect(np.repeat(SOURCES[rank:rank+1],8,axis=0),p)
            assert ref.valid.all() and np.array_equal(a.response(SOURCES[rank],p),ref.value==0)
            result.append({'family':name,'source_rank':rank,'source_id':a.source_id,'small_fields':rows,'stratified_1024_bricks':10,'canonical_rays':8,'all_equal':True})
            atomic_json(ROOT/'additional-source-audit-progress.json',result)
    atomic_json(ROOT/'additional-source-audit.json',{'pairs':result,'passed':True,'wall_seconds':time.perf_counter()-start,'scope':'Additional fixed sources; separate from original source-0 before/full-field gate'})


if __name__=='__main__':main()
