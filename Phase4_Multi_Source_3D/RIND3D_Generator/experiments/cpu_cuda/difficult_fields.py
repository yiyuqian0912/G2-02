"""Complete small-field checks for the frozen difficult canonical cases."""
import time
import numpy as np
from rind3d.geometry import save_geometry,centres
from rind3d.pack import RINDPack,atomic_json
from .benchmark import ROOT,Control,field,compare
from .cpu import GeometryGroup
from .cuda_runtime import Runtime
from .gpu import GPUGroup
from .canonical_audit import difficult_cases


def main():
    results=[]
    with Runtime() as runtime:
        for name,bundle,source,points in difficult_cases():
            path=ROOT/'difficult-small'/name
            if (path/'result.json').exists():continue
            path.mkdir(parents=True,exist_ok=True);save_geometry(path/'geometry.npz',bundle)
            source=np.asarray(source,dtype='<f8');source.tofile(path/'source.private.f64')
            g=GeometryGroup(bundle);cpu=g.bind(source);ref=Control(g,source,'C_shadow');gg=GPUGroup(g,runtime);gpu=gg.bind(source)
            rows=[];checks=[]
            try:
                for n in (32,64,128):
                    for method,a in [('CPU_clip',cpu),('GPU',gpu)]:
                        r=field(a,path/f'{method}-{n}',n,method,60);assert r['complete'];rows.append(r)
                    compare(path/f'CPU_clip-{n}',path/f'GPU-{n}');t=time.perf_counter();count=0
                    with RINDPack(path/f'GPU-{n}/response.rindb') as y,RINDPack(path/f'GPU-{n}/solid.rindb') as m:
                        for z in range(0,n,16):
                            for v in range(0,n,16):
                                for x in range(0,n,16):
                                    p=centres((x,v,z),16,n);ry,rm,_=ref.native.points(p,True)
                                    assert np.array_equal(y.read_window(x,v,z,16).ravel(),ry)
                                    assert np.array_equal(m.read_window(x,v,z,16).ravel(),rm);count+=len(p)
                    checks.append({'resolution':n,'checked_voxels':count,'passed':True,'independent_reference_seconds':time.perf_counter()-t})
                result={'case':name,'geometry_id':g.geometry_id,'source':source.tolist(),'rows':rows,'checks':checks,
                        'all_response_and_solid_bits_equal':True,'production_ready':False};atomic_json(path/'result.json',result);results.append(result)
            finally:gpu.close();gg.close()
    atomic_json(ROOT/'difficult-small-summary.json',{'passed':True,'cases':[r['case'] for r in results],
                'note':'Complete 32³/64³/128³ fields, plus the separately recorded direct canonical CGAL targeted rays'})


if __name__=='__main__':main()
