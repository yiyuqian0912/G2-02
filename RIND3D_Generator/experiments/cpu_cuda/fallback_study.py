"""Explicit certified-native CPU fallback study; canonical default is unchanged."""
import json
import time
import numpy as np
from rind3d.geometry import centres,save_geometry
from rind3d.pack import RINDPack,atomic_json
from .benchmark import ROOT,Control,field,compare
from .cpu import GeometryGroup
from .cuda_runtime import Runtime
from .gpu import GPUGroup
from .canonical_audit import difficult_cases


def independent(path,n,reference):
    start=time.perf_counter();count=0
    with RINDPack(path/'response.rindb',verify_file=True) as y,RINDPack(path/'solid.rindb',verify_file=True) as m:
        for z in range(0,n,16):
            for v in range(0,n,16):
                for x in range(0,n,16):
                    p=centres((x,v,z),16,n);ry,rm,_=reference.native.points(p,True)
                    assert np.array_equal(y.read_window(x,v,z,16).ravel(),ry)
                    assert np.array_equal(m.read_window(x,v,z,16).ravel(),rm);count+=len(p)
    return {'checked_voxels':count,'passed':True,'seconds':time.perf_counter()-start,'reference':'unchanged independent interval/GMP clipping'}


def main():
    root=ROOT/'certified-native-fallback';root.mkdir(parents=True,exist_ok=True)
    with Runtime() as r:
        for name,bundle,source,_ in difficult_cases():
            out=root/name
            if (out/'result.json').exists():continue
            out.mkdir(parents=True,exist_ok=True);save_geometry(out/'geometry.npz',bundle)
            source=np.asarray(source,dtype='<f8');source.tofile(out/'source.private.f64')
            g=GeometryGroup(bundle);cpu=g.bind(source);reference=Control(g,source,'C_shadow');gg=GPUGroup(g,r)
            gpu=gg.bind(source,fallback_backend='certified_native');gpu.attach_independent_reference(reference)
            rows=[];checks=[]
            try:
                for n in (32,64,128):
                    for method,a in [('CPU_clip',cpu),('GPU',gpu)]:
                        row=field(a,out/f'{method}-{n}',n,method,60);assert row['complete'];rows.append(row)
                    compare(out/f'CPU_clip-{n}',out/f'GPU-{n}')
                    checks.append(dict(resolution=n,**independent(out/f'GPU-{n}',n,reference)))
                    previous=ROOT/'difficult-small'/name/f'GPU-{n}'
                    if (previous/'result.json').exists():
                        compare(out/f'GPU-{n}',previous)
                        checks[-1]['all_bits_equal_original_cgal_fallback_field']=True
                    print(name,n,'all bits PASS',flush=True)
                assert g._oracle is None,'Certified scope must not silently construct/claim CGAL'
                atomic_json(out/'result.json',{'case':name,'geometry_id':g.geometry_id,'source':source.tolist(),
                            'rows':rows,'checks':checks,'complete_small_correctness_passed':True,
                            'fallback_backend':'certified_native','every_fallback_point_independently_rechecked':True,'production_ready':False})
            finally:gpu.close();gg.close()
        # A difficult, bounded full-field experiment only after all small gates.
        name,bundle,source,_=difficult_cases()[1];g=GeometryGroup(bundle);source=np.asarray(source)
        cpu=g.bind(source);reference=Control(g,source,'C_shadow');gg=GPUGroup(g,r)
        gpu=gg.bind(source,fallback_backend='certified_native');gpu.attach_independent_reference(reference)
        path=root/'full-nearly-coplanar';records=[]
        try:
            rng=np.random.default_rng(902);origins=[]
            for x in (0,8):
                for y in (0,8):
                    for z in (0,8):origins.append(tuple((np.array([x,y,z])+rng.integers(0,8,3))*64))
            origins.append((391,477,503))
            for origin in origins:
                p=centres(origin,64,1024);ry,rm,_=reference.native.points(p,True)
                y,m,_=cpu.render(source,origin,64,1024);pair=gpu.packed_batch([origin],64,1024)[0]
                assert np.array_equal(y.ravel(),ry) and np.array_equal(m.ravel(),rm)
                for raw,v in zip(pair,(ry,rm)):assert np.array_equal(np.unpackbits(np.frombuffer(raw,dtype=np.uint8),bitorder='little'),v)
                records.append({'origin':list(map(int,origin)),'all_bits_equal_independent_reference':True})
            path.mkdir(parents=True,exist_ok=True);atomic_json(path/'stratified.json',{'passed':True,'bricks':records})
            a=field(cpu,path/'CPU_clip',1024,'CPU_clip',90);assert a['complete']
            b=field(gpu,path/'GPU-with-independent-fallback-check',1024,'GPU',90);assert b['complete']
            compare(path/'CPU_clip',path/'GPU-with-independent-fallback-check')
            # Timed filter without the extra experimental per-point checker.
            # Its entire output is then checked against the audited field.
            gpu.independent_reference=None
            c=field(gpu,path/'GPU',1024,'GPU',90);assert c['complete']
            compare(path/'CPU_clip',path/'GPU');compare(path/'GPU',path/'GPU-with-independent-fallback-check')
            assert g._oracle is None
            atomic_json(path/'result.json',{'case':name,'resolution':1024,'complete':True,'all_response_and_solid_bits_equal':True,
                        'CPU_clip':a,'GPU_with_independent_fallback_checks':b,'GPU':c,'production_ready':False,
                        'scope':'Exact convex predicates only; original CGAL default retained and READY gates unchanged'})
        finally:gpu.close();gg.close()
    atomic_json(root/'result.json',{'complete':True,'small_cases':3,'small_resolutions':[32,64,128],
                'bounded_full_field_complete':True,'original_cgal_default_changed':False,'production_ready':False})


if __name__=='__main__':main()
