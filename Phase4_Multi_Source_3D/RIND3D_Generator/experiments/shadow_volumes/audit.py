"""Recheck the mask-reuse build against all frozen smaller fields."""
import hashlib,json
from pathlib import Path
import numpy as np
from rind3d.geometry import load_geometry
from rind3d.pack import RINDPack,atomic_json
from .cases import NAMES,SOURCES
from .renderer import extract,shadow_planes,strings,native
from .benchmark import ROOT


def main():
    results=[]
    for name in NAMES:
        bundle=load_geometry(ROOT/'cases'/name/'geometry.npz');bodies,old=extract(bundle)
        model=native.Model(native.Geometry(strings(bodies),old),strings([shadow_planes(b,SOURCES[0]) for b in bodies]),SOURCES[0].tolist())
        for n in (32,64,128):
            path=ROOT/'small'/name/f'C-{n}';different=0
            with RINDPack(path/'response.rindb') as ry,RINDPack(path/'solid.rindb') as rm:
                side=min(n,64)
                for z in range(0,n,side):
                    for v in range(0,n,side):
                        for x in range(0,n,side):
                            y,m,_=model.raster([x,v,z],side,n,8,False)
                            assert np.all(y>=0) and np.all(m>=0),'Canonical fallback needed; use explicit full audit'
                            different+=int(np.sum(y!=ry.read_window(x,v,z,side)))+int(np.sum(m!=rm.read_window(x,v,z,side)))
            assert not different
            results.append({'family':name,'resolution':n,'all_voxels_equal':True,'differences':different})
    atomic_json(ROOT/'mask-reuse-build-audit.json',{'native_sha256':hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),'fields':results,'passed':True,'production_ready':False})
    print('Mask-reuse build matches all',len(results),'frozen fields bitwise')


if __name__=='__main__':main()
