"""Stored-field CLI operations; selection output is never READY publication."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from .pack import RINDPack,atomic_json
from .information import InformationIndex
from .stored_windows import select_stored_windows


def build_index(response,solid,output):
    with RINDPack(response) as r,RINDPack(solid) as m:
        index=InformationIndex.build(r,m);size=index.save(output)
        return {**index.build_report,'path':str(output),'persistent_bytes':size,'identity':r.identity}


def select_scene(scene,output,cfg,count=4096,*,index_path=None,diagnostic=False,allow_synthetic=False):
    scene=Path(scene);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source=np.fromfile(scene/'driver.f64',dtype='<f8');geometry_path=scene/'geometry.npz'
    preparation=json.loads((scene/'preparation.json').read_text()) if (scene/'preparation.json').exists() else {}
    if not geometry_path.exists() and preparation.get('geometry'):geometry_path=Path(preparation['geometry'])
    start=time.perf_counter();cpu=time.process_time()
    with RINDPack(scene/'response.rindb') as r,RINDPack(scene/'solid.rindb') as m:
        if r.shape!=(1024,)*3 and not diagnostic:raise ValueError('Non-1024 fields require explicit --diagnostic')
        if geometry_path.exists():
            from .geometry import load_geometry
            geometry=load_geometry(geometry_path)
            identity=hashlib.sha256(geometry.geometries[0].content_hash+source.astype('<f8').tobytes()+bytes.fromhex(r.policy)).hexdigest()
            if r.identity!=identity:raise ValueError('True source/canonical geometry do not match the stored field')
            binding='canonical geometry/source/field identity verified'
        elif not allow_synthetic:raise ValueError('Canonical source binding is missing; only explicit --allow-synthetic permits test fields')
        else:binding='EXPLICIT SYNTHETIC/UNBOUND INPUT: never a published dataset'
        if index_path:
            index=InformationIndex.load(index_path,r,m);index_report={'reused':True,'index_load_and_verification_wall':time.perf_counter()-start,'persistent_bytes':Path(index_path).stat().st_size}
        else:
            index=InformationIndex.build(r,m);size=index.save(output/'information.rindi');index_report={**index.build_report,'persistent_bytes':size,'reused':False}
        result=select_stored_windows(index,source,count,seed=cfg['seed'],scene=0,settings=cfg.get('window_selection'),output=output)
        report={**result.report,'input_scene':str(scene),'input_binding':binding,'index':index_report,
                'end_to_end_wall_seconds':time.perf_counter()-start,'end_to_end_process_seconds':time.process_time()-cpu,
                'publication_attempted':False,'production_ready':False}
        atomic_json(output/'selection.json',report);return report
