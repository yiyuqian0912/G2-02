"""Rehearse interrupted publication on a COPY, preserving every original scene."""
import argparse,json,shutil,time
from pathlib import Path
from rind3d.cli import load_config
from rind3d.core_research import run
from rind3d.pack import atomic_json,sha256_file

def recover(source,config,output):
    source=Path(source);output=Path(output)
    if output.exists():raise FileExistsError('Use a fresh recovery rehearsal directory')
    before={str(p.relative_to(source)):sha256_file(p) for p in source.rglob('*') if p.is_file()}
    shutil.copytree(source,output);desc_path=output/'jobs'/'00000.json';desc=json.loads(desc_path.read_text());sid=desc['scene_id'];published=output/'scenes'/sid
    stage=output/'.staging'/'recovered-publication';published.rename(stage);desc['stage']=str(stage.resolve());atomic_json(desc_path,desc)
    t=time.perf_counter();r=run(load_config(config),output,10,workers=2,resume=True)
    after={str(p.relative_to(source)):sha256_file(p) for p in source.rglob('*') if p.is_file()}
    if before!=after:raise AssertionError('Recovery rehearsal changed the original pilot')
    recovered=output/'scenes'/sid
    original_hashes={p.name:sha256_file(p) for p in (source/'scenes'/sid).iterdir() if p.is_file()}
    hashes={p.name:sha256_file(p) for p in recovered.iterdir() if p.is_file()}
    if not r['complete'] or r['ready_scenes']!=10 or hashes!=original_hashes:raise AssertionError('Atomic publication recovery differed scientifically')
    atomic_json(output/'recovery-result.json',{'complete':True,'ready_scenes':10,'restored_ready':r['restored_ready_scenes'],'republished_same_identity':sid,
                                             'scientific_files_byte_identical':True,'original_pilot_unchanged':True,'wall_seconds':time.perf_counter()-t,
                                             'new_physical_evaluations':0,'reason':'Complete identity-bound dual physical evidence reused, not replaced by hashes.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();recover(a.source,a.config,a.output)
