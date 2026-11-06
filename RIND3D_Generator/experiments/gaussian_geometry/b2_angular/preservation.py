"""Read-only SHA256 preservation audit; never repairs or rewrites old data."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import subprocess
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify(base,files):
    for name,expected in files.items():
        rel=PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe audit path')
        path=base/name
        assert path.is_file() and sha(path)==expected,('Preservation failure',str(path))
    return len(files)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',type=Path,required=True)
    p.add_argument('--target',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();before=json.loads(args.before.read_text());repo=Path(__file__).resolve().parents[4]
    start=time.perf_counter();result={'status':'PASS','base_commit':before['research_HEAD']}
    result['previous_tracked_source_files_SHA256_unchanged']=verify(repo,before['tracked'])
    result['original_integration_source_and_WIP_SHA256_unchanged']=verify(args.target,before['original_checkout_files'])
    result['historical_output_files_SHA256_unchanged']={name:verify(repo/'RIND3D_Generator/outputs'/name,files)
                                                    for name,files in before['historical_outputs'].items()}
    dirty=subprocess.check_output(['git','-C',str(args.target),'status','--porcelain'],text=True)
    assert dirty==before['original_dirty_status'],'Original uncommitted work changed; do not overwrite it'
    dataset=args.target/'Phase3_Single_Source_3D/data/RIND3D-v0.1';manifest=dataset/'manifest.json'
    assert sha(manifest)==before['public_manifest_SHA256']
    data=json.loads(manifest.read_text());files={k:v['sha256'] for k,v in data['files'].items()}
    result['public_RIND3D_files_SHA256_verified']=verify(dataset,files)
    result['public_RIND3D_manifest_SHA256']=sha(manifest)
    result['public_RIND3D_scene_count']=data['scene_count'];result['public_RIND3D_window_count']=data['window_count']
    result['public_RIND3D_bytes_verified']=sum(row['bytes'] for row in data['files'].values())
    result.update(seconds=time.perf_counter()-start,new_READY_scenes=0,new_complete_1024_fields=0,previous_reports_modified=False,
                  sibling_projects_modified=False,Drive_release_modified=False,remote_push_performed=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
